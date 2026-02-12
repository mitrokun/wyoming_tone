import argparse
import asyncio
import logging
from typing import Optional, Any, List

import numpy as np
import soxr

from tone import StreamingCTCPipeline
from wyoming.info import Info
from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.error import Error
from wyoming.info import Describe
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

INCOMING_SAMPLE_RATE = 16000
MODEL_SAMPLE_RATE = 8000
# T-one требует 2400 сэмплов на частоте 8кГц (300 мс)
# REQUIRED_SAMPLES = 2400 * (INCOMING_SAMPLE_RATE/MODEL_SAMPLE_RATE)
REQUIRED_SAMPLES = 4800 

VAD_SILENCE_THRESHOLD_RATIO = 0.3
VAD_PATIENCE_CHUNKS = 5


class StreamAGC:
    """
    Dynamic AGC адаптированный под потоковую обработку.
    """
    def __init__(self, target_level=0.6, max_gain=30.0, min_gain=1.0):
        self.target_level = target_level
        self.max_gain = max_gain
        self.min_gain = min_gain
        
        # Калибровка
        self.calib_frames = 12 
        self.calib_peak = 0.0
        self.is_calibrated = False
        
        # Порог определения "умного" микрофона (0.1 = -20dB; 0.031 = -30dB)
        self.loud_threshold = 0.03
        
        # Параметры динамики (как в шерпа)
        self.current_peak_envelope = target_level 
        self.active_max_gain = 1.0 # По умолчанию усиление выключено

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) == 0:
            return audio_chunk

        chunk_max = np.max(np.abs(audio_chunk))

        # --- ЭТАП 1: КАЛИБРОВКА ---
        if not self.is_calibrated:
            self.calib_peak = max(self.calib_peak, chunk_max)
            self.calib_frames -= 1
            
            if self.calib_frames <= 0 or self.calib_peak > self.loud_threshold:
                self._finalize_calibration()
            
            return audio_chunk

        # --- ЭТАП 2: РАБОЧИЙ РЕЖИМ ---
        
        # Если определили, что микрофон и так громкий — ничего не делаем
        if self.active_max_gain <= 1.0:
            return np.clip(audio_chunk, -1.0, 1.0)

        # Подстройка
        # Fast Attack (0.5) / Slow Decay (0.01)
        alpha = 0.5 if chunk_max > self.current_peak_envelope else 0.01
        self.current_peak_envelope = (1 - alpha) * self.current_peak_envelope + alpha * chunk_max
        
        # Расчет текущего усиления
        safe_envelope = max(self.current_peak_envelope, 1e-6)
        target_gain = self.target_level / safe_envelope
        
        # Ограничиваем усиление выбранным максимумом (30.0)
        final_gain = np.clip(target_gain, self.min_gain, self.active_max_gain)
        
        # Применяем и мягко лимитируем
        return np.tanh(audio_chunk * final_gain)

    def _finalize_calibration(self):
        self.is_calibrated = True
        
        if self.calib_peak > self.loud_threshold:
            # Устройство само справляется с громкостью
            self.active_max_gain = 1.0
            _LOGGER.info(f"AGC: Hardware DSP detected (Peak={self.calib_peak:.2f}). Dynamic AGC OFF.")
        else:
            # Устройство тихое, включаем динамический режим
            self.active_max_gain = self.max_gain
            # Устанавливаем начальную огибающую пониже, 
            # чтобы сразу начать с хорошего усиления
            self.current_peak_envelope = max(self.calib_peak, 0.05)
            
            _LOGGER.info(f"AGC: Raw mic detected (Peak={self.calib_peak:.2f}). Dynamic AGC ON (Max x{self.max_gain}).")


class ToneEventHandler(AsyncEventHandler):
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        pipeline: StreamingCTCPipeline,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.cli_args = cli_args
        self.pipeline = pipeline
        self.language = self.cli_args.language
        self.state: Optional[Any] = None
        self.accumulated_text: str = ""
        
        # Буфер float32 сэмплов
        self.sample_buffer: List[float] = [] 
        
        self.vad_peak_energy: float = 0.0
        self.vad_quiet_chunks: int = 0
        self.vad_triggered: bool = False
        self.is_done = False
        
        if self.cli_args.agc:
             self.agc = StreamAGC(target_level=0.6, max_gain=30.0)
        else:
             self.agc = None

        _LOGGER.debug("Event handler initialized")

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True
        if AudioStart.is_type(event.type):
            await self._handle_audio_start()
            return True
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            await self._handle_audio_chunk(chunk.audio)
            return True
        if AudioStop.is_type(event.type):
            if not self.vad_triggered:
                await self._handle_audio_stop()
            return False
        if Error.is_type(event.type):
            _LOGGER.error("Received error from client: %s", event.text)
        return True

    async def _handle_audio_start(self) -> None:
        _LOGGER.debug("Audio stream started.")
        self.state = None
        self.accumulated_text = ""
        self.sample_buffer = [] 
        self.vad_peak_energy = 0.0
        self.vad_quiet_chunks = 0
        self.vad_triggered = False
        self.is_done = False
        
        if self.cli_args.agc:
            self.agc = StreamAGC(target_level=0.6, max_gain=30.0)
        
        await self.write_event(TranscriptStart(language=self.language).event())

    async def _handle_audio_chunk(self, audio_chunk_bytes: bytes) -> None:
        if self.is_done:
            return

        # 1. Байты -> Float (-1.0 ... 1.0)
        samples_float = np.frombuffer(audio_chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. AGC (применяется сразу)
        if self.agc:
            samples_float = self.agc.process(samples_float)
        
        # 3. Добавляем в буфер
        self.sample_buffer.extend(samples_float)

        try:
            # 4. Проверяем количество СЭМПЛОВ
            while len(self.sample_buffer) >= REQUIRED_SAMPLES: 
                if self.is_done: return
                
                # Забираем кусок нужного размера
                chunk_to_process = np.array(self.sample_buffer[:REQUIRED_SAMPLES], dtype=np.float32)
                
                # Удаляем из буфера
                self.sample_buffer = self.sample_buffer[REQUIRED_SAMPLES:]
                
                # Отправляем на инференс
                await self._process_chunk(chunk_to_process)
                
        except Exception as e:
            _LOGGER.exception("Error processing audio chunk")
            await self.write_event(Error(text=str(e)).event())

    async def _process_chunk(self, samples_float: np.ndarray):
        """
        samples_float: массив float32 (-1.0 ... 1.0)
        """
        if self.is_done:
            return

        # 1. Возвращаем масштаб амплитуды +/- 32767 для VAD и Soxr
        samples_scaled = samples_float * 32767.0

        # 2. VAD Logic
        rms_energy = np.sqrt(np.mean(samples_scaled**2))
        self.vad_peak_energy = max(self.vad_peak_energy, rms_energy)
        
        is_quiet = False
        if self.vad_peak_energy > 0:
            is_quiet = rms_energy < self.vad_peak_energy * VAD_SILENCE_THRESHOLD_RATIO
        
        if is_quiet:
            self.vad_quiet_chunks += 1
        else:
            self.vad_quiet_chunks = 0
            
        if self.vad_quiet_chunks >= VAD_PATIENCE_CHUNKS:
            if not self.vad_triggered:
                self.vad_triggered = True
                _LOGGER.debug("VAD triggered. Forcing end of speech.")
                asyncio.create_task(self._handle_audio_stop())
            return

        # 3. Resampling (soxr)
        # Soxr работает с float, но ожидает масштаб входных данных, соответствующий выходному
        resampled = soxr.resample(
            samples_scaled, INCOMING_SAMPLE_RATE, MODEL_SAMPLE_RATE
        )
        
        np.clip(resampled, -32768, 32767, out=resampled)
        samples_int32 = resampled.astype(np.int32)
        
        # 4. ASR Inference
        new_phrases, self.state = self.pipeline.forward(samples_int32, self.state)
        if new_phrases:
            chunk_text = " ".join(p.text for p in new_phrases if p.text)
            if chunk_text:
                _LOGGER.debug("New phrases received: '%s'", chunk_text)
                await self.write_event(TranscriptChunk(text=chunk_text).event())
                self.accumulated_text = (self.accumulated_text + " " + chunk_text).strip()

    async def _handle_audio_stop(self) -> None:
        if self.is_done:
            return
        self.vad_triggered = True 
        _LOGGER.debug("End of audio stream.")
        try:
            # Обработка хвоста
            if self.sample_buffer:
                # Паддинг нулями до REQUIRED_SAMPLES
                padding_len = REQUIRED_SAMPLES - len(self.sample_buffer)
                tail = np.array(self.sample_buffer, dtype=np.float32)
                if padding_len > 0:
                    tail = np.pad(tail, (0, padding_len), 'constant')
                
                await self._process_chunk(tail)
                self.sample_buffer = []

            final_phrases, _ = self.pipeline.finalize(self.state)
            if final_phrases:
                final_text_part = " ".join(p.text for p in final_phrases if p.text)
                if final_text_part:
                     await self.write_event(TranscriptChunk(text=final_text_part).event())
                     self.accumulated_text = (self.accumulated_text + " " + final_text_part).strip()
            
            await self._finalize_recognition(self.accumulated_text)
        except Exception as e:
            _LOGGER.exception("Error during finalization")
            await self.write_event(Error(text=str(e)).event())

    async def _finalize_recognition(self, text: str) -> None:
        if self.is_done: return
        final_text = text.strip()
        _LOGGER.info("Final result: '%s'", final_text)
        await self.write_event(Transcript(text=final_text if final_text else "").event())
        await self.write_event(TranscriptStop().event())

        self.is_done = True
