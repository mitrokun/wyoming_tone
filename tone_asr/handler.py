import argparse
import asyncio
import logging
from typing import Optional, Any

import numpy as np
import resampy

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
REQUIRED_SAMPLES = 4800
REQUIRED_BYTES = REQUIRED_SAMPLES * 2

VAD_SILENCE_THRESHOLD_RATIO = 0.3
VAD_PATIENCE_CHUNKS = 5


class StreamAGC:
    """
    Simple Automatic Gain Control with Auto-Calibration.
    Detects if the source is already normalized (DSP) or quiet (raw mic).
    """
    def __init__(self, target_level=0.6, max_gain=30.0, min_gain=1.0):
        self.target_level = target_level
        self.absolute_max_gain = max_gain
        self.min_gain = min_gain
        
        self.current_peak_envelope = target_level 
        self.calib_peak = 0.0
        self.calib_frames = 12
        self.is_calibrated = False
        self.dsp_threshold = 0.031 
        self.active_max_gain = 1.0 

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) == 0:
            return audio_chunk

        chunk_max = np.max(np.abs(audio_chunk))

        if self.calib_frames > 0:
            self.calib_peak = max(self.calib_peak, chunk_max)
            self.calib_frames -= 1
            return audio_chunk

        if not self.is_calibrated:
            if self.calib_peak > self.dsp_threshold:
                self.active_max_gain = 1.0
            else:
                self.active_max_gain = self.absolute_max_gain
                self.current_peak_envelope = 0.1
            self.is_calibrated = True

        if self.active_max_gain <= 1.0:
            return np.clip(audio_chunk, -1.0, 1.0)

        alpha = 0.5 if chunk_max > self.current_peak_envelope else 0.01
        self.current_peak_envelope = (1 - alpha) * self.current_peak_envelope + alpha * chunk_max
        safe_envelope = max(self.current_peak_envelope, 1e-6)
        target_gain = self.target_level / safe_envelope
        final_gain = np.clip(target_gain, self.min_gain, self.active_max_gain)
        return np.tanh(audio_chunk * final_gain)


class ToneEventHandler(AsyncEventHandler):
    """Event handler for each client using T-one with a custom VAD and AGC."""

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
        self.audio_buffer = bytearray()
        
        self.vad_peak_energy: float = 0.0
        self.vad_quiet_chunks: int = 0
        self.vad_triggered: bool = False
        self.is_done = False
        
        # Initialize AGC
        self.agc = StreamAGC(target_level=0.7, max_gain=30.0)

        _LOGGER.debug("Event handler initialized")

    async def _process_chunk(self, chunk_bytes: bytes):
        """Processes a chunk of audio, including VAD logic and AGC."""
        if self.is_done:
            return

        # 1. Convert bytes to float (raw amplitude +/- 32768)
        samples_raw = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)

        # 2. VAD Logic (Running on raw data to avoid triggering on amplified noise)
        rms_energy = np.sqrt(np.mean(samples_raw**2))
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

        # 3. AGC Logic
        # Normalize to -1.0...1.0 for AGC processing
        samples_norm = samples_raw / 32768.0
        # Apply AGC
        samples_norm = self.agc.process(samples_norm)
        # Scale back to float amplitude for resampling
        samples_float = samples_norm * 32767.0

        # 4. Resampling
        resampled_samples_float = resampy.resample(
            samples_float, INCOMING_SAMPLE_RATE, MODEL_SAMPLE_RATE
        )
        np.clip(resampled_samples_float, -32768, 32767, out=resampled_samples_float)
        samples_int32 = resampled_samples_float.astype(np.int32)
        
        # 5. ASR Inference
        new_phrases, self.state = self.pipeline.forward(samples_int32, self.state)
        if new_phrases:
            chunk_text = " ".join(p.text for p in new_phrases if p.text)
            if chunk_text:
                _LOGGER.debug("New phrases received: '%s'", chunk_text)
                await self.write_event(TranscriptChunk(text=chunk_text).event())
                self.accumulated_text = (self.accumulated_text + " " + chunk_text).strip()

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
        _LOGGER.debug("Audio stream started. Resetting ASR, VAD and AGC state.")
        self.state = None
        self.accumulated_text = ""
        self.audio_buffer.clear()
        self.vad_peak_energy = 0.0
        self.vad_quiet_chunks = 0
        self.vad_triggered = False
        self.is_done = False
        # Reset AGC for new stream
        self.agc = StreamAGC(target_level=0.7, max_gain=30.0)
        await self.write_event(TranscriptStart(language=self.language).event())

    async def _handle_audio_chunk(self, audio_chunk_bytes: bytes) -> None:
        if self.is_done:
            return
        self.audio_buffer.extend(audio_chunk_bytes)
        try:
            while len(self.audio_buffer) >= REQUIRED_BYTES:
                if self.is_done: return
                chunk_to_process = self.audio_buffer[:REQUIRED_BYTES]
                self.audio_buffer = self.audio_buffer[REQUIRED_BYTES:]
                await self._process_chunk(bytes(chunk_to_process))
        except Exception as e:
            _LOGGER.exception("Error processing audio chunk")
            await self.write_event(Error(text=str(e)).event())

    async def _handle_audio_stop(self) -> None:
        if self.is_done:
            return
        self.vad_triggered = True 
        _LOGGER.debug("End of audio stream. Processing remaining buffer and finalizing.")
        try:
            if self.audio_buffer:
                padding_needed = REQUIRED_BYTES - len(self.audio_buffer)
                padded_chunk = self.audio_buffer + (b'\x00' * padding_needed)
                await self._process_chunk_final(bytes(padded_chunk))
                self.audio_buffer.clear()

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

    async def _process_chunk_final(self, chunk_bytes: bytes):
        """Final processing of the buffer without VAD logic but WITH AGC."""
        if self.is_done:
            return
            
        samples_raw = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
        
        # Apply AGC to final chunk as well
        samples_norm = samples_raw / 32768.0
        samples_norm = self.agc.process(samples_norm)
        samples_float = samples_norm * 32767.0
        
        resampled = resampy.resample(samples_float, INCOMING_SAMPLE_RATE, MODEL_SAMPLE_RATE)
        np.clip(resampled, -32768, 32767, out=resampled)
        samples_int32 = resampled.astype(np.int32)
        
        new_phrases, self.state = self.pipeline.forward(samples_int32, self.state)
        if new_phrases:
            chunk_text = " ".join(p.text for p in new_phrases if p.text)
            if chunk_text:
                await self.write_event(TranscriptChunk(text=chunk_text).event())
                self.accumulated_text = (self.accumulated_text + " " + chunk_text).strip()

    async def _finalize_recognition(self, text: str) -> None:
        if self.is_done: return
        final_text = text.strip()
        _LOGGER.info("Final result: '%s'", final_text)
        await self.write_event(Transcript(text=final_text if final_text else "").event())
        await self.write_event(TranscriptStop().event())
        self.is_done = True