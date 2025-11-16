# file: handler.py

import argparse
import asyncio
import logging
from typing import Set, List, Optional, Any

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

VAD_SILENCE_THRESHOLD_RATIO = 0.35
VAD_PATIENCE_CHUNKS = 6


class ToneEventHandler(AsyncEventHandler):
    """Event handler for each client using T-one with a custom VAD."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        pipeline: StreamingCTCPipeline,
        commands: Set[str],
        command_max_words: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.cli_args = cli_args
        self.pipeline = pipeline
        self.commands = commands
        self.command_max_words = command_max_words
        
        self.sorted_commands: List[str] = sorted(list(self.commands), key=len, reverse=True)
        self.language = self.cli_args.language
        self.state: Optional[Any] = None
        self.accumulated_text: str = ""
        self.command_recognized = False
        self.check_performed = False
        self.audio_buffer = bytearray()
        
        self.vad_peak_energy: float = 0.0
        self.vad_quiet_chunks: int = 0
        self.vad_triggered: bool = False
        
        _LOGGER.debug("Event handler initialized")

    async def _process_chunk(self, chunk_bytes: bytes):
        """Processes a chunk of audio, including VAD logic."""
        if self.command_recognized or self.vad_triggered:
            return

        samples_float = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)

        # --- VAD Logic ---
        rms_energy = np.sqrt(np.mean(samples_float**2))
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

        # Audio Processing
        amplification = self.cli_args.amplification_factor
        if amplification != 1.0:
            samples_float *= amplification
        resampled_samples_float = resampy.resample(
            samples_float, INCOMING_SAMPLE_RATE, MODEL_SAMPLE_RATE
        )
        np.clip(resampled_samples_float, -32768, 32767, out=resampled_samples_float)
        samples_int32 = resampled_samples_float.astype(np.int32)
        
        # ASR
        new_phrases, self.state = self.pipeline.forward(samples_int32, self.state)
        if new_phrases:
            chunk_text = " ".join(p.text for p in new_phrases if p.text)
            if chunk_text:
                _LOGGER.debug("New phrases received: '%s'", chunk_text)
                await self.write_event(TranscriptChunk(text=chunk_text).event())
                self.accumulated_text = (self.accumulated_text + " " + chunk_text).strip()
        
        # Command Check
        if self.sorted_commands and not self.check_performed and self.command_max_words > 0:
            word_count = len(self.accumulated_text.split())
            if word_count >= self.command_max_words:
                self.check_performed = True
                matched_command = self._check_for_command(self.accumulated_text.lower())
                if matched_command:
                    await self._finalize_recognition(matched_command)


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
        _LOGGER.debug("Audio stream started. Resetting ASR and VAD state.")
        self.state = None
        self.accumulated_text = ""
        self.command_recognized = False
        self.check_performed = False
        self.audio_buffer.clear()
        
        self.vad_peak_energy = 0.0
        self.vad_quiet_chunks = 0
        self.vad_triggered = False
        
        await self.write_event(TranscriptStart(language=self.language).event())

    async def _handle_audio_chunk(self, audio_chunk_bytes: bytes) -> None:
        if self.command_recognized or self.vad_triggered:
            return
        
        self.audio_buffer.extend(audio_chunk_bytes)
        try:
            while len(self.audio_buffer) >= REQUIRED_BYTES:
                if self.command_recognized or self.vad_triggered: return
                chunk_to_process = self.audio_buffer[:REQUIRED_BYTES]
                self.audio_buffer = self.audio_buffer[REQUIRED_BYTES:]
                await self._process_chunk(bytes(chunk_to_process))
        except Exception as e:
            _LOGGER.exception("Error processing audio chunk")
            await self.write_event(Error(text=str(e)).event())

    async def _handle_audio_stop(self) -> None:
        if self.command_recognized:
            return
        
        self.vad_triggered = True 
        
        _LOGGER.debug("End of audio stream. Processing remaining buffer and finalizing.")
        try:
            if self.audio_buffer:
                padding_needed = REQUIRED_BYTES - len(self.audio_buffer)
                padded_chunk = self.audio_buffer + (b'\x00' * padding_needed)
                await self._process_chunk_final(bytes(padded_chunk))
                self.audio_buffer.clear()

            if self.command_recognized: return

            final_phrases, _ = self.pipeline.finalize(self.state)
            if final_phrases:
                final_text_part = " ".join(p.text for p in final_phrases if p.text)
                if final_text_part:
                     await self.write_event(TranscriptChunk(text=final_text_part).event())
                     self.accumulated_text = (self.accumulated_text + " " + final_text_part).strip()
            
            _LOGGER.debug("Full final text for checking: '%s'", self.accumulated_text)

            if not self.check_performed:
                matched_command = self._check_for_command(self.accumulated_text.lower())
                if matched_command:
                    await self._finalize_recognition(matched_command)
                    return
            
            await self._finalize_recognition(self.accumulated_text)
            
        except Exception as e:
            _LOGGER.exception("Error during finalization")
            await self.write_event(Error(text=str(e)).event())

    async def _process_chunk_final(self, chunk_bytes: bytes):
        """Final processing of the buffer without VAD logic."""
        samples_float = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
        amplification = self.cli_args.amplification_factor
        if amplification != 1.0:
            samples_float *= amplification
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
        if self.command_recognized: return
        final_text = text.strip()
        if not final_text: return
        _LOGGER.info("Final result: '%s'", final_text)
        await self.write_event(Transcript(text=final_text).event())
        await self.write_event(TranscriptStop().event())
        self.command_recognized = True

    def _check_for_command(self, text: str) -> Optional[str]:
        if not self.sorted_commands: return None
        for command in self.sorted_commands:
            if command in text: return command

        return None

