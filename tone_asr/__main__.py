import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Set

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from tone import StreamingCTCPipeline, DecoderType
from .handler import ToneEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        default="ru",
        help="Default language for transcription",
    )
    parser.add_argument(
        "--uri", default="tcp://0.0.0.0:10303", help="URI for the server to listen on"
    )
    
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["greedy", "beam_search"],
        default="greedy",
        help="Decoding method. 'greedy' is fast and lightweight (default). "
             "'beam_search' is more accurate but requires downloading a large ~5.5GB language model.",
    )
    
    parser.add_argument(
        "--amplification-factor",
        type=float,
        default=2.0,
        help="Factor to multiply audio samples by. 1.0 means no change.",
    )
    parser.add_argument(
        "--command-file",
        default="",
        help="Path to a file with commands for early stopping.",
    )
    parser.add_argument(
        "--command-max-words",
        type=int,
        default=5,
        help="Word count threshold to trigger a one-time command check.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    _LOGGER.debug(args)

    commands: Set[str] = set()
    if args.command_file:
        command_path = Path(args.command_file)
        try:
            with open(command_path, "r", encoding="utf-8") as f:
                commands = {line.strip().lower() for line in f if line.strip()}
            _LOGGER.info("Loaded %s command(s) for early stopping from %s", len(commands), command_path)
        except Exception as e:
            _LOGGER.warning("Failed to read command file %s: %s", command_path, e)
    
    try:
        if args.decoder == "greedy":
            _LOGGER.info("Loading T-one model in GREEDY mode (lightweight)...")
            pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
        elif args.decoder == "beam_search":
            _LOGGER.info("Loading T-one model in BEAM_SEARCH mode (heavy, usw ~5.5GB LM)...")
            pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.BEAM_SEARCH)
        
        _LOGGER.info("Model loaded successfully.")
        
    except Exception as e:
        _LOGGER.exception("Error loading model")
        sys.exit(1)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="T-one",
                description="Fast, local speech recognition with T-one",
                attribution=Attribution(
                    name="T-one by voicekit-team", url="https://github.com/voicekit-team/T-one"
                ),
                installed=True,
                supports_transcript_streaming=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name="t-tech/T-one",
                        description="T-one Streaming ASR for Russian Telephony",
                        attribution=Attribution(
                            name="t-tech",
                            url="https://huggingface.co/t-tech/T-one",
                        ),
                        installed=True,
                        languages=[args.language],
                        version="1.0",
                    )
                ],
            )
        ],
    )

    _LOGGER.info("Server is ready and listening at %s", args.uri)

    handler_factory = partial(
        ToneEventHandler,
        wyoming_info=wyoming_info,
        cli_args=args,
        pipeline=pipeline,
        commands=commands,
        command_max_words=args.command_max_words,
    )

    server = AsyncServer.from_uri(args.uri)
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass