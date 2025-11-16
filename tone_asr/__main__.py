import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path

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
        default=1.0,
        help="Factor to multiply audio samples by. 1.0 means no change.",
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

    try:
        if args.decoder == "greedy":
            _LOGGER.info("Loading T-one model in GREEDY mode (lightweight)...")
            pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
        elif args.decoder == "beam_search":
            _LOGGER.info("Loading T-one model in BEAM_SEARCH mode (heavy, will download ~5.5GB language model)...")
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
    )

    server = AsyncServer.from_uri(args.uri)
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass