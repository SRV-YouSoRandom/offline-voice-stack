import asyncio
import json
import logging

import websockets

from stt_service.config import Settings
from stt_service.whisper_wrapper import WhisperTranscriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_service")


async def handle_connection(websocket, transcriber: WhisperTranscriber) -> None:
    buffer = bytearray()
    logger.info("client connected")

    async for message in websocket:
        if isinstance(message, bytes):
            buffer.extend(message)
            continue

        if message == "end":
            if len(buffer) == 0:
                await websocket.send(json.dumps({"type": "error", "message": "no audio received"}))
                break

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, transcriber.transcribe_pcm16, bytes(buffer))

            await websocket.send(json.dumps({"type": "final", "text": text}))
            buffer.clear()
            break

    logger.info("client disconnected")


async def main() -> None:
    settings = Settings.from_env()
    logger.info(f"loading model from {settings.model_path}")
    transcriber = WhisperTranscriber(settings)
    logger.info("model loaded")

    async def handler(websocket):
        await handle_connection(websocket, transcriber)

    async with websockets.serve(handler, settings.host, settings.port, max_size=None):
        logger.info(f"stt service listening on ws://{settings.host}:{settings.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())