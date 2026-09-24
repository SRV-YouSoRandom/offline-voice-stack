import asyncio
import json
import logging

import websockets

from tts_service.config import Settings
from tts_service.kokoro_wrapper import KokoroSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_service")


async def handle_connection(websocket, synthesizer: KokoroSynthesizer) -> None:
    logger.info("client connected")

    async for message in websocket:
        if isinstance(message, bytes):
            continue

        payload = json.loads(message)
        text = payload.get("text", "").strip()

        if not text:
            await websocket.send(json.dumps({"type": "error", "message": "no text provided"}))
            break

        loop = asyncio.get_running_loop()
        pcm_bytes, sample_rate = await loop.run_in_executor(None, synthesizer.synthesize, text)

        await websocket.send(pcm_bytes)
        await websocket.send(json.dumps({"type": "final", "sample_rate": sample_rate}))
        break

    logger.info("client disconnected")


async def main() -> None:
    settings = Settings.from_env()
    logger.info(f"loading model from {settings.model_path}")
    synthesizer = KokoroSynthesizer(settings)
    logger.info("model loaded")

    async def handler(websocket):
        await handle_connection(websocket, synthesizer)

    async with websockets.serve(handler, settings.host, settings.port, max_size=None):
        logger.info(f"tts service listening on ws://{settings.host}:{settings.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())