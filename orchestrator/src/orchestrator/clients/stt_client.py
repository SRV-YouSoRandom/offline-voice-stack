import asyncio
import json

import websockets


async def transcribe(pcm_bytes: bytes, uri: str, max_retries: int = 5, retry_delay_seconds: float = 2.0) -> str:
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with websockets.connect(uri, max_size=None) as websocket:
                chunk_size = 32000
                for i in range(0, len(pcm_bytes), chunk_size):
                    await websocket.send(pcm_bytes[i:i + chunk_size])
                await websocket.send("end")

                response = await websocket.recv()
                result = json.loads(response)

                if result.get("type") == "error":
                    raise RuntimeError(f"stt error: {result.get('message')}")

                return result["text"]
        except (ConnectionRefusedError, OSError, websockets.exceptions.WebSocketException) as exc:
            last_error = exc
            await asyncio.sleep(retry_delay_seconds)

    raise RuntimeError(f"stt service unavailable after {max_retries} retries") from last_error