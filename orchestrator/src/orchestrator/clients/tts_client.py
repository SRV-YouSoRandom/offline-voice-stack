import asyncio
import json

import numpy as np
import websockets


async def synthesize(text: str, uri: str, max_retries: int = 5, retry_delay_seconds: float = 2.0) -> tuple[np.ndarray, int]:
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with websockets.connect(uri, max_size=None) as websocket:
                await websocket.send(json.dumps({"text": text}))

                audio_bytes = await websocket.recv()
                result = await websocket.recv()
                info = json.loads(result)

                if info.get("type") == "error":
                    raise RuntimeError(f"tts error: {info.get('message')}")

                sample_rate = info["sample_rate"]
                pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
                samples = pcm16.astype(np.float32) / 32768.0

                return samples, sample_rate
        except (ConnectionRefusedError, OSError, websockets.exceptions.WebSocketException) as exc:
            last_error = exc
            await asyncio.sleep(retry_delay_seconds)

    raise RuntimeError(f"tts service unavailable after {max_retries} retries") from last_error