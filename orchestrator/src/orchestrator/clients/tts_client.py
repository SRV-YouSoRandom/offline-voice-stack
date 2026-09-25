import json

import numpy as np
import websockets


async def synthesize(text: str, uri: str) -> tuple[np.ndarray, int]:
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