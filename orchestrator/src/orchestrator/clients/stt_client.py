import json

import websockets


async def transcribe(pcm_bytes: bytes, uri: str) -> str:
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