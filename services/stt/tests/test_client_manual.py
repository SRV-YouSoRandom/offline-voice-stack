import asyncio
import json
import sys
import wave

import websockets


async def send_wav(path: str, uri: str = "ws://localhost:8001") -> None:
    with wave.open(path, "rb") as wav_file:
        if wav_file.getframerate() != 16000:
            raise ValueError("wav file must be 16000 Hz")
        if wav_file.getnchannels() != 1:
            raise ValueError("wav file must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError("wav file must be 16-bit PCM")
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    async with websockets.connect(uri, max_size=None) as websocket:
        chunk_size = 32000
        for i in range(0, len(pcm_bytes), chunk_size):
            await websocket.send(pcm_bytes[i:i + chunk_size])
        await websocket.send("end")

        response = await websocket.recv()
        print(json.loads(response))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: test_client_manual.py <path-to-16khz-mono-wav>")
        sys.exit(1)

    asyncio.run(send_wav(sys.argv[1]))