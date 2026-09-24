import asyncio
import json
import sys
import wave

import websockets


async def synthesize(text: str, output_path: str, uri: str = "ws://localhost:8003") -> None:
    async with websockets.connect(uri, max_size=None) as websocket:
        await websocket.send(json.dumps({"text": text}))

        audio_bytes = await websocket.recv()
        result = await websocket.recv()
        info = json.loads(result)

        if info.get("type") == "error":
            print(f"error: {info.get('message')}")
            return

        sample_rate = info["sample_rate"]

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        print(f"wrote {output_path}: {sample_rate} Hz, {len(audio_bytes)} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: test_client_manual.py <text> <output-wav-path>")
        sys.exit(1)

    asyncio.run(synthesize(sys.argv[1], sys.argv[2]))