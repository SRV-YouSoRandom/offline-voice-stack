import asyncio
import time

from orchestrator.clients import tts_client
from orchestrator.config import Settings

TEST_SENTENCE = "Alright, no problem."


async def main() -> None:
    settings = Settings.from_env()

    for i in range(3):
        start = time.perf_counter()
        samples, sample_rate = await tts_client.synthesize(TEST_SENTENCE, settings.tts_uri)
        elapsed = time.perf_counter() - start
        audio_seconds = len(samples) / sample_rate
        print(f"call {i + 1}: {elapsed:.3f}s wall time for {audio_seconds:.2f}s of audio")


if __name__ == "__main__":
    asyncio.run(main())