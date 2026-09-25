import asyncio
import time

from orchestrator.clients import llm_client, tts_client
from orchestrator.config import Settings
from orchestrator.sentence_chunker import SentenceChunker

TEST_PROMPT = "Tell me three interesting facts about the ocean, in three sentences."


async def bench_non_streaming(settings) -> float:
    start = time.perf_counter()

    full_text = ""
    for token in llm_client.stream_chat(TEST_PROMPT, settings.llm_uri):
        full_text += token

    await tts_client.synthesize(full_text, settings.tts_uri)

    return time.perf_counter() - start


async def bench_streaming(settings) -> float:
    start = time.perf_counter()
    chunker = SentenceChunker()
    first_audio_time = None

    for token in llm_client.stream_chat(TEST_PROMPT, settings.llm_uri):
        for sentence in chunker.feed(token):
            await tts_client.synthesize(sentence, settings.tts_uri)
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - start

    remaining = chunker.flush()
    if remaining:
        await tts_client.synthesize(remaining, settings.tts_uri)
        if first_audio_time is None:
            first_audio_time = time.perf_counter() - start

    return first_audio_time if first_audio_time is not None else time.perf_counter() - start


async def main() -> None:
    settings = Settings.from_env()

    print("benchmarking non-streaming (full response required before any audio)...")
    non_streaming_time = await bench_non_streaming(settings)
    print(f"non-streaming time to first audio: {non_streaming_time:.2f}s")

    print("benchmarking streaming (first sentence audio as soon as ready)...")
    streaming_time = await bench_streaming(settings)
    print(f"streaming time to first audio: {streaming_time:.2f}s")

    improvement = (non_streaming_time - streaming_time) / non_streaming_time * 100
    print(f"improvement: {improvement:.1f}% faster time to first audio")


if __name__ == "__main__":
    asyncio.run(main())