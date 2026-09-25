import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd

from orchestrator.audio_io import AudioIO
from orchestrator.clients import llm_client, stt_client, tts_client
from orchestrator.config import Settings
from orchestrator.recorder import float32_to_pcm16, record_utterance
from orchestrator.sentence_chunker import SentenceChunker
from orchestrator.vad import SileroVAD


def _llm_stream_worker(transcript: str, uri: str, token_queue: "queue.Queue[str | None]") -> None:
    try:
        for token in llm_client.stream_chat(transcript, uri):
            token_queue.put(token)
    finally:
        token_queue.put(None)


async def produce_sentences(transcript: str, settings: Settings, sentence_queue: asyncio.Queue) -> None:
    token_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_llm_stream_worker,
        args=(transcript, settings.llm_uri, token_queue),
        daemon=True,
    )
    thread.start()

    chunker = SentenceChunker()
    loop = asyncio.get_running_loop()
    reply_parts = []

    while True:
        token = await loop.run_in_executor(None, token_queue.get)
        if token is None:
            break

        reply_parts.append(token)
        for sentence in chunker.feed(token):
            await sentence_queue.put(sentence)

    remaining = chunker.flush()
    if remaining:
        await sentence_queue.put(remaining)

    await sentence_queue.put(None)
    print(f"assistant: {''.join(reply_parts)}")


async def synthesize_sentences(
    sentence_queue: asyncio.Queue, audio_queue: asyncio.Queue, settings: Settings
) -> None:
    while True:
        sentence = await sentence_queue.get()
        if sentence is None:
            await audio_queue.put(None)
            break

        samples, sample_rate = await tts_client.synthesize(sentence, settings.tts_uri)
        await audio_queue.put((samples, sample_rate))


def _play_blocking(samples: np.ndarray, sample_rate: int) -> None:
    sd.play(samples, sample_rate)
    sd.wait()


async def play_audio(audio_queue: asyncio.Queue) -> None:
    loop = asyncio.get_running_loop()

    while True:
        item = await audio_queue.get()
        if item is None:
            break

        samples, sample_rate = item
        await loop.run_in_executor(None, _play_blocking, samples, sample_rate)


async def run_turn(audio: AudioIO, vad: SileroVAD, settings: Settings) -> None:
    utterance = record_utterance(audio, vad, settings)
    audio.stop()

    pcm_bytes = float32_to_pcm16(utterance)

    print("transcribing...")
    transcript = await stt_client.transcribe(pcm_bytes, settings.stt_uri)
    print(f"you said: {transcript}")

    if not transcript.strip():
        print("no speech detected, skipping")
        return

    sentence_queue: asyncio.Queue = asyncio.Queue()
    audio_queue: asyncio.Queue = asyncio.Queue()

    print("thinking...")

    await asyncio.gather(
        produce_sentences(transcript, settings, sentence_queue),
        synthesize_sentences(sentence_queue, audio_queue, settings),
        play_audio(audio_queue),
    )


async def main() -> None:
    settings = Settings.from_env()
    vad = SileroVAD(settings.vad_model_path, threshold=settings.vad_threshold)
    audio = AudioIO()

    print("streaming pipeline, press ctrl+c to stop")

    try:
        while True:
            audio.start()
            try:
                await run_turn(audio, vad, settings)
            finally:
                audio.stop()
            vad.reset()
    except KeyboardInterrupt:
        print("stopping")


if __name__ == "__main__":
    asyncio.run(main())