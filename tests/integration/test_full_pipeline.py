import numpy as np
import pytest

from orchestrator.clients import llm_client, stt_client, tts_client
from orchestrator.config import Settings


def _generate_silence_pcm16(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    samples = np.zeros(int(seconds * sample_rate), dtype=np.int16)
    return samples.tobytes()


async def test_stt_service_responds_to_silence():
    settings = Settings.from_env()
    pcm_bytes = _generate_silence_pcm16()

    text = await stt_client.transcribe(pcm_bytes, settings.stt_uri)

    assert isinstance(text, str)


def test_llm_service_responds_to_prompt():
    settings = Settings.from_env()

    reply = ""
    for token in llm_client.stream_chat([{"role": "user", "content": "Say hello in one word."}], settings.llm_uri):
        reply += token

    assert len(reply.strip()) > 0


async def test_tts_service_returns_audio():
    settings = Settings.from_env()

    samples, sample_rate = await tts_client.synthesize("This is a test.", settings.tts_uri)

    assert isinstance(samples, np.ndarray)
    assert len(samples) > 0
    assert sample_rate == 24000


async def test_full_round_trip():
    settings = Settings.from_env()

    reply = ""
    for token in llm_client.stream_chat([{"role": "user", "content": "Say hello in one word."}], settings.llm_uri):
        reply += token

    assert len(reply.strip()) > 0

    samples, sample_rate = await tts_client.synthesize(reply, settings.tts_uri)

    assert len(samples) > 0
    assert sample_rate == 24000


async def test_stt_client_fails_cleanly_against_unreachable_service():
    with pytest.raises(RuntimeError):
        await stt_client.transcribe(
            _generate_silence_pcm16(),
            "ws://localhost:59999",
            max_retries=1,
            retry_delay_seconds=0.1,
        )