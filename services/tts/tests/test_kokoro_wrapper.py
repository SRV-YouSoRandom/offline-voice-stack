import time

from tts_service.config import Settings
from tts_service.kokoro_wrapper import KokoroSynthesizer


def test_synthesize_returns_audio_bytes():
    settings = Settings.from_env()
    synthesizer = KokoroSynthesizer(settings)

    pcm_bytes, sample_rate = synthesizer.synthesize("Hello world.")

    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) > 0
    assert sample_rate == 24000


def test_synthesize_runs_faster_than_real_time():
    settings = Settings.from_env()
    synthesizer = KokoroSynthesizer(settings)

    text = "This is a longer sentence used to measure synthesis speed against real time."

    start = time.perf_counter()
    pcm_bytes, sample_rate = synthesizer.synthesize(text)
    elapsed = time.perf_counter() - start

    audio_seconds = (len(pcm_bytes) / 2) / sample_rate
    real_time_factor = elapsed / audio_seconds

    assert real_time_factor < 1.0