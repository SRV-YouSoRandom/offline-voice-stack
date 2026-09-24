import logging
import time

import numpy as np
from kokoro_onnx import Kokoro

from tts_service.config import Settings

logger = logging.getLogger("tts_service.kokoro_wrapper")


class KokoroSynthesizer:
    def __init__(self, settings: Settings):
        self._kokoro = Kokoro(settings.model_path, settings.voices_path)
        self._voice = settings.voice
        self._lang = settings.lang

    def synthesize(self, text: str) -> tuple[bytes, int]:
        start = time.perf_counter()
        samples, sample_rate = self._kokoro.create(text, voice=self._voice, lang=self._lang)
        elapsed = time.perf_counter() - start

        audio_seconds = len(samples) / sample_rate
        real_time_factor = elapsed / audio_seconds if audio_seconds > 0 else 0.0

        logger.info(
            f"synthesized {audio_seconds:.2f}s of audio in {elapsed:.2f}s "
            f"(real-time factor {real_time_factor:.2f})"
        )

        pcm16 = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        return pcm16.tobytes(), sample_rate