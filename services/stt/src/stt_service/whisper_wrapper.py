from pathlib import Path

import numpy as np
from pywhispercpp.model import Model

from stt_service.config import Settings


class WhisperTranscriber:
    def __init__(self, settings: Settings):
        model_path = str(Path(settings.model_path).resolve())
        self._model = Model(
            model_path,
            n_threads=settings.n_threads,
            print_realtime=False,
            print_progress=False,
        )

    def transcribe_pcm16(self, pcm_bytes: bytes) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments = self._model.transcribe(audio)
        return " ".join(segment.text for segment in segments).strip()