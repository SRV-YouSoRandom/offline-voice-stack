import numpy as np
import openwakeword
from openwakeword.model import Model


class WakeWordDetector:
    CHUNK_SAMPLES = 1280

    def __init__(self, threshold: float = 0.5):
        openwakeword.utils.download_models()
        self._model = Model(inference_framework="onnx")
        self._threshold = threshold
        self._buffer = np.zeros(0, dtype=np.float32)

    def feed(self, frame: np.ndarray) -> str | None:
        self._buffer = np.concatenate([self._buffer, frame])
        detected = None

        while self._buffer.shape[0] >= self.CHUNK_SAMPLES:
            chunk = self._buffer[:self.CHUNK_SAMPLES]
            self._buffer = self._buffer[self.CHUNK_SAMPLES:]

            pcm16_chunk = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            predictions = self._model.predict(pcm16_chunk)

            for name, score in predictions.items():
                if score >= self._threshold:
                    detected = name

        return detected