import numpy as np
import onnxruntime as ort


class SileroVAD:
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512
    CONTEXT_SAMPLES = 64

    def __init__(self, model_path: str, threshold: float = 0.5):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._threshold = threshold
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)
        self._sr = np.array([self.SAMPLE_RATE], dtype=np.int64)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> float:
        if frame.shape[0] != self.FRAME_SAMPLES:
            raise ValueError(f"frame must be {self.FRAME_SAMPLES} samples, got {frame.shape[0]}")

        frame = frame.reshape(1, self.FRAME_SAMPLES).astype(np.float32)
        input_tensor = np.concatenate([self._context, frame], axis=1)

        ort_inputs = {
            "input": input_tensor,
            "state": self._state,
            "sr": self._sr,
        }
        probability, new_state = self._session.run(None, ort_inputs)

        self._state = new_state
        self._context = input_tensor[:, -self.CONTEXT_SAMPLES:]

        return float(probability.item())