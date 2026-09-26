import numpy as np


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()