import queue

import numpy as np
import sounddevice as sd


class AudioIO:
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512
    CHANNELS = 1

    def __init__(self):
        self._input_queue: queue.Queue = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"audio status: {status}")

        mono = indata[:, 0].copy()
        self._input_queue.put(mono)

    def start(self) -> None:
        while not self._input_queue.empty():
            self._input_queue.get_nowait()

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.FRAME_SAMPLES,
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_frame(self, timeout: float | None = None) -> np.ndarray:
        return self._input_queue.get(timeout=timeout)