import collections

import numpy as np

from orchestrator.audio_io import AudioIO
from orchestrator.config import Settings
from orchestrator.vad import SileroVAD


def record_utterance(audio: AudioIO, vad: SileroVAD, settings: Settings) -> np.ndarray:
    pre_buffer = collections.deque(maxlen=settings.speech_onset_frames)
    utterance_frames = []
    speaking = False
    consecutive_speech = 0
    consecutive_silence = 0

    print("listening...")

    while True:
        frame = audio.read_frame(timeout=10.0)
        probability = vad.process_frame(frame)

        if probability >= settings.vad_threshold:
            consecutive_speech += 1
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            consecutive_speech = 0

        if not speaking:
            pre_buffer.append(frame)

        if speaking:
            utterance_frames.append(frame)

        if not speaking and consecutive_speech >= settings.speech_onset_frames:
            speaking = True
            utterance_frames.extend(pre_buffer)
            print("speech started")

        if speaking and consecutive_silence >= settings.speech_offset_frames:
            print("speech ended")
            break

    return np.concatenate(utterance_frames)


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()