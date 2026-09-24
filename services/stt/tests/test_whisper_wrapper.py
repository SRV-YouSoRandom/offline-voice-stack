import numpy as np

from stt_service.config import Settings
from stt_service.whisper_wrapper import WhisperTranscriber


def test_transcribe_returns_string_on_silence():
    settings = Settings.from_env()
    transcriber = WhisperTranscriber(settings)

    silence = np.zeros(16000, dtype=np.int16)
    result = transcriber.transcribe_pcm16(silence.tobytes())

    assert isinstance(result, str)