import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    vad_model_path: str
    vad_threshold: float
    speech_onset_frames: int
    speech_offset_frames: int
    wake_word_threshold: float
    barge_in_threshold: float
    barge_in_onset_frames: int
    barge_in_rms_threshold: float
    stt_uri: str
    llm_uri: str
    tts_uri: str

    @classmethod
    def from_env(cls) -> "Settings":
        default_model_path = str(REPO_ROOT / "orchestrator" / "models" / "silero_vad.onnx")
        return cls(
            vad_model_path=os.environ.get("VAD_MODEL_PATH", default_model_path),
            vad_threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            speech_onset_frames=int(os.environ.get("VAD_ONSET_FRAMES", "3")),
            speech_offset_frames=int(os.environ.get("VAD_OFFSET_FRAMES", "15")),
            wake_word_threshold=float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5")),
            barge_in_threshold=float(os.environ.get("BARGE_IN_THRESHOLD", "0.8")),
            barge_in_onset_frames=int(os.environ.get("BARGE_IN_ONSET_FRAMES", "10")),
            barge_in_rms_threshold=float(os.environ.get("BARGE_IN_RMS_THRESHOLD", "0.015")),
            stt_uri=os.environ.get("STT_URI", "ws://localhost:8001"),
            llm_uri=os.environ.get("LLM_URI", "http://localhost:8002/v1/chat/completions"),
            tts_uri=os.environ.get("TTS_URI", "ws://localhost:8003"),
        )