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
            stt_uri=os.environ.get("STT_URI", "ws://localhost:8001"),
            llm_uri=os.environ.get("LLM_URI", "http://localhost:8002/v1/chat/completions"),
            tts_uri=os.environ.get("TTS_URI", "ws://localhost:8003"),
        )