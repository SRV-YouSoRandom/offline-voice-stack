import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str
    voices_path: str
    host: str
    port: int
    voice: str
    lang: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=os.environ.get("TTS_MODEL_PATH", "/app/models/kokoro-v1.0.fp16.onnx"),
            voices_path=os.environ.get("TTS_VOICES_PATH", "/app/models/voices-v1.0.bin"),
            host=os.environ.get("TTS_HOST", "0.0.0.0"),
            port=int(os.environ.get("TTS_PORT", "8003")),
            voice=os.environ.get("TTS_VOICE", "af_heart"),
            lang=os.environ.get("TTS_LANG", "en-us"),
        )