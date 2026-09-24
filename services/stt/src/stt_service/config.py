import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str
    host: str
    port: int
    n_threads: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=os.environ.get("STT_MODEL_PATH", "/app/models/ggml-base.en.bin"),
            host=os.environ.get("STT_HOST", "0.0.0.0"),
            port=int(os.environ.get("STT_PORT", "8001")),
            n_threads=int(os.environ.get("STT_N_THREADS", "4")),
        )