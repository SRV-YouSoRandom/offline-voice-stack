# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=1.32.0",
#     "requests>=2.34.2",
#     "tqdm>=4.70.1",
# ]
# ///
"""
Downloads Silero VAD, whisper.cpp ggml, Qwen3-4B Q4_K_M GGUF, and
Kokoro-82M ONNX into their service directories. Run with:
    uv run scripts/download_models.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DirectDownload:
    name: str
    url: str
    dest: Path


@dataclass(frozen=True)
class HFDownload:
    name: str
    repo_id: str
    filename: str
    dest_dir: Path
    dest_filename: str


DIRECT_DOWNLOADS: list[DirectDownload] = [
    DirectDownload(
        name="Silero VAD (ONNX)",
        url=(
            "https://github.com/snakers4/silero-vad/raw/"
            "refs/heads/master/src/silero_vad/data/silero_vad.onnx"
        ),
        dest=REPO_ROOT / "orchestrator" / "models" / "silero_vad.onnx",
    ),
    DirectDownload(
        name="whisper.cpp ggml base.en (STT)",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        dest=REPO_ROOT / "services" / "stt" / "models" / "ggml-base.en.bin",
    ),
    DirectDownload(
        name="Kokoro-82M ONNX, int8 (TTS)",
        url=(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/"
            "download/model-files-v1.0/kokoro-v1.0.int8.onnx"
        ),
        dest=REPO_ROOT / "services" / "tts" / "models" / "kokoro-v1.0.int8.onnx",
    ),
    DirectDownload(
        name="Kokoro-82M voice pack (TTS)",
        url=(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/"
            "download/model-files-v1.0/voices-v1.0.bin"
        ),
        dest=REPO_ROOT / "services" / "tts" / "models" / "voices-v1.0.bin",
    ),
]

HF_DOWNLOADS: list[HFDownload] = [
    HFDownload(
        name="Qwen3-4B Q4_K_M GGUF (LLM)",
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        dest_dir=REPO_ROOT / "services" / "llm" / "models",
        dest_filename="Qwen3-4B-Q4_K_M.gguf",
    ),
]


def download_direct(item: DirectDownload) -> None:
    item.dest.parent.mkdir(parents=True, exist_ok=True)

    if item.dest.exists() and item.dest.stat().st_size > 0:
        print(f"[skip] {item.name} already present at {item.dest}")
        return

    print(f"[fetch] {item.name}")
    print(f"        {item.url}")

    tmp_path = item.dest.with_suffix(item.dest.suffix + ".part")

    with requests.get(item.url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with (
            open(tmp_path, "wb") as f,
            tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=item.name,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    tmp_path.replace(item.dest)
    print(f"[done] {item.name} -> {item.dest}\n")


def download_hf(item: HFDownload) -> None:
    dest_path = item.dest_dir / item.dest_filename
    item.dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[skip] {item.name} already present at {dest_path}")
        return

    print(f"[fetch] {item.name}")
    print(f"        repo_id={item.repo_id} filename={item.filename}")

    downloaded_path = Path(
        hf_hub_download(
            repo_id=item.repo_id,
            filename=item.filename,
            local_dir=str(item.dest_dir),
        )
    )

    if downloaded_path != dest_path:
        downloaded_path.replace(dest_path)

    print(f"[done] {item.name} -> {dest_path}\n")


def main() -> int:
    print(f"Repo root: {REPO_ROOT}\n")

    failures: list[str] = []

    for item in DIRECT_DOWNLOADS:
        try:
            download_direct(item)
        except Exception as exc:
            print(f"[error] failed to download {item.name}: {exc}\n")
            failures.append(item.name)

    for item in HF_DOWNLOADS:
        try:
            download_hf(item)
        except Exception as exc:
            print(f"[error] failed to download {item.name}: {exc}\n")
            failures.append(item.name)

    print("=" * 60)
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for name in failures:
            print(f"  - {name}")
        return 1

    print("All models downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())