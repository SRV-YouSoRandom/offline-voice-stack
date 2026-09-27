# offline-voice-stack

A zero-cloud-dependency, real-time voice-to-voice personal assistant. Speech-to-text, language generation, and text-to-speech all run locally, no API keys, no internet requirement once the models are downloaded.

<!-- Demo video/GIF goes here -->

## Architecture

Two runtime domains: a native Windows process handling audio I/O, wake word detection, voice activity detection, and orchestration, talking over WebSocket and HTTP to three Dockerized services, each with one responsibility.

- **STT**: whisper.cpp (`ggml-base.en`), wrapped in a WebSocket server
- **LLM**: `llama-server` (llama.cpp) running Qwen3-4B, Q4_K_M quantized, thinking mode disabled for latency
- **TTS**: Kokoro-82M (ONNX, fp16), wrapped in a WebSocket server
- **VAD**: Silero VAD (ONNX), run directly in the orchestrator
- **Wake word**: openWakeWord
- **Dashboard**: Streamlit, subscribed to a live event bus from the orchestrator

Full design decisions, protocols, and rationale are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Real, measured results

Numbers, not adjectives, from actual runs against this stack:

- **Streaming reduced time-to-first-audio by 32.8%** (11.36s to 7.64s) by synthesizing and playing each sentence as soon as the LLM finishes it, rather than waiting for the full response (`scripts/bench_latency.py`)
- **Disabling Qwen3's thinking mode cut a trivial response from 655 tokens and 59s to 8 tokens and 1.1s**, a hidden reasoning block was running by default even for "say hello in five words"
- **Kokoro's fp16 model ran 5.6x faster than int8 on CPU** (real-time factor 0.48 vs 2.69), despite being the larger file, ONNX Runtime's int8 kernels weren't well optimized for this CPU
- **Barge-in detection required two independent signals, not one.** VAD probability alone couldn't distinguish genuine user interruption from audio leaking out of wired-earbud drivers into their inline mic, both registered as high-confidence speech. Adding an RMS energy gate (leaked audio measured under 0.007, genuine speech measured above 0.03) resolved it cleanly

## Prerequisites

- Windows with WSL2 and Docker Desktop
- [uv](https://docs.astral.sh/uv/) installed on the host
- A CPU-only baseline is assumed; see `docs/ARCHITECTURE.md` section 7 and section 10 for the deferred AMD Vulkan GPU offload path

## Setup

Clone the repo, then pull all four models (roughly 2.7GB):

```powershell
uv sync --all-packages
uv run scripts\download_models.py
```

Build and start the three containerized services:

```powershell
docker compose up -d stt llm tts
docker compose logs llm
```

Wait for `model loaded` in the LLM logs before continuing, it's the slowest to start.

## Running it

Two processes, in separate terminals, both from the repo root:

**The voice pipeline:**

```powershell
cd orchestrator
uv run python -m orchestrator.pipeline
```

Say a wake word (`alexa`, `hey jarvis`, `hey mycroft`, or `hey rhasspy`), then speak. Interrupt it mid-response by talking over it.

**The live dashboard** (optional, shows pipeline state, transcript, reply, and per-stage latency in real time):

```powershell
uv run streamlit run dashboard\src\dashboard\app.py
```

## Testing

Unit tests run inside each service's own container:

```powershell
docker compose run --rm --entrypoint uv stt run pytest -q
docker compose run --rm --entrypoint uv tts run pytest -q
```

Integration tests run from the host against the live running services:

```powershell
uv run --all-packages pytest tests\integration -v
```

## Project structure

offline-voice-stack/
├── docs/ARCHITECTURE.md full design doc: protocols, latency strategy, decisions, deferred work
├── services/
│ ├── stt/ whisper.cpp, Dockerized
│ ├── llm/ llama-server + Qwen3-4B, Dockerized
│ └── tts/ kokoro-onnx, Dockerized
├── orchestrator/ native process: audio I/O, VAD, wake word, pipeline, event bus
├── dashboard/ native Streamlit app
├── shared/protocol/ schemas shared across services
├── scripts/ model download, latency benchmarking, diagnostics
└── tests/integration/ end-to-end tests against live services


## Known limitations

- Barge-in detection assumes a mic not acoustically coupled to the output device (headphones without an inline mic, or a boom mic). Wired earbuds with an inline mic will false-trigger from driver leakage even with the RMS gate; this is a hardware limitation, not a software one, real acoustic echo cancellation would require a reference signal of the output audio, out of scope here.
- GPU acceleration is not implemented; the LLM and STT services run CPU-only. A Vulkan-based path for the AMD integrated GPU present on the dev machine is documented as deferred work in `docs/ARCHITECTURE.md`.
- No conversation memory across turns; each turn is stateless from the LLM's perspective.