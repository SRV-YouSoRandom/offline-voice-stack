# offline-voice-stack, Architecture

## 1. Overview

offline-voice-stack is a zero-cloud-dependency, real-time voice-to-voice
personal assistant. Every model in the pipeline runs locally: speech-to-text,
language generation, and text-to-speech all execute on-device, with no
network calls leaving the machine at inference time.

The system is split into two runtime domains:

- Native Windows process: handles microphone capture, speaker playback,
  voice activity detection, wake word detection, and orchestration. Runs
  directly on the host via uv, not in Docker, because Docker Desktop's
  WSL2 backend does not have native access to Windows audio devices.
- Docker Compose stack (WSL2 backend): three isolated, swappable services
  (STT, LLM, TTS), each exposing a local network API. Compute-heavy and
  model-loading logic lives here, decoupled from audio I/O.

A Streamlit dashboard runs natively alongside the orchestrator, subscribing
to a lightweight event stream to visualize the pipeline in real time:
partial transcripts, streaming LLM tokens, TTS status, and per-stage
latency.

## 2. Design goals

1. Zero cloud dependency at inference time. All four models (VAD, STT,
   LLM, TTS) run locally. No API keys, no internet requirement once models
   are downloaded.
2. Perceived low latency over raw throughput, achieved through streaming
   at every stage rather than waiting for each stage to fully complete
   before starting the next.
3. Clean service boundaries. Each service is independently buildable,
   testable, and replaceable, so swapping Kokoro for a different TTS
   engine should not require touching the STT or LLM services.
4. Reproducibility. Models are pulled by a single script into gitignored
   models directories, so the repo itself stays small.

## 3. High-level architecture

Windows laptop
Native Python process (uv, runs on host)
Mic capture
Silero VAD
openWakeWord
Pipeline orchestrator
Speaker playback
Event bus (for dashboard)
connects over WebSocket and HTTP to
Docker Compose stack (WSL2)
STT service: whisper.cpp
LLM service: llama.cpp server, Qwen3-4B Q4_K_M
TTS service: Kokoro-82M (ONNX)
Native Python process also feeds
Streamlit dashboard (uv, runs on host)

## 4. Services

### 4.1 STT service (services/stt)

Engine: whisper.cpp, built from source with CMake inside the container.
Model: ggml-base.en.bin, English only, about 142 MB.
Interface: WebSocket server. Accepts streaming 16 kHz mono PCM audio
chunks, emits partial transcripts as speech continues and a final
transcript on end of utterance.
Why WebSocket: audio must be streamed in continuously. A single
request and response HTTP call would force the orchestrator to wait for
the full utterance before sending anything.

### 4.2 LLM service (services/llm)

Engine: llama.cpp's built-in llama-server binary, no custom Python
wrapper needed.
Model: Qwen3-4B-Q4_K_M.gguf, 4-bit K-quant, about 2.5 GB.
Interface: OpenAI-compatible HTTP API (/v1/chat/completions) with SSE
token streaming, provided natively by llama-server.
Why HTTP and SSE instead of WebSocket: llama-server already implements
streaming over SSE correctly. There is no benefit to reimplementing this
over WebSocket.

### 4.3 TTS service (services/tts)

Engine: kokoro-onnx (Python), wrapping the Kokoro-82M ONNX model.
Model: kokoro-v1.0.int8.onnx, int8 quantized, about 88 MB, plus
voices-v1.0.bin as the voice pack.
Interface: WebSocket server. Accepts text chunks, sentence level, from
the orchestrator's sentence-chunking logic, and streams back synthesized
audio bytes as they are generated.
Why WebSocket: synthesis for sentence one must start streaming back
before sentence two has even been requested. This is the core mechanism
behind perceived low latency, covered in section 6.

### 4.4 Orchestrator (orchestrator, native)

Owns the full conversational state machine:

1. Continuously listens via mic, gated by openWakeWord.
2. Once woken, streams audio to the STT service and monitors Silero VAD
   locally to detect end of utterance.
3. Sends the finalized transcript to the LLM service, streaming tokens
   back.
4. As complete sentences accumulate in the token stream, forwards each
   sentence to the TTS service immediately, not waiting for the full
   response.
5. Plays back TTS audio chunks as they arrive, monitors the mic during
   playback for barge-in, and if speech is detected, halts playback and
   returns to listening.
6. Publishes state transitions and stage latency events to the event bus
   for the dashboard to consume.

### 4.5 Dashboard (dashboard, native)

A Streamlit app subscribed to the orchestrator's event stream. Renders:

- Current pipeline state: idle, listening, transcribing, thinking,
  speaking.
- Live partial transcript text.
- Streaming LLM token output.
- Per-stage latency: STT, LLM time to first token, TTS time to first
  audio.

Not on the critical audio path. If the dashboard is closed or crashes,
the voice pipeline continues unaffected.

## 5. Communication protocols

| Link                      | Protocol                    | Payload                                                     |
| ------------------------- | --------------------------- | ----------------------------------------------------------- |
| Orchestrator to STT       | WebSocket                   | Binary PCM chunks in, JSON partial or final transcript out  |
| Orchestrator to LLM       | HTTP, SSE                   | OpenAI-compatible chat completion request, SSE token deltas |
| Orchestrator to TTS       | WebSocket                   | JSON text in, binary audio chunks out                       |
| Orchestrator to Dashboard | WebSocket or SSE, event bus | JSON state and latency events                               |

Message schemas are centralized in shared/protocol/schemas.py as Pydantic
models, imported by both the orchestrator and the Python services, so the
contract cannot silently drift between sides.

## 6. Latency strategy

The naive pipeline (record full utterance, then full transcript, then full
LLM response, then full TTS audio, then play) is correct but feels slow.
Perceived latency is reduced by streaming at each boundary:

- STT emits partial transcripts as the user speaks, finalized on VAD
  silence detection, so there is no need to wait after the user stops
  talking.
- LLM tokens stream via SSE as they are generated.
- The orchestrator buffers LLM tokens and, on each sentence boundary,
  immediately dispatches that sentence to TTS, so audio for sentence one
  starts playing while sentence two is still being generated.
- Barge-in: the orchestrator keeps listening via VAD during TTS playback.
  Detected speech immediately halts playback.

## 7. Hardware assumptions

Baseline target is a CPU-only consumer laptop. All three containerized
services are built and run without GPU acceleration by default, in
docker-compose.yml. An optional docker-compose.gpu.yml override adds
CUDA-enabled builds for llama.cpp and whisper.cpp on machines with an
NVIDIA GPU and nvidia-container-toolkit configured, without changing the
directory structure or service contracts.

## 8. Directory structure

See the repository root for the current layout. Summary:

- services/stt, services/llm, services/tts: Dockerized, one
  responsibility each.
- orchestrator: native process, owns audio I/O and pipeline state.
- dashboard: native Streamlit app, visualization only.
- shared/protocol: schemas shared across orchestrator and services.
- scripts: model download and latency benchmarking utilities.
- tests/integration: end-to-end tests spanning the full stack.
- docs/ARCHITECTURE.md: this document.

## 9. Build phases

The project is built in nine phases, tracked outside this document but
referenced here for context:

0. Environment and scaffolding
1. Services in isolation, with unit tests per service
2. Native audio I/O skeleton
3. Full round trip, non-streaming
4. Streaming and latency optimization
5. Wake word and interruption handling
6. Streamlit dashboard
7. Integration testing and hardening
8. Polish for portfolio

Each phase has an explicit exit condition and is completed in full,
including tests where applicable, before moving to the next.

## 10. Deferred work

### AMD GPU offload via Vulkan

The LLM service currently runs CPU-only. The dev laptop has an integrated
AMD GPU (496 MB dedicated, 7.7 GB shared system memory, not a discrete
card). CPU decode was confirmed memory-bandwidth-bound rather than
thread-bound (doubling threads from 4 to 8 produced no measurable change
in tokens per second).

A real path exists: llama.cpp's Vulkan backend can target AMD GPUs,
including integrated ones, and WSL2 exposes GPU access to containers
through Microsoft's DirectX-to-Vulkan translation layer. This would need:

- A different base image carrying Mesa's Vulkan driver.
- `llama-server` rebuilt with `GGML_VULKAN=1`.
- Docker Desktop's GPU passthrough enabled.
- A `docker-compose.gpu.yml` override, per the hardware assumptions in
  section 7.

Deferred rather than attempted now, since it is unproven on integrated
AMD hardware specifically and adds real setup risk mid-build. Revisit
during Phase 4 (streaming and latency optimization), where the actual
CPU-only latency numbers will show whether this is worth the effort.

### Barge-in detection: dual-signal gating

Initial barge-in detection used Silero VAD probability alone. Testing
with wired earbuds carrying an inline mic revealed a failure mode:
audio leaking from the earbud drivers into the inline mic (a known
acoustic coupling issue with that hardware pairing) registered as
high-confidence speech, VAD probability alone cannot distinguish "real
user speech" from "leaked assistant audio picked up by a mic
physically close to the output," since both are genuinely
speech-shaped signals.

Measured RMS energy across both cases showed a clear, wide separation:
leaked audio stayed below 0.007, genuine interruptions measured 0.03
and above. Barge-in detection now requires both VAD probability above
threshold and RMS energy above threshold to agree before triggering,
eliminating false interruptions from this failure mode while keeping
genuine interruptions responsive.

### tts_ttfa latency investigation

Dashboard showed tts_ttfa (time from turn start to first audio) at
2.74s in one exchange, notably higher than stt (0.92s) and llm_ttft
(0.50s). Isolated TTS calls (scripts/diagnose_tts_ttfa.py) measured
consistently under 0.6s with no warm-up cost across repeated calls,
ruling out the TTS service itself as the cause. Since tts_ttfa is
measured from turn start, it includes llm_ttft plus the time to
generate the remainder of the first sentence plus the TTS call
itself. The gap is explained by CPU LLM decode speed (roughly 11
tokens/second, measured in Phase 1), consistent with the length of
the first sentence in that exchange. No fix applied; this confirms
the CPU decode bottleneck already identified in Phase 1 rather than
revealing a new one.