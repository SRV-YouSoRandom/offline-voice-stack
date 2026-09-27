import asyncio
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from orchestrator.audio_io import AudioIO
from orchestrator.clients import llm_client, stt_client, tts_client
from orchestrator.config import Settings
from orchestrator.event_bus import EventBus
from orchestrator.recorder import float32_to_pcm16
from orchestrator.sentence_chunker import SentenceChunker
from orchestrator.vad import SileroVAD
from orchestrator.wake_word import WakeWordDetector

WAITING = "waiting_for_wake_word"
RECORDING = "recording"
SPEAKING = "speaking"


def _llm_stream_worker(transcript, uri, token_queue, interrupt_event):
    try:
        for token in llm_client.stream_chat(transcript, uri):
            if interrupt_event.is_set():
                break
            token_queue.put(token)
    finally:
        token_queue.put(None)


async def produce_sentences(transcript, settings, sentence_queue, interrupt_event, event_bus, turn_start):
    token_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_llm_stream_worker,
        args=(transcript, settings.llm_uri, token_queue, interrupt_event),
        daemon=True,
    )
    thread.start()

    chunker = SentenceChunker()
    loop = asyncio.get_running_loop()
    reply_parts = []
    first_token_seen = False

    while True:
        if interrupt_event.is_set():
            break

        token = await loop.run_in_executor(None, token_queue.get)
        if token is None:
            break

        if not first_token_seen:
            first_token_seen = True
            await event_bus.publish({
                "type": "latency",
                "stage": "llm_ttft",
                "seconds": time.perf_counter() - turn_start,
            })

        reply_parts.append(token)
        for sentence in chunker.feed(token):
            await sentence_queue.put(sentence)
            await event_bus.publish({"type": "reply_partial", "text": "".join(reply_parts)})

    remaining = chunker.flush()
    if remaining and not interrupt_event.is_set():
        await sentence_queue.put(remaining)

    await sentence_queue.put(None)

    if reply_parts:
        full_reply = "".join(reply_parts)
        print(f"assistant: {full_reply}")
        await event_bus.publish({"type": "assistant_reply", "text": full_reply})


async def synthesize_sentences(sentence_queue, audio_queue, settings, interrupt_event, event_bus, turn_start):
    first_audio_seen = False

    while True:
        sentence = await sentence_queue.get()
        if sentence is None or interrupt_event.is_set():
            await audio_queue.put(None)
            break

        samples, sample_rate = await tts_client.synthesize(sentence, settings.tts_uri)

        if not first_audio_seen:
            first_audio_seen = True
            await event_bus.publish({
                "type": "latency",
                "stage": "tts_ttfa",
                "seconds": time.perf_counter() - turn_start,
            })

        await audio_queue.put((samples, sample_rate))


def _play_blocking(samples, sample_rate):
    sd.play(samples, sample_rate)
    sd.wait()


async def play_audio(audio_queue, interrupt_event):
    loop = asyncio.get_running_loop()

    while True:
        item = await audio_queue.get()
        if item is None or interrupt_event.is_set():
            break

        samples, sample_rate = item
        await loop.run_in_executor(None, _play_blocking, samples, sample_rate)


async def speak_reply(transcript, settings, interrupt_event, event_bus, turn_start):
    sentence_queue: asyncio.Queue = asyncio.Queue()
    audio_queue: asyncio.Queue = asyncio.Queue()

    await asyncio.gather(
        produce_sentences(transcript, settings, sentence_queue, interrupt_event, event_bus, turn_start),
        synthesize_sentences(sentence_queue, audio_queue, settings, interrupt_event, event_bus, turn_start),
        play_audio(audio_queue, interrupt_event),
    )


async def run_pipeline(settings: Settings, event_bus: EventBus) -> None:
    vad = SileroVAD(settings.vad_model_path, threshold=settings.vad_threshold)
    wake_word = WakeWordDetector(threshold=settings.wake_word_threshold)
    audio = AudioIO()

    print("wake word pipeline, press ctrl+c to stop")
    print("say the wake word to start")
    await event_bus.publish({"type": "state", "value": WAITING})

    audio.start()
    loop = asyncio.get_running_loop()

    state = WAITING
    utterance_frames: list = []
    pre_buffer: list = []
    consecutive_speech = 0
    consecutive_silence = 0
    speaking_task: asyncio.Task | None = None
    interrupt_event = threading.Event()
    turn_start = 0.0

    try:
        while True:
            try:
                frame = await loop.run_in_executor(None, audio.read_frame, 1.0)
            except queue.Empty:
                continue

            if state == WAITING:
                detected = wake_word.feed(frame)
                if detected:
                    print(f"wake word detected: {detected}")
                    await event_bus.publish({"type": "wake_word", "name": detected})
                    state = RECORDING
                    await event_bus.publish({"type": "state", "value": RECORDING})
                    vad.reset()
                    utterance_frames = []
                    pre_buffer = []
                    consecutive_speech = 0
                    consecutive_silence = 0
                    print("listening...")

            elif state == RECORDING:
                probability = vad.process_frame(frame)

                if probability >= settings.vad_threshold:
                    consecutive_speech += 1
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1
                    consecutive_speech = 0

                if len(utterance_frames) == 0:
                    pre_buffer.append(frame)
                    if len(pre_buffer) > settings.speech_onset_frames:
                        pre_buffer.pop(0)

                if consecutive_speech >= settings.speech_onset_frames and len(utterance_frames) == 0:
                    utterance_frames.extend(pre_buffer)
                    print("speech confirmed")

                if len(utterance_frames) > 0:
                    utterance_frames.append(frame)

                if len(utterance_frames) > 0 and consecutive_silence >= settings.speech_offset_frames:
                    print("speech ended")
                    utterance = np.concatenate(utterance_frames)
                    pcm_bytes = float32_to_pcm16(utterance)

                    stt_start = time.perf_counter()
                    print("transcribing...")

                    try:
                        transcript = await stt_client.transcribe(pcm_bytes, settings.stt_uri)
                    except RuntimeError as exc:
                        print(f"stt failed: {exc}")
                        await event_bus.publish({"type": "error", "stage": "stt", "message": str(exc)})
                        state = WAITING
                        await event_bus.publish({"type": "state", "value": WAITING})
                        print("say the wake word to start")
                        continue

                    await event_bus.publish({
                        "type": "latency",
                        "stage": "stt",
                        "seconds": time.perf_counter() - stt_start,
                    })
                    print(f"you said: {transcript}")
                    await event_bus.publish({"type": "transcript", "text": transcript})

                    if transcript.strip():
                        print("thinking...")
                        interrupt_event = threading.Event()
                        turn_start = time.perf_counter()
                        state = SPEAKING
                        await event_bus.publish({"type": "state", "value": SPEAKING})
                        speaking_task = asyncio.create_task(
                            speak_reply(transcript, settings, interrupt_event, event_bus, turn_start)
                        )
                        vad.reset()
                        consecutive_speech = 0
                        consecutive_silence = 0
                    else:
                        print("no speech detected, listening for wake word again")
                        state = WAITING
                        await event_bus.publish({"type": "state", "value": WAITING})

            elif state == SPEAKING:
                probability = vad.process_frame(frame)
                rms = float(np.sqrt(np.mean(np.square(frame))))

                if probability >= settings.barge_in_threshold and rms >= settings.barge_in_rms_threshold:
                    consecutive_speech += 1
                else:
                    consecutive_speech = 0

                if consecutive_speech >= settings.barge_in_onset_frames and not interrupt_event.is_set():
                    print("interrupted, stopping playback")
                    interrupt_event.set()
                    sd.stop()
                    await event_bus.publish({"type": "interrupted"})

                if speaking_task.done():
                    task_exception = speaking_task.exception()
                    if task_exception is not None:
                        print(f"speak_reply failed: {task_exception}")
                        await event_bus.publish({
                            "type": "error",
                            "stage": "speak_reply",
                            "message": str(task_exception),
                        })

                    if interrupt_event.is_set():
                        state = RECORDING
                        await event_bus.publish({"type": "state", "value": RECORDING})
                        vad.reset()
                        utterance_frames = []
                        pre_buffer = []
                        consecutive_speech = 0
                        consecutive_silence = 0
                        print("listening...")
                    else:
                        state = WAITING
                        await event_bus.publish({"type": "state", "value": WAITING})
                        print("say the wake word to start")

    finally:
        audio.stop()


async def main() -> None:
    settings = Settings.from_env()
    event_bus = EventBus()

    await asyncio.gather(
        run_pipeline(settings, event_bus),
        event_bus.serve(settings.event_bus_host, settings.event_bus_port),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopping")