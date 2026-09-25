import asyncio

import sounddevice as sd

from orchestrator.audio_io import AudioIO
from orchestrator.clients import llm_client, stt_client, tts_client
from orchestrator.config import Settings
from orchestrator.recorder import float32_to_pcm16, record_utterance
from orchestrator.vad import SileroVAD


async def run_turn(audio: AudioIO, vad: SileroVAD, settings: Settings) -> None:
    utterance = record_utterance(audio, vad, settings)
    audio.stop()

    pcm_bytes = float32_to_pcm16(utterance)

    print("transcribing...")
    transcript = await stt_client.transcribe(pcm_bytes, settings.stt_uri)
    print(f"you said: {transcript}")

    if not transcript.strip():
        print("no speech detected, skipping")
        return

    print("thinking...")
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, llm_client.chat, transcript, settings.llm_uri)
    print(f"assistant: {reply}")

    print("synthesizing...")
    samples, sample_rate = await tts_client.synthesize(reply, settings.tts_uri)

    sd.play(samples, sample_rate)
    sd.wait()


async def main() -> None:
    settings = Settings.from_env()
    vad = SileroVAD(settings.vad_model_path, threshold=settings.vad_threshold)
    audio = AudioIO()

    print("full round trip pipeline, press ctrl+c to stop")

    try:
        while True:
            audio.start()
            try:
                await run_turn(audio, vad, settings)
            finally:
                audio.stop()
            vad.reset()
    except KeyboardInterrupt:
        print("stopping")


if __name__ == "__main__":
    asyncio.run(main())