import sys

from orchestrator.audio_io import AudioIO
from orchestrator.config import Settings
from orchestrator.vad import SileroVAD


def main() -> None:
    settings = Settings.from_env()

    vad = SileroVAD(settings.vad_model_path, threshold=settings.vad_threshold)
    audio = AudioIO()

    speaking = False
    consecutive_speech = 0
    consecutive_silence = 0

    print("starting audio loop, press ctrl+c to stop")
    audio.start()

    try:
        while True:
            frame = audio.read_frame(timeout=1.0)
            probability = vad.process_frame(frame)

            if probability >= settings.vad_threshold:
                consecutive_speech += 1
                consecutive_silence = 0
            else:
                consecutive_silence += 1
                consecutive_speech = 0

            if not speaking and consecutive_speech >= settings.speech_onset_frames:
                speaking = True
                print(f"speech started (p={probability:.2f})")

            if speaking and consecutive_silence >= settings.speech_offset_frames:
                speaking = False
                print(f"speech ended (p={probability:.2f})")

    except KeyboardInterrupt:
        print("stopping")
    finally:
        audio.stop()


if __name__ == "__main__":
    sys.exit(main() or 0)