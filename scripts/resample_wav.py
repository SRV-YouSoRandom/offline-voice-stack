import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample_to_16k_mono(input_path: str, output_path: str) -> None:
    data, sr = sf.read(input_path, always_2d=True)
    mono = data.mean(axis=1)

    if sr != 16000:
        gcd = np.gcd(sr, 16000)
        up = 16000 // gcd
        down = sr // gcd
        mono = resample_poly(mono, up, down)

    mono = np.clip(mono, -1.0, 1.0)
    pcm16 = (mono * 32767).astype(np.int16)
    sf.write(output_path, pcm16, 16000, subtype="PCM_16")
    print(f"wrote {output_path}: 16000 Hz, mono, 16-bit PCM")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: resample_wav.py <input> <output>")
        sys.exit(1)

    resample_to_16k_mono(sys.argv[1], sys.argv[2])