import numpy as np
import librosa
import soundfile as sf

from . import SR


def load_audio(path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def save_audio(path, y, sr: int = SR):
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
