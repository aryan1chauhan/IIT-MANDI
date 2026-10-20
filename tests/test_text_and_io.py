import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import soundfile as sf

from speechlens.io import load_audio, save_audio
from speechlens.text import normalize_words


def test_normalize_words_handles_smart_quotes():
    text = "It’s wonderful, don‘t you think? `SpeechLens` is live."
    words = normalize_words(text)
    assert words == ["it's", "wonderful", "don't", "you", "think", "speechlens", "is", "live"]


def test_normalize_words_rejects_numeric_digits():
    with pytest.raises(ValueError, match="numbers must be spelled out"):
        normalize_words("The score was 100 to 90.")


def test_load_audio_warns_on_low_sample_rate(tmp_path):
    low_sr_path = tmp_path / "low_sr.wav"
    # Create an 8 kHz mono wav file
    y_8k = np.zeros(8000, dtype=np.float32)
    sf.write(str(low_sr_path), y_8k, 8000)

    with pytest.warns(UserWarning, match="Audio sample rate.*below target 16000 Hz"):
        y = load_audio(low_sr_path)
    # Output should still be resampled to 16 kHz
    assert len(y) == 16000


def test_load_audio_clean_on_16k(tmp_path):
    clean_path = tmp_path / "clean_16k.wav"
    y_16k = np.zeros(16000, dtype=np.float32)
    save_audio(clean_path, y_16k)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = load_audio(clean_path)
    assert len(y) == 16000
