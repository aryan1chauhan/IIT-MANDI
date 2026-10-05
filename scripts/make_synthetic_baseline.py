"""Build a SYNTHETIC test baseline with exact word timings (espeak-ng, word by word).

This is a unit-test fixture for the pipeline, NOT the dataset: real baselines are recordings of
public-domain/openly-licensed speeches, aligned with speechlens.align.
Requires the `espeak-ng` binary on PATH.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechlens import SR  # noqa: E402
from speechlens.io import save_audio  # noqa: E402
from speechlens.text import save_alignment  # noqa: E402

TEXT = (
    "We gather today, not to mourn what was lost, but to build what comes next. "
    "Every great effort begins with a small decision, and a stubborn refusal to turn back. "
    "When the road ahead is steep, we do not ask whether the climb is easy, we ask whether it is worth it. "
    "It is worth it, because the people who follow us will inherit whatever we choose to make. "
    "So let us work with patience, let us listen with care, and let us speak with courage. "
    "The future is not a place we are waiting to reach, it is a promise we keep every single day. "
    "Let us begin now, together, and let us not stop until the work is done."
)


def render_word(word, pitch, speed, tmp):
    wav = Path(tmp) / "w.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", str(speed), "-p", str(pitch), "-a", "120",
                    "-w", str(wav), word], check=True, capture_output=True)
    y, sr = sf.read(wav)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    act = np.where(np.abs(y) > 0.02 * np.abs(y).max())[0]  # trim silence around the word
    return y[act[0]: act[-1] + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--id", default="synthetic01")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    tokens = TEXT.split()
    audio, words, t = [np.zeros(int(0.5 * SR), np.float32)], [], 0.5
    phrase_pos = 0
    with tempfile.TemporaryDirectory() as tmp:
        for tok in tokens:
            w = "".join(c for c in tok.lower() if c.isalpha() or c == "'")
            pitch = int(np.clip(50 + 18 * np.sin(phrase_pos * 0.9) + rng.normal(0, 6), 20, 80))
            y = render_word(w, pitch, 150, tmp)
            words.append({"word": w, "start": t, "end": t + len(y) / SR})
            audio.append(y)
            t += len(y) / SR
            end_punct = tok[-1] in ".,"
            gap = rng.uniform(0.5, 0.7) if tok[-1] == "." else rng.uniform(0.3, 0.45) if tok[-1] == "," \
                else rng.uniform(0.04, 0.09)
            phrase_pos = 0 if end_punct else phrase_pos + 1
            audio.append(np.zeros(int(gap * SR), np.float32))
            t += gap
    y = np.concatenate(audio + [np.zeros(int(0.5 * SR), np.float32)])
    y = y + rng.normal(0, 0.001, len(y)).astype(np.float32)  # faint room noise
    y = 0.6 * y / np.abs(y).max()
    (out / "baseline").mkdir(parents=True, exist_ok=True)
    (out / "alignments").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    save_audio(out / "baseline" / f"{a.id}.wav", y)
    save_alignment(out / "alignments" / f"{a.id}.json", words)
    (out / "transcripts" / f"{a.id}.txt").write_text(TEXT, encoding="utf-8")
    print(f"{a.id}: {len(words)} words, {len(y) / SR:.1f} s")


if __name__ == "__main__":
    main()
