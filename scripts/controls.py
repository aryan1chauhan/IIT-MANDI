# scripts/controls.py: no-flaw controls. Every detected region is a false positive.
import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys
import zlib

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechlens.align import align_words
from speechlens.analyze import analyze
from speechlens.io import load_audio
from speechlens.text import load_alignment

SR = 16000   # assumes load_audio returns 16 kHz mono float32 (MMS_FA requires 16 kHz)


def _best_alignment(data, stem, subdir="alignments"):
    """Prefer MMS_FA alignment (matched provenance) over eSpeak ground-truth."""
    mmsfa = data / subdir / f"{stem}_mmsfa.json"
    plain = data / subdir / f"{stem}.json"
    if mmsfa.exists():
        return load_alignment(mmsfa)
    return load_alignment(plain) if plain.exists() else None


def find_clean_baselines(data_dir="data"):
    data, out = Path(data_dir), []
    for p in (data / "baseline").glob("*.wav"):
        out.append(dict(id=p.stem, path=str(p), words=_best_alignment(data, p.stem),
                        transcript=(data / "transcripts" / f"{p.stem}.txt").read_text(encoding="utf-8")))
    for p in (data / "real" / "baseline").glob("*.wav"):
        t = data / "real" / "transcripts" / f"{p.stem}.txt"
        t = t if t.exists() else data / "transcripts" / f"{p.stem}.txt"
        out.append(dict(id=p.stem, path=str(p), words=_best_alignment(data, p.stem, "real/alignments"),
                        transcript=t.read_text(encoding="utf-8")))
    return out


# ---- transforms --------------------------------------------------------------
def _ff(args, data):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", *args, "pipe:1"],
                          input=data, capture_output=True, check=True).stdout


def _wav_bytes(y):
    b = io.BytesIO()
    sf.write(b, y, SR, format="WAV", subtype="FLOAT")
    return b.getvalue()


def _from_wav(b):
    return sf.read(io.BytesIO(b))[0].astype(np.float32)


def tempo(y, r):
    return _from_wav(_ff(["-filter:a", f"atempo={r}", "-ar", str(SR), "-ac", "1", "-f", "wav"], _wav_bytes(y)))


def pitch(y, st):
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, 50, 500)
    pt = call(manip, "Extract pitch tier")
    call(pt, "Multiply frequencies", 0.0, snd.duration, 2.0**(st / 12.0))
    call([manip, pt], "Replace pitch tier")
    resynth = call(manip, "Get resynthesis (overlap-add)")
    y_out = resynth.values[0].astype(np.float32)
    return y_out[:len(y)] if len(y_out) >= len(y) else np.pad(y_out, (0, len(y) - len(y_out)))


def mp3(y, kbps=64):
    enc = _ff(["-b:a", f"{kbps}k", "-f", "mp3"], _wav_bytes(y))
    return _from_wav(_ff(["-ar", str(SR), "-ac", "1", "-f", "wav"], enc))


def noise(y, snr_db, rng):
    n = rng.standard_normal(len(y)) * np.sqrt(np.mean(y**2) / 10**(snr_db / 10))
    return (y + n).astype(np.float32)


def gain(y, db):
    z = y * 10**(db / 20)
    return None if np.max(np.abs(z)) > 1.0 else z.astype(np.float32)   # None = would clip, skip


CONTROLS = {
    "identity":   lambda y, r: y,
    "tempo_0.9":  lambda y, r: tempo(y, 0.9),   "tempo_1.1": lambda y, r: tempo(y, 1.1),
    "pitch_-3":   lambda y, r: pitch(y, -3),    "pitch_+3":  lambda y, r: pitch(y, 3),
    "mp3_64k":    lambda y, r: mp3(y),
    "noise_20dB": lambda y, r: noise(y, 20, r), "noise_10dB": lambda y, r: noise(y, 10, r),
    "gain_-6":    lambda y, r: gain(y, -6),     "gain_+6":    lambda y, r: gain(y, 6),
}


def measure_lag_s(ref, y, max_s=0.1):
    n = min(3 * SR, len(ref), len(y))
    m = int(max_s * SR)
    c = np.correlate(y[:n], ref[m:n - m], mode="valid")
    return (int(np.argmax(c)) - m) / SR


def analytic_alignment(words, name, lag_s):
    if name.startswith("tempo_"):
        r = float(name.split("_")[1])
        return [{**w, "start": w["start"] / r, "end": w["end"] / r} for w in words]
    if name.startswith("mp3"):
        return [{**w, "start": w["start"] + lag_s, "end": w["end"] + lag_s} for w in words]
    return words


# ---- run ---------------------------------------------------------------------
def main(data_dir, out_dir, tmp="data/_controls_tmp"):
    Path(tmp).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows, fps = [], []
    for b in find_clean_baselines(data_dir):
        y_b = load_audio(b["path"]).astype(np.float32)
        base_words = b["words"] or align_words(b["path"], b["transcript"])
        for name, fn in CONTROLS.items():
            rng = np.random.default_rng(zlib.crc32(f"{b['id']}|{name}".encode()))
            z = fn(y_b, rng)
            if z is None:
                print(f"SKIP {b['id']} {name}: would clip")
                continue
            p = Path(tmp) / f"{b['id']}__{name}.wav"
            sf.write(str(p), z, SR)
            lag = measure_lag_s(y_b, z) if name.startswith("mp3") else 0.0
            y_p = load_audio(str(p))
            for kind, pw in (("analytic", analytic_alignment(base_words, name, lag)),
                             ("realigned", align_words(str(p), b["transcript"]))):
                res = analyze(y_p, pw, y_b, base_words)
                regs, sc = res["regions"], res["scores"]
                rows.append(dict(id=b["id"], control=name, path=kind, lag_s=round(lag, 4),
                                 n_fp=len(regs), overall=sc["overall"]))
                for r in regs:
                    w_start, w_end = r["word_start"], r["word_end"]
                    dur_p = [pw[k]["end"] - pw[k]["start"] for k in range(w_start, w_end + 1)]
                    dur_b = [base_words[k]["end"] - base_words[k]["start"] for k in range(w_start, w_end + 1)]
                    n_le_40ms = sum(dp <= 0.040 or db <= 0.040 for dp, db in zip(dur_p, dur_b))
                    fps.append(dict(id=b["id"], control=name, path=kind, type=r["type"],
                                    start=round(r["start"], 3), end=round(r["end"], 3),
                                    base_dur_ms=round(1000 * (r["base_end"] - r["base_start"])),
                                    magnitude=round(r["magnitude"], 2),
                                    n_words=len(dur_p),
                                    n_words_le_40ms=n_le_40ms))

    row_fields = ["id", "control", "path", "lag_s", "n_fp", "overall"]
    fp_fields = ["id", "control", "path", "type", "start", "end", "base_dur_ms", "magnitude", "n_words", "n_words_le_40ms"]
    for fname, d, fields in (("controls_rows.csv", rows, row_fields), ("controls_fp.csv", fps, fp_fields)):
        with open(Path(out_dir) / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            if d:
                w.writerows(d)

    summarize(rows, fps, Path(out_dir) / "controls_matched.txt")


def summarize(rows, fps, out_path):
    L = ["control        path       files  FP_total  files_with_FP  mean_overall"]
    for c in CONTROLS:
        for pk in ("analytic", "realigned"):
            s = [r for r in rows if r["control"] == c and r["path"] == pk]
            if s:
                L.append(f"{c:<14} {pk:<10} {len(s):>5} {sum(r['n_fp'] for r in s):>9} "
                         f"{sum(r['n_fp'] > 0 for r in s):>14} {np.mean([r['overall'] for r in s]):>12.1f}")
    for pk in ("analytic", "realigned"):
        L += ["", f"FPs by type, {pk}:"]
        types = sorted({f["type"] for f in fps})
        L += [f"  {t}: {sum(f['type'] == t and f['path'] == pk for f in fps)}" for t in types]
    for pk in ("analytic", "realigned"):
        L += ["", f"Flagged words with duration <= 40 ms on either side ({pk}):"]
        total_w = sum(f.get("n_words", 0) for f in fps if f["path"] == pk)
        total_short = sum(f.get("n_words_le_40ms", 0) for f in fps if f["path"] == pk)
        rate_w = sum(f.get("n_words", 0) for f in fps if f["path"] == pk and f["type"] in ("rushed", "dragging"))
        rate_short = sum(f.get("n_words_le_40ms", 0) for f in fps if f["path"] == pk and f["type"] in ("rushed", "dragging"))
        L.append(f"  All types: {total_short}/{total_w} words ({100*total_short/max(total_w, 1):.1f}%) <= 40 ms on either side")
        L.append(f"  Rate only: {rate_short}/{rate_w} words ({100*rate_short/max(rate_w, 1):.1f}%) <= 40 ms on either side")
    L += ["", "Realigned FPs by baseline duration (ms), all types, and rate only:"]
    for lo, hi in ((0, 60), (60, 120), (120, 10**6)):
        sel = [f for f in fps if f["path"] == "realigned" and lo <= f["base_dur_ms"] < hi]
        L.append(f"  {lo}-{hi if hi < 10**6 else 'inf'}: {len(sel)} (rate: {sum(f['type'] == 'rushed' or f['type'] == 'dragging' for f in sel)})")
    L += ["", "FP regions (file, control, path, type, start-end, base_ms, magnitude, words, <=40ms):"]
    L += [f"  {f['id']}  {f['control']}  {f['path']}  {f['type']}  {f['start']}-{f['end']}  {f['base_dur_ms']}  {f['magnitude']}  words={f.get('n_words', 0)}  short={f.get('n_words_le_40ms', 0)}" for f in fps]
    out_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="eval")
    a = ap.parse_args()
    main(a.data, a.out)
