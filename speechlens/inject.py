"""Deterministic delivery-flaw injection with exact ground-truth labels.

Every flaw is applied to a span of whole words (or a word boundary for pauses), so the
injected region and the new word timings are known exactly -- no re-alignment needed.
Severity 1..4 runs from "almost perfect" (barely perceptible) to "botched".
"""
import io
import subprocess
import zlib

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from . import SR

FLAWS = ["rushed", "dragging", "monotone", "long_pause", "volume_drop", "muffled"]

# parameter per severity level 1..4
SEVERITY = {
    "rushed":      [1.15, 1.35, 1.65, 2.0],     # speed-up factor
    "dragging":    [1.0 / 1.15, 1.0 / 1.35, 1.0 / 1.65, 1.0 / 2.0],
    "monotone":    [0.2, 0.45, 0.7, 0.9],       # fraction of pitch excursion removed
    "long_pause":  [0.4, 0.8, 1.4, 2.2],        # seconds of silence inserted
    "volume_drop": [-3.0, -6.0, -10.0, -16.0],  # dB
    "muffled":     [3800.0, 3600.0, 3350.0, 3000.0],  # low-pass cutoff Hz
}


def seed_for(*parts) -> int:
    """Process-independent seed (Python's hash() is randomised per run)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def _fade(x, n):
    n = min(n, len(x) // 2)
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


def _room_noise(y, n, rng):
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    return rng.normal(0, np.percentile(rms, 10), n).astype(np.float32)


def _monotone(seg, sr, k):
    snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=sr)
    manip = call(snd, "To Manipulation", 0.01, 75, 500)
    tier = call(manip, "Extract pitch tier")
    n = call(tier, "Get number of points")
    if n < 3:
        return seg
    times = np.array([call(tier, "Get time from index", i) for i in range(1, n + 1)])
    f0 = np.array([call(tier, "Get value at index", i) for i in range(1, n + 1)])
    med = np.median(f0)
    new = med * (f0 / med) ** (1.0 - k)
    flat = call("Create PitchTier", "flat", snd.xmin, snd.xmax)
    for t, f in zip(times, new):
        call(flat, "Add point", float(t), float(f))
    call([flat, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    if len(out) < len(seg):
        out = np.pad(out, (0, len(seg) - len(out)))
    return out[:len(seg)]


def _atempo(seg, rate, sr):
    """Change duration with ffmpeg's time-domain atempo filter, preserving pitch."""
    if not 0.5 <= rate <= 2.0:
        raise ValueError(f"atempo rate must be within [0.5, 2.0], got {rate}")
    source = io.BytesIO()
    sf.write(source, seg, sr, format="WAV", subtype="FLOAT")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-filter:a", f"atempo={rate}",
             "-ar", str(sr), "-ac", "1", "-f", "wav", "pipe:1"],
            input=source.getvalue(), capture_output=True, check=True,
        )
    except FileNotFoundError as err:
        raise RuntimeError("ffmpeg is required for rushed and dragging injection") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"ffmpeg atempo failed: {err.stderr.decode(errors='replace').strip()}") from err
    out, out_sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    if out_sr != sr or len(out) == 0:
        raise RuntimeError("ffmpeg atempo produced invalid audio")
    return out


def _process(seg, flaw, param, sr):
    if flaw in ("rushed", "dragging"):
        return _atempo(seg, float(param), sr)
    if flaw == "volume_drop":
        g = 10 ** (param / 20)
        ramp = int(0.02 * sr)
        env = np.full(len(seg), g, dtype=np.float32)
        if ramp * 2 < len(seg):
            env[:ramp] = np.linspace(1, g, ramp)
            env[-ramp:] = np.linspace(g, 1, ramp)
        return seg * env
    if flaw == "muffled":
        sos = butter(6, param / (sr / 2), btype="low", output="sos")
        return sosfiltfilt(sos, seg).astype(np.float32)
    if flaw == "monotone":
        return _monotone(seg, sr, param)
    raise ValueError(flaw)


def apply_flaw(y, words, flaw, severity, i, j, rng, sr=SR):
    """Apply one flaw. Returns (audio, words, region, delta_seconds). Word indices never change."""
    if not (1 <= severity <= 4):
        raise ValueError(f"severity must be 1-4, got {severity}")
    param = SEVERITY[flaw][severity - 1]
    words = [dict(w) for w in words]
    if flaw == "long_pause":
        if i < 1:
            raise ValueError("Cannot insert long_pause before the first word (i=0)")
        prev_end, nxt = words[i - 1]["end"], words[i]["start"]
        b = int(round((prev_end + nxt) / 2 * sr))
        ins = _room_noise(y, int(param * sr), rng)
        y2 = np.concatenate([y[:b], ins, y[b:]])
        for k in range(i, len(words)):
            words[k]["start"] += param
            words[k]["end"] += param
        region = dict(type=flaw, severity=severity, start=prev_end, end=nxt + param,
                      word_start=i, word_end=i, base_start=prev_end, base_end=nxt)
        return y2, words, region, param

    s, e = words[i]["start"], words[j]["end"]
    a, b = int(round(s * sr)), int(round(e * sr))
    seg = y[a:b]
    if len(seg) == 0:
        raise ValueError(f"cannot inject {flaw} into empty word span {i}..{j}")
    seg2 = _fade(_process(seg, flaw, param, sr), int(0.005 * sr))
    scale = len(seg2) / len(seg)
    for k in range(i, j + 1):
        words[k]["start"] = s + (words[k]["start"] - s) * scale
        words[k]["end"] = s + (words[k]["end"] - s) * scale
    delta = (len(seg2) - len(seg)) / sr
    for k in range(j + 1, len(words)):
        words[k]["start"] += delta
        words[k]["end"] += delta
    y2 = np.concatenate([y[:a], seg2, y[b:]])
    region = dict(type=flaw, severity=severity, start=s, end=s + len(seg2) / sr,
                  word_start=i, word_end=j, base_start=s, base_end=e)
    return y2.astype(np.float32), words, region, delta


def choose_span(words, rng, lo=6, hi=10, margin=3, min_dur=1.5, max_inner_gap=None):
    """Pick a span of words [i, j] in the body of the speech with no long internal pause."""
    n = len(words)
    if n < 2 * margin + hi + 1:
        raise ValueError(f"too few words for span selection: {n}")
    if max_inner_gap is None:
        gaps_all = [words[k]["start"] - words[k - 1]["end"] for k in range(1, n)]
        max_inner_gap = max(0.4, 3 * float(np.median(gaps_all)))
    for _ in range(200):
        L = int(rng.integers(lo, hi + 1))
        i = int(rng.integers(margin, n - L - margin))
        j = i + L - 1
        dur = words[j]["end"] - words[i]["start"]
        gaps = [words[k]["start"] - words[k - 1]["end"] for k in range(i + 1, j + 1)]
        if dur >= min_dur and max(gaps) <= max_inner_gap:
            return i, j
    raise RuntimeError("no valid span found")


def choose_boundary(words, rng, margin=4, max_gap=0.2):
    """Pick a mid-phrase word boundary (no existing pause) for a pause insertion."""
    n = len(words)
    for _ in range(200):
        i = int(rng.integers(margin, n - margin))
        if words[i]["start"] - words[i - 1]["end"] <= max_gap:
            return i
    return margin


def make_flawed(y, words, plan, sr=SR):
    """plan: list of dicts {flaw, severity, i, j, seed}; spans must be disjoint.
    Applied from the end of the speech backwards so earlier indices stay valid.
    Returns (audio, words, regions) with regions in the *flawed* timeline."""
    orig = [dict(w) for w in words]
    regions = []
    cur_words = [dict(w) for w in words]
    for step in sorted(plan, key=lambda p: -p["i"]):
        rng = np.random.default_rng(step["seed"])
        y, cur_words, region, delta = apply_flaw(
            y, cur_words, step["flaw"], step["severity"], step["i"], step["j"], rng, sr)
        for r in regions:  # earlier-applied regions all lie after this one: shift them
            r["start"] += delta
            r["end"] += delta
        region["base_start"] = orig[step["i"] - 1]["end"] if step["flaw"] == "long_pause" else orig[step["i"]]["start"]
        region["base_end"] = orig[step["i"]]["start"] if step["flaw"] == "long_pause" else orig[step["j"]]["end"]
        regions.append(region)
    regions.sort(key=lambda r: r["start"])
    return y, cur_words, regions
