"""Acoustic feature extraction. All features are speaker-normalised so that a baseline and a
participant with different voices / recording gains are comparable:
  * F0 in semitones relative to the speaker's own median F0
  * energy in dB relative to the speaker's own median speech level
  * cepstral mean subtraction on MFCCs
"""
from dataclasses import dataclass

import librosa
import numpy as np
import parselmouth

from . import SR

HOP = 160    # 10 ms
WIN = 400    # 25 ms
NFFT = 512   # 31.25 Hz bins at 16 kHz
HF_CUTOFF_HZ = 3000


@dataclass
class Features:
    sr: int
    t: np.ndarray          # frame centre times (s)
    f0_st: np.ndarray      # semitones re speaker median; NaN where unvoiced
    energy_db: np.ndarray  # dB re speaker median speech level
    hf_db: np.ndarray      # share of power above 3 kHz, in dB (clarity / brightness proxy)
    mfcc: np.ndarray       # (13, T), cepstral-mean-normalised
    active: np.ndarray     # speech-activity mask
    f0_median_hz: float
    f0_floor_hz: float
    f0_ceiling_hz: float


def _drop_octave_jumps(f0_st, win=11, max_dev=6.0):
    """Remove pitch-tracker octave errors: frames > 12 st from the speaker median (speech F0 rarely
    leaves +/-1 octave) or > 6 st from the local median."""
    f0_st = np.where(np.abs(f0_st) > 12.0, np.nan, f0_st)
    pad = win // 2
    x = np.pad(f0_st, (pad, pad), constant_values=np.nan)
    wins = np.lib.stride_tricks.sliding_window_view(x, win)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = np.nanmedian(wins, axis=1)
    out = f0_st.copy()
    out[np.abs(f0_st - local) > max_dev] = np.nan
    return out


def _adaptive_pitch_bounds(snd: parselmouth.Sound) -> tuple[float, float]:
    """Estimate stable F0 bounds from a wide first pass."""
    p1 = snd.to_pitch_ac(time_step=HOP / snd.sampling_frequency, pitch_floor=50, pitch_ceiling=600,
                         silence_threshold=0.01)
    f0_1 = p1.selected_array["frequency"]
    v1 = f0_1[f0_1 > 0]
    if len(v1) < 10:
        return 75.0, 500.0
    q25 = float(np.percentile(v1, 25))
    q75 = float(np.percentile(v1, 75))
    return max(45.0, 0.75 * q25), min(650.0, 2.5 * q75)


def extract(y: np.ndarray, sr: int = SR) -> Features:
    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP, center=True)[0]
    e_db = 20 * np.log10(rms + 1e-6)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, win_length=WIN, center=True)) ** 2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=NFFT, hop_length=HOP, win_length=WIN)
    T = min(len(e_db), S.shape[1], mfcc.shape[1])
    e_db, S, mfcc = e_db[:T], S[:, :T], mfcc[:, :T]
    t = np.arange(T) * HOP / sr

    floor = np.percentile(e_db, 10)
    active = e_db > floor + 12.0
    ref = np.median(e_db[active]) if active.any() else np.median(e_db)
    energy_db = e_db - ref

    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    hf_db = 10 * np.log10(S[freqs >= HF_CUTOFF_HZ].sum(0) / (S.sum(0) + 1e-12) + 1e-9)

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    # de Looze & Hirst (2014) two-pass pitch estimation.
    floor, ceiling = _adaptive_pitch_bounds(snd)

    # Pass 2: Re-estimate with speaker-adapted boundaries
    pitch = snd.to_pitch_ac(time_step=HOP / sr, pitch_floor=floor, pitch_ceiling=ceiling,
                            silence_threshold=0.005)  # low: keep quiet passages voiced
    f0 = pitch.selected_array["frequency"]
    px = pitch.xs()
    idx = np.clip(np.searchsorted(px, t), 0, len(px) - 1)
    f0g = f0[idx]
    voiced = f0g > 0
    med = float(np.median(f0g[voiced])) if voiced.any() else 120.0
    f0_st = np.full(T, np.nan)
    f0_st[voiced] = 12 * np.log2(f0g[voiced] / med)
    f0_st = _drop_octave_jumps(f0_st)

    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)
    return Features(sr, t, f0_st, energy_db, hf_db, mfcc, active, med, floor, ceiling)


def _sl(t, a, b):
    i0 = int(np.searchsorted(t, a))
    i1 = int(np.searchsorted(t, b))
    return slice(i0, max(i1, i0 + 1))


def word_table(feat: Features, words: list[dict]) -> dict[str, np.ndarray]:
    """Per-word measurements aligned to the transcript (one entry per word)."""
    n = len(words)
    dur = np.array([w["end"] - w["start"] for w in words])
    gap = np.zeros(n)
    for i in range(1, n):
        gap[i] = max(0.0, words[i]["start"] - words[i - 1]["end"])
    energy = np.zeros(n)
    hf = np.zeros(n)
    f0_vals = []
    for i, w in enumerate(words):
        sl = _sl(feat.t, w["start"], w["end"])
        energy[i] = np.mean(feat.energy_db[sl])
        hf[i] = np.mean(feat.hf_db[sl])
        v = feat.f0_st[sl]
        loud = feat.energy_db[sl] >= feat.energy_db[sl].max() - 20.0  # ignore low-level tails/noise
        f0_vals.append(v[loud & ~np.isnan(v)])
    # pitch range (p90-p10, semitones) pooled over a 3-word window: single words are too short
    prange = np.full(n, np.nan)
    pmed = np.full(n, np.nan)
    for i in range(n):
        pool = np.concatenate(f0_vals[max(0, i - 1): i + 2])
        if len(pool) >= 8:
            prange[i] = np.percentile(pool, 90) - np.percentile(pool, 10)
            pmed[i] = np.median(pool)
    return {"dur": dur, "gap": gap, "energy": energy, "hf": hf, "pitch_range": prange, "pitch_med": pmed}
