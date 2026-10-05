"""Baseline-vs-participant comparison: temporal flaw grounding, causal explanations, rubric scores.

Method
------
Both recordings are forced-aligned to the same transcript, so words are exact anchors between
the two timelines (no DTW needed). For every word i we compute a delta between participant (P)
and baseline (B):

  rate     d = ln(dur_P / dur_B)                       (<0 rushed, >0 dragging)
  pause    d = gap_P - gap_B            [s]            (gap = silence before the word)
  pitch    d = ln((range_P+0.5)/(range_B+0.5))         (<0 monotone, >0 erratic); range = p90-p10 in
                                                       semitones over a 3-word window
  energy   d = E_P - E_B               [dB]            (speaker-normalised levels)
  clarity  d = HF_P - HF_B             [dB]            (share of power above 3 kHz)

Each delta series is robust-centred (subtract its median: removes global tempo/gain/voice offsets
between speakers) and flagged where |d - median| > max(perceptual floor, K * 1.4826 * MAD).
Consecutive flagged words become one region with exact start/end timestamps.
"""
import numpy as np

from .features import extract, word_table

K_MAD = 3.0
FLOORS = {"rate": 0.30, "pause": 0.35, "pitch": 0.30, "energy": 5.0, "clarity": 4.0}

DIMENSIONS = {  # rubric: dimension -> (flaw types, weight)
    "pace":           (("rushed", "dragging"), 0.20),
    "pausing":        (("long_pause", "missing_pause"), 0.20),
    "expressiveness": (("monotone", "erratic_pitch"), 0.25),
    "volume":         (("volume_drop", "volume_spike"), 0.15),
    "clarity":        (("muffled",), 0.20),
}
SCORE_GAIN = 15.0


def _runs(flag, close=1, min_len=2):
    flag = flag.copy()
    n = len(flag)
    i = 0
    while i < n:  # close gaps of <= `close` unflagged words between flagged ones
        if not flag[i]:
            j = i
            while j < n and not flag[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= close:
                flag[i:j] = True
            i = j
        else:
            i += 1
    out, i = [], 0
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _threshold(d, floor):
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return floor
    return max(floor, K_MAD * 1.4826 * np.median(np.abs(d - np.median(d))))


def _fmt(txt, words, i, j):
    return " ".join(w["word"] for w in words[i:j + 1])


def detect_regions(wt_p, wt_b, words_p, words_b=None):
    n = len(words_p)
    regions = []

    def add(kind, i, j, d, thr, obs, ref, unit, metric, start=None, end=None, base_start=None, base_end=None):
        mag = float(np.nanmean(np.abs(d[i:j + 1])) / thr)
        r_start = float(words_p[i]["start"] if start is None else start)
        r_end = float(words_p[j]["end"] if end is None else end)
        if words_b is not None:
            b_start = float(words_b[i]["start"] if base_start is None else base_start)
            b_end = float(words_b[j]["end"] if base_end is None else base_end)
        else:
            b_start, b_end = r_start, r_end
        regions.append(dict(
            type=kind, word_start=i, word_end=j,
            start=r_start, end=r_end,
            base_start=b_start, base_end=b_end,
            text=_fmt(None, words_p, i, j), metric=metric, unit=unit,
            observed=float(obs), reference=float(ref), delta=float(np.nanmean(d[i:j + 1])),
            threshold=float(thr), magnitude=mag, severity=float(np.clip((mag - 1) / 3, 0, 1))))

    # --- pace
    d = np.log(np.maximum(wt_p["dur"], 1e-3) / np.maximum(wt_b["dur"], 1e-3))
    dc = d - np.median(d)
    thr = _threshold(d, FLOORS["rate"])
    for kind, flag in (("rushed", dc < -thr), ("dragging", dc > thr)):
        for i, j in _runs(flag):
            add(kind, i, j, dc, thr, np.median(wt_p["dur"][i:j + 1]), np.median(wt_b["dur"][i:j + 1]),
                "s/word", "median word duration")
    # --- pauses
    d = wt_p["gap"] - wt_b["gap"]
    thr = _threshold(d, FLOORS["pause"])
    for i in np.where(d - np.median(d) > thr)[0]:
        add("long_pause", int(i), int(i), d - np.median(d), thr, wt_p["gap"][i], wt_b["gap"][i], "s",
            "silence before word", start=words_p[i - 1]["end"] if i > 0 else words_p[i]["start"],
            end=words_p[i]["start"],
            base_start=words_b[i - 1]["end"] if (words_b is not None and i > 0) else (words_b[i]["start"] if words_b is not None else None),
            base_end=words_b[i]["start"] if words_b is not None else None)
    for i in np.where((wt_b["gap"] >= 0.35) & (wt_p["gap"] < 0.1))[0]:
        add("missing_pause", int(i), int(i), wt_p["gap"] - wt_b["gap"], 0.25, wt_p["gap"][i], wt_b["gap"][i],
            "s", "silence before word", start=words_p[i - 1]["end"] if i > 0 else words_p[i]["start"],
            end=words_p[i]["start"] + 0.05,
            base_start=words_b[i - 1]["end"] if (words_b is not None and i > 0) else (words_b[i]["start"] if words_b is not None else None),
            base_end=words_b[i]["start"] if words_b is not None else None)
    # --- pitch expressiveness
    d = np.log((wt_p["pitch_range"] + 0.5) / (wt_b["pitch_range"] + 0.5))
    dcc = d - np.nanmedian(d)
    thr = _threshold(d, FLOORS["pitch"])
    dcc = np.nan_to_num(dcc)
    for kind, flag in (("monotone", dcc < -thr), ("erratic_pitch", dcc > thr)):
        for i, j in _runs(flag, min_len=3):
            add(kind, i, j, dcc, thr, np.nanmean(wt_p["pitch_range"][i:j + 1]),
                np.nanmean(wt_b["pitch_range"][i:j + 1]), "semitones", "pitch range (p90-p10)")
    # --- volume
    d = wt_p["energy"] - wt_b["energy"]
    dc = d - np.median(d)
    thr = _threshold(d, FLOORS["energy"])
    for kind, flag in (("volume_drop", dc < -thr), ("volume_spike", dc > thr)):
        for i, j in _runs(flag, min_len=3):
            add(kind, i, j, dc, thr, np.mean(wt_p["energy"][i:j + 1]), np.mean(wt_b["energy"][i:j + 1]),
                "dB", "mean level re speaker median")
    # --- clarity
    d = wt_p["hf"] - wt_b["hf"]
    dc = d - np.median(d)
    thr = _threshold(d, FLOORS["clarity"])
    for i, j in _runs(dc < -thr):
        add("muffled", i, j, dc, thr, np.mean(wt_p["hf"][i:j + 1]), np.mean(wt_b["hf"][i:j + 1]), "dB",
            "high-band (>3 kHz) power share")
    regions.sort(key=lambda r: r["start"])
    return regions


def explain(r):
    """Turn the numeric delta into a structured, human-readable causal explanation."""
    d, thr, k = r["delta"], r["threshold"], r["type"]
    o, ref, u = r["observed"], r["reference"], r["unit"]
    zs = f"{abs(d) / thr:.1f}x the tolerance" if thr else ""
    if k in ("rushed", "dragging"):
        f = float(np.exp(abs(d)))
        txt = (f"Words were spoken {f:.2f}x {'faster' if k == 'rushed' else 'slower'} than the reference "
               f"(median word duration {o * 1000:.0f} ms vs {ref * 1000:.0f} ms; delta ln-ratio {d:+.2f}, {zs}). "
               + ("Compressed articulation leaves no room for emphasis and hurts intelligibility."
                  if k == "rushed" else "Drawn-out words break the rhythm and lose audience energy."))
    elif k == "long_pause":
        txt = (f"Silence before this word lasts {o:.2f} s against {ref:.2f} s in the reference "
               f"(+{o - ref:.2f} s, {zs}). Unplanned mid-phrase silence reads as hesitation.")
    elif k == "missing_pause":
        txt = (f"The reference pauses {ref:.2f} s here but the participant continues after {o:.2f} s. "
               "Skipping phrase-boundary pauses removes the listener's processing time.")
    elif k in ("monotone", "erratic_pitch"):
        txt = (f"Pitch range over this passage is {o:.1f} vs {ref:.1f} semitones in the reference "
               f"(ln-ratio {d:+.2f}, {zs}). "
               + ("A flattened F0 contour sounds monotone and drops emphasis cues."
                  if k == "monotone" else "Excessive pitch swings sound unstable and distract from meaning."))
    elif k in ("volume_drop", "volume_spike"):
        txt = (f"Level is {o:+.1f} dB vs {ref:+.1f} dB (re each speaker's median) "
               f"-> {abs(d):.1f} dB {'quieter' if k == 'volume_drop' else 'louder'} than the reference "
               f"({zs}). " + ("Quiet passages risk being lost to the audience."
                              if k == "volume_drop" else "Sudden loudness reads as shouting."))
    else:  # muffled
        txt = (f"Share of energy above 3 kHz is {o:.1f} dB vs {ref:.1f} dB in the reference "
               f"({abs(d):.1f} dB lower, {zs}). Loss of high-frequency detail (consonants, sibilants) "
               "indicates mumbled or covered articulation.")
    return txt


def score(regions, ref_duration):
    ref_duration = max(ref_duration, 0.1)
    out = {}
    for dim, (types, _) in DIMENSIONS.items():
        impact = 0.0
        for r in regions:
            if r["type"] not in types:
                continue
            part = r["end"] - r["start"]
            base = r.get("base_end", r["end"]) - r.get("base_start", r["start"])
            impact += (max(part, base) / ref_duration) * (0.3 + 0.7 * r["severity"])
        out[dim] = round(float(100 * np.exp(-SCORE_GAIN * impact)), 1)
    wmean = sum(out[d] * w for d, (_, w) in DIMENSIONS.items())
    # judges penalise the weakest aspect: overall = 0.5 * weighted mean + 0.5 * worst dimension
    out["overall"] = round(0.5 * wmean + 0.5 * min(out[d] for d in DIMENSIONS), 1)
    return out


def _warp(t_p, words_p, words_b):
    """Map participant times onto baseline times via word anchors (for overlay plots)."""
    xp, xb = [0.0], [0.0]
    for wp, wb in zip(words_p, words_b):
        xp += [wp["start"], wp["end"]]
        xb += [wb["start"], wb["end"]]
    xp, xb = np.array(xp), np.array(xb)
    order = np.argsort(xp, kind="stable")
    xp, xb = xp[order], xb[order]
    return np.interp(t_p, xp, xb, left=None) + np.where(t_p > xp[-1], t_p - xp[-1], 0.0)


def _clean(a, nd=2, step=2):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a[::step]]


def analyze(y_p, words_p, y_b, words_b):
    if len(words_p) != len(words_b):
        raise ValueError(f"Alignments differ in length ({len(words_p)} vs {len(words_b)}): "
                         "participant and baseline must use the same transcript")
    f_p, f_b = extract(y_p), extract(y_b)
    wt_p, wt_b = word_table(f_p, words_p), word_table(f_b, words_b)
    regions = detect_regions(wt_p, wt_b, words_p, words_b)
    for r in regions:
        r["explanation"] = explain(r)
    T = max(words_p[-1]["end"] - words_p[0]["start"], 0.1)
    ref_duration = max(words_b[-1]["end"] - words_b[0]["start"], 0.1)
    scores = score(regions, ref_duration)

    tb = _warp(f_p.t, words_p, words_b)
    ib = np.clip(np.searchsorted(f_b.t, tb), 0, len(f_b.t) - 1)
    series = {
        "t": _clean(f_p.t),
        "participant": {"f0_st": _clean(f_p.f0_st), "energy_db": _clean(f_p.energy_db), "hf_db": _clean(f_p.hf_db)},
        "baseline": {"f0_st": _clean(f_b.f0_st[ib]), "energy_db": _clean(f_b.energy_db[ib]),
                     "hf_db": _clean(f_b.hf_db[ib])},
    }
    glob = {
        "speech_duration_s": round(float(T), 2),
        "articulation_wps": {"participant": round(len(words_p) / max(wt_p["dur"].sum(), 1e-6), 2),
                             "baseline": round(len(words_b) / max(wt_b["dur"].sum(), 1e-6), 2)},
        "pause_time_s": {"participant": round(float(wt_p["gap"].sum()), 2), "baseline": round(float(wt_b["gap"].sum()), 2)},
        "f0_median_hz": {"participant": round(f_p.f0_median_hz, 1), "baseline": round(f_b.f0_median_hz, 1)},
    }
    return {"scores": scores, "regions": regions, "global": glob, "series": series,
            "words": [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in words_p]}
