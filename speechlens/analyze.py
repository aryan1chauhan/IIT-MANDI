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
EDGE_RATIO = 0.65
PITCH_EDGE_RATIO = 0.75

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


def _expand_runs(runs, dcc, thr, direction, edge_ratio):
    """Expand confirmed runs outward across contiguous same-sign words >= edge_ratio * thr, merging overlaps."""
    if not runs:
        return []
    n = len(dcc)
    low = edge_ratio * thr
    expanded = []
    for i, j in runs:
        while i > 0 and np.isfinite(dcc[i - 1]) and direction * dcc[i - 1] >= low:
            i -= 1
        while j + 1 < n and np.isfinite(dcc[j + 1]) and direction * dcc[j + 1] >= low:
            j += 1
        expanded.append((i, j))
    expanded.sort()
    merged = [expanded[0]]
    for i, j in expanded[1:]:
        if i <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], j))
        else:
            merged.append((i, j))
    return merged


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

    # --- pace (per-word + 3-word rolling articulation window)
    dur_p = wt_p["dur"]
    dur_b = wt_b["dur"]
    d = np.log(np.maximum(dur_p, 1e-3) / np.maximum(dur_b, 1e-3))
    dc = d - np.median(d)
    thr = _threshold(d, FLOORS["rate"])

    dw = np.full(n, np.nan)
    for k in range(1, n - 1):
        if np.all(dur_b[k - 1:k + 2] > 0) and np.all(dur_p[k - 1:k + 2] > 0):
            dw[k] = np.log(np.sum(dur_p[k - 1:k + 2]) / np.sum(dur_b[k - 1:k + 2]))
    med_w = float(np.nanmedian(dw)) if np.isfinite(dw).any() else 0.0
    dcc_win = np.nan_to_num(dw - med_w)
    thr_win = _threshold(dw, FLOORS["rate"])

    norm_word = dc / thr
    norm_win = dcc_win / thr_win
    norm_rate = np.where(np.abs(norm_win) > np.abs(norm_word), norm_win, norm_word)

    dragging_spans = []
    for kind, flag_word, flag_win, direction in (
        ("rushed", dc < -thr, dcc_win < -thr_win, -1),
        ("dragging", dc > thr, dcc_win > thr_win, 1),
    ):
        seeds = _runs(flag_word) + _runs(flag_win)
        if not seeds:
            continue
        seeds.sort()
        merged_seeds = [seeds[0]]
        for s, e in seeds[1:]:
            if s <= merged_seeds[-1][1] + 1:
                merged_seeds[-1] = (merged_seeds[-1][0], max(merged_seeds[-1][1], e))
            else:
                merged_seeds.append((s, e))
        for i, j in _expand_runs(merged_seeds, norm_rate, 1.0, direction, EDGE_RATIO):
            add(kind, i, j, dc, thr, np.median(dur_p[i:j + 1]), np.median(dur_b[i:j + 1]),
                "s/word", "median word duration")
            if kind == "dragging":
                local_scale = max(1.0, float(np.nanmedian(dur_p[i:j + 1] / np.maximum(dur_b[i:j + 1], 1e-6))))
                dragging_spans.append((i, j, local_scale))
    # --- pauses
    d = wt_p["gap"] - wt_b["gap"]
    med_pause = np.median(d)
    thr = _threshold(d, FLOORS["pause"])
    for i in np.where(d - med_pause > thr)[0]:
        idx = int(i)
        scales = [loc_s for s, e, loc_s in dragging_spans if s <= idx <= e]
        if scales and (wt_p["gap"][idx] - wt_b["gap"][idx] * max(scales) <= thr):
            continue
        add("long_pause", idx, idx, d - med_pause, thr, wt_p["gap"][idx], wt_b["gap"][idx], "s",
            "silence before word", start=words_p[idx - 1]["end"] if idx > 0 else words_p[idx]["start"],
            end=words_p[idx]["start"],
            base_start=words_b[idx - 1]["end"] if (words_b is not None and idx > 0) else (words_b[idx]["start"] if words_b is not None else None),
            base_end=words_b[idx]["start"] if words_b is not None else None)
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
    for kind, flag, direction in (("monotone", dcc < -thr, -1), ("erratic_pitch", dcc > thr, 1)):
        for i, j in _expand_runs(_runs(flag, min_len=3), dcc, thr, direction, PITCH_EDGE_RATIO):
            add(kind, i, j, dcc, thr, np.nanmean(wt_p["pitch_range"][i:j + 1]),
                np.nanmean(wt_b["pitch_range"][i:j + 1]), "semitones", "pitch range (p90-p10)")
    # --- volume
    d = wt_p["energy"] - wt_b["energy"]
    dc = d - np.median(d)
    thr = _threshold(d, FLOORS["energy"])
    for kind, flag, direction in (("volume_drop", dc < -thr, -1), ("volume_spike", dc > thr, 1)):
        for i, j in _expand_runs(_runs(flag, min_len=3), dc, thr, direction, EDGE_RATIO):
            add(kind, i, j, dc, thr, np.mean(wt_p["energy"][i:j + 1]), np.mean(wt_b["energy"][i:j + 1]),
                "dB", "mean level re speaker median")
    # --- clarity
    d = wt_p["hf"] - wt_b["hf"]
    dc = d - np.median(d)
    thr = _threshold(d, FLOORS["clarity"])
    for i, j in _expand_runs(_runs(dc < -thr), dc, thr, -1, EDGE_RATIO):
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
        "f0_bounds_hz": {"participant": [round(f_p.f0_floor_hz, 1), round(f_p.f0_ceiling_hz, 1)],
                         "baseline": [round(f_b.f0_floor_hz, 1), round(f_b.f0_ceiling_hz, 1)]},
    }
    return {"scores": scores, "regions": regions, "global": glob, "series": series,
            "words": [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in words_p]}
