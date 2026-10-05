"""Build the contrastive dataset from aligned baselines.

For every baseline in data/baseline (with data/alignments/<id>.json) this writes:
  * 6 flaw types x 4 severities of single-flaw "mirrors"  (severity 1 = almost perfect, 4 = botched)
  * mixed multi-flaw files
  * negative controls (global gain change, faint added noise) that must NOT be flagged
plus a label JSON per file (flaw type, severity, exact start/end, word indices, re-timed alignment)
and data/manifest.csv. Everything is seeded and reproducible.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechlens.inject import FLAWS, choose_boundary, choose_span, make_flawed, seed_for  # noqa: E402
from speechlens.io import load_audio, save_audio  # noqa: E402
from speechlens.text import load_alignment  # noqa: E402


def single_plan(words, flaw, sev, base_id):
    rng = np.random.default_rng(seed_for(base_id, flaw, "place"))
    if flaw == "long_pause":
        i = choose_boundary(words, rng)
        return [dict(flaw=flaw, severity=sev, i=i, j=i, seed=seed_for(base_id, flaw, sev))]
    i, j = choose_span(words, rng)
    return [dict(flaw=flaw, severity=sev, i=i, j=j, seed=seed_for(base_id, flaw, sev))]


def mixed_plan(words, base_id, k):
    rng = np.random.default_rng(seed_for(base_id, "mixed", k))
    n = len(words)
    kinds = list(rng.choice(FLAWS, size=3, replace=False))
    slots = np.linspace(4, n - 14, 3).astype(int)  # three disjoint zones of the speech
    plan = []
    for kind, s0 in zip(kinds, slots):
        sev = int(rng.integers(2, 5))
        if kind == "long_pause":
            i = int(s0) + int(rng.integers(0, 4))
            plan.append(dict(flaw=kind, severity=sev, i=i, j=i, seed=seed_for(base_id, "mixed", k, kind)))
        else:
            L = int(rng.integers(6, 9))
            plan.append(dict(flaw=kind, severity=sev, i=int(s0), j=int(s0) + L - 1,
                             seed=seed_for(base_id, "mixed", k, kind)))
    return plan


def write(out, name, y, words, regions, base_id, kind, ext):
    save_audio(out / "flawed" / f"{name}.{ext}", y)
    with open(out / "labels" / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"id": name, "baseline": base_id, "kind": kind,
                   "regions": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                               for r in regions],
                   "words": [{"word": w["word"], "start": round(w["start"], 4), "end": round(w["end"], 4)}
                             for w in words]}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mixed", type=int, default=4)
    ap.add_argument("--ext", default="flac", choices=["flac", "wav"])
    a = ap.parse_args()
    data = Path(a.data)
    (data / "flawed").mkdir(exist_ok=True, parents=True)
    (data / "labels").mkdir(exist_ok=True, parents=True)
    rows = []
    for ap_json in sorted((data / "alignments").glob("*.json")):
        base_id = ap_json.stem
        wavs = list((data / "baseline").glob(base_id + ".*"))
        if not wavs:
            continue
        y0 = load_audio(wavs[0])
        words = load_alignment(ap_json)
        for flaw in FLAWS:
            for sev in (1, 2, 3, 4):
                try:
                    plan = single_plan(words, flaw, sev, base_id)
                except RuntimeError as err:
                    print(f"Skipping {base_id}__{flaw}_s{sev}: {err}")
                    continue
                y, w, regs = make_flawed(y0, words, plan)
                name = f"{base_id}__{flaw}_s{sev}"
                write(data, name, y, w, regs, base_id, "single", a.ext)
                rows.append([name, base_id, "single", flaw, sev])
        for k in range(a.mixed):
            y, w, regs = make_flawed(y0, words, mixed_plan(words, base_id, k))
            name = f"{base_id}__mixed_{k}"
            write(data, name, y, w, regs, base_id, "mixed", a.ext)
            rows.append([name, base_id, "mixed", "+".join(r["type"] for r in regs), ""])
        rng = np.random.default_rng(seed_for(base_id, "control"))
        write(data, f"{base_id}__control_gain", y0 * 0.5, words, [], base_id, "control", a.ext)
        write(data, f"{base_id}__control_noise", y0 + rng.normal(0, 0.003, len(y0)).astype(np.float32),
              words, [], base_id, "control", a.ext)
        rows += [[f"{base_id}__control_gain", base_id, "control", "none", ""],
                 [f"{base_id}__control_noise", base_id, "control", "none", ""]]
    with open(data / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file", "baseline", "kind", "flaws", "severity"]] + rows)
    print(f"wrote {len(rows)} files")


if __name__ == "__main__":
    main()
