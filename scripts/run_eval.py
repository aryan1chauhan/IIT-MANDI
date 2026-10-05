"""Evaluate flaw detection against injected ground truth. Writes eval/results.json and prints tables."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechlens.analyze import analyze  # noqa: E402
from speechlens.evaluate import boundary_errors, match  # noqa: E402
from speechlens.io import load_audio  # noqa: E402
from speechlens.text import load_alignment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="eval/results.json")
    a = ap.parse_args()
    data = Path(a.data)
    base_cache, per_file = {}, []
    for lab_path in sorted((data / "labels").glob("*.json")):
        lab = json.load(open(lab_path))
        b = lab["baseline"]
        if b not in base_cache:
            f = list((data / "baseline").glob(b + ".*"))[0]
            base_cache[b] = (load_audio(f), load_alignment(data / "alignments" / f"{b}.json"))
        yb, wb = base_cache[b]
        fl = list((data / "flawed").glob(lab["id"] + ".*"))[0]
        res = analyze(load_audio(fl), lab["words"], yb, wb)
        per_file.append((lab, res))
        print(f"{lab['id']:42s} overall={res['scores']['overall']:6.1f} "
              f"pred={[r['type'] for r in res['regions']]}", flush=True)

    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    sev_tp, sev_n = defaultdict(int), defaultdict(int)
    errs, ious = [], []
    controls_fp = 0
    n_controls = 0
    for lab, res in per_file:
        pairs, up, ut = match(res["regions"], lab["regions"])
        for pi, ti, v in pairs:
            tp[lab["regions"][ti]["type"]] += 1
            sev_tp[lab["regions"][ti]["severity"]] += 1
            ious.append(v)
        errs += boundary_errors(res["regions"], lab["regions"], pairs)
        for ti in ut:
            fn[lab["regions"][ti]["type"]] += 1
        for t in lab["regions"]:
            sev_n[t["severity"]] += 1
        for pi in up:
            fp[res["regions"][pi]["type"]] += 1
        if lab["kind"] == "control":
            n_controls += 1
            controls_fp += len(res["regions"])

    out = {"per_type": {}, "per_severity": {}}
    print("\nPer flaw type (IoU>=0.3, type must match)")
    print(f"{'type':14s} {'recall':>7s} {'precision':>10s} {'F1':>6s}")
    for k in sorted(set(tp) | set(fn) | set(fp)):
        r = tp[k] / max(tp[k] + fn[k], 1)
        p = tp[k] / max(tp[k] + fp[k], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        out["per_type"][k] = dict(recall=r, precision=p, f1=f1, tp=tp[k], fn=fn[k], fp=fp[k])
        print(f"{k:14s} {r:7.2f} {p:10.2f} {f1:6.2f}")
    print("\nRecall by severity (1 = almost perfect ... 4 = botched)")
    for s in sorted(sev_n):
        out["per_severity"][s] = sev_tp[s] / sev_n[s]
        print(f"  severity {s}: {sev_tp[s] / sev_n[s]:.2f}  ({sev_tp[s]}/{sev_n[s]})")
    e = np.array(errs) if errs else np.zeros((0, 2))
    out["boundary_mae_s"] = {"start": float(e[:, 0].mean()) if len(e) else None,
                             "end": float(e[:, 1].mean()) if len(e) else None,
                             "mean_iou": float(np.mean(ious)) if ious else None}
    print(f"\nBoundary MAE (s): start={out['boundary_mae_s']['start']}, end={out['boundary_mae_s']['end']}, "
          f"mean IoU={out['boundary_mae_s']['mean_iou']}")
    out["controls"] = {"files": n_controls, "false_regions": controls_fp}
    print(f"Negative controls: {controls_fp} false regions across {n_controls} files")

    # score gradient: does overall score fall as severity rises?
    rho = {}
    for flaw in sorted({lab["regions"][0]["type"] for lab, _ in per_file if lab["kind"] == "single"}):
        xs, ys = [], []
        for lab, res in per_file:
            if lab["kind"] == "single" and lab["regions"][0]["type"] == flaw:
                xs.append(lab["regions"][0]["severity"])
                ys.append(res["scores"]["overall"])
        rho[flaw] = float(spearmanr(xs, ys)[0]) if len(set(xs)) > 1 else None
    out["score_vs_severity_spearman"] = rho
    print("Spearman(severity, overall score):", {k: round(v, 2) for k, v in rho.items() if v is not None})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
