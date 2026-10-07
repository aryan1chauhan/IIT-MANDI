"""Evaluate flaw detection against injected ground truth. Writes eval/results.json and prints tables."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechlens.align import align_words  # noqa: E402
from speechlens.analyze import analyze  # noqa: E402
from speechlens.evaluate import boundary_errors, match  # noqa: E402
from speechlens.io import load_audio  # noqa: E402
from speechlens.text import load_alignment, save_alignment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--realign", action="store_true", help="Realign participant audio with MMS_FA")
    ap.add_argument("--realign-baseline", action="store_true",
                    help="Realign baseline audio with MMS_FA (matched provenance)")
    a = ap.parse_args()
    data = Path(a.data)
    base_cache, per_file = {}, []
    for lab_path in sorted((data / "labels").glob("*.json")):
        lab = json.load(open(lab_path))
        b = lab["baseline"]
        if b not in base_cache:
            f = [p for p in (data / "baseline").glob(b + ".*") if p.suffix.lower() in (".wav", ".mp3", ".flac")][0]
            if a.realign_baseline:
                cache_b = data / "alignments" / f"{b}_mmsfa.json"
                if cache_b.exists():
                    wb = load_alignment(cache_b)
                else:
                    transcript = (data / "transcripts" / f"{b}.txt").read_text(encoding="utf-8")
                    wb = align_words(str(f), transcript)
                    save_alignment(cache_b, wb)
            else:
                wb = load_alignment(data / "alignments" / f"{b}.json")
            base_cache[b] = (load_audio(f), wb)
        yb, wb = base_cache[b]
        fl = [p for p in (data / "flawed").glob(lab["id"] + ".*") if p.suffix.lower() in (".wav", ".mp3", ".flac")][0]
        if a.realign:
            cache_p = data / "flawed" / f"{lab['id']}.align.json"
            if cache_p.exists():
                wp = load_alignment(cache_p)
            else:
                transcript = (data / "transcripts" / f"{b}.txt").read_text(encoding="utf-8")
                wp = align_words(str(fl), transcript)
                save_alignment(cache_p, wp)
        else:
            wp = lab["words"]
        res = analyze(load_audio(fl), wp, yb, wb)
        per_file.append((lab, res, wp, wb))
        print(f"{lab['id']:42s} overall={res['scores']['overall']:6.1f} "
              f"pred={[r['type'] for r in res['regions']]}", flush=True)

    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    sev_tp, sev_n = defaultdict(int), defaultdict(int)
    errs, ious = [], []
    controls_fp = 0
    n_controls = 0

    total_flagged_words = 0
    short_flagged_words = 0
    rate_flagged_words = 0
    rate_short_flagged_words = 0
    fp_flagged_words = 0
    fp_short_flagged_words = 0

    for lab, res, wp, wb in per_file:
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

        for idx, r in enumerate(res["regions"]):
            w0, w1 = r["word_start"], r["word_end"]
            dur_p = [wp[k]["end"] - wp[k]["start"] for k in range(w0, w1 + 1)]
            dur_b = [wb[k]["end"] - wb[k]["start"] for k in range(w0, w1 + 1)]
            n_short = sum(dp <= 0.040 or db <= 0.040 for dp, db in zip(dur_p, dur_b))
            n_tot = len(dur_p)
            total_flagged_words += n_tot
            short_flagged_words += n_short
            if r["type"] in ("rushed", "dragging"):
                rate_flagged_words += n_tot
                rate_short_flagged_words += n_short
            if idx in up:
                fp_flagged_words += n_tot
                fp_short_flagged_words += n_short

    out = {"per_type": {}, "per_severity": {}, "iou_0.5": {}}
    for thr in (0.3, 0.5):
        t_tp, t_fn, t_fp = defaultdict(int), defaultdict(int), defaultdict(int)
        s_tp, s_n = defaultdict(int), defaultdict(int)
        for lab, res, _, _ in per_file:
            pairs, up, ut = match(res["regions"], lab["regions"], iou_thr=thr)
            for pi, ti, _ in pairs:
                t_tp[lab["regions"][ti]["type"]] += 1
                s_tp[lab["regions"][ti]["severity"]] += 1
            for ti in ut:
                t_fn[lab["regions"][ti]["type"]] += 1
            for t in lab["regions"]:
                s_n[t["severity"]] += 1
            for pi in up:
                t_fp[res["regions"][pi]["type"]] += 1

        tot_tp = sum(t_tp.values())
        tot_fn = sum(t_fn.values())
        tot_fp = sum(t_fp.values())
        micro_r = tot_tp / max(tot_tp + tot_fn, 1)
        micro_p = tot_tp / max(tot_tp + tot_fp, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

        print(f"\nPer flaw type (IoU>={thr:.1f}, type must match)")
        print(f"Overall: Recall={tot_tp}/{tot_tp+tot_fn} ({micro_r:.2f}), Precision={tot_tp}/{tot_tp+tot_fp} ({micro_p:.2f}), F1={micro_f1:.2f}")
        print(f"{'type':14s} {'recall':>7s} {'precision':>10s} {'F1':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s}")
        target_dict = out["per_type"] if thr == 0.3 else out["iou_0.5"]
        for k in sorted(set(t_tp) | set(t_fn) | set(t_fp)):
            has_ground_truth = (t_tp[k] + t_fn[k]) > 0
            r = t_tp[k] / max(t_tp[k] + t_fn[k], 1) if has_ground_truth else None
            p = t_tp[k] / max(t_tp[k] + t_fp[k], 1)
            f1 = 2 * p * r / max(p + r, 1e-9) if r is not None else 0.0
            target_dict[k] = dict(recall=r, precision=p, f1=f1, tp=t_tp[k], fn=t_fn[k], fp=t_fp[k])
            r_str = f"{r:7.2f}" if r is not None else "    n/a"
            print(f"{k:14s} {r_str} {p:10.2f} {f1:6.2f} {t_tp[k]:4d} {t_fn[k]:4d} {t_fp[k]:4d}")

        if thr == 0.3:
            print("\nRecall by severity (1 = almost perfect ... 4 = botched)")
            for s in sorted(s_n):
                out["per_severity"][s] = s_tp[s] / s_n[s]
                print(f"  severity {s}: {s_tp[s] / s_n[s]:.2f}  ({s_tp[s]}/{s_n[s]})")

    e = np.array(errs) if errs else np.zeros((0, 2))
    out["boundary_mae_s"] = {"start": float(e[:, 0].mean()) if len(e) else None,
                             "end": float(e[:, 1].mean()) if len(e) else None,
                             "mean_iou": float(np.mean(ious)) if ious else None}
    print(f"\nBoundary MAE (s): start={out['boundary_mae_s']['start']}, end={out['boundary_mae_s']['end']}, "
          f"mean IoU={out['boundary_mae_s']['mean_iou']}")
    out["controls"] = {"files": n_controls, "false_regions": controls_fp}
    print(f"Negative controls: {controls_fp} false regions across {n_controls} files")

    print(f"\nFlagged words with duration <= 40 ms on either side:")
    print(f"  All detected regions: {short_flagged_words}/{total_flagged_words} words ({100 * short_flagged_words / max(total_flagged_words, 1):.1f}%)")
    print(f"  Rate flaws only:      {rate_short_flagged_words}/{rate_flagged_words} words ({100 * rate_short_flagged_words / max(rate_flagged_words, 1):.1f}%)")
    print(f"  False positives (FP): {fp_short_flagged_words}/{fp_flagged_words} words ({100 * fp_short_flagged_words / max(fp_flagged_words, 1):.1f}%)")
    out["flagged_words_le_40ms"] = {
        "all": {"short": short_flagged_words, "total": total_flagged_words},
        "rate": {"short": rate_short_flagged_words, "total": rate_flagged_words},
        "fp": {"short": fp_short_flagged_words, "total": fp_flagged_words},
    }

    # score gradient: does overall score fall as severity rises?
    rho = {}
    for flaw in sorted({lab["regions"][0]["type"] for lab, *_ in per_file if lab["kind"] == "single"}):
        xs, ys = [], []
        for lab, res, *_ in per_file:
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
