"""Evaluate SpeechLens flaw detection on the real-voice dataset.

Reads ground-truth labels produced by import_audacity_labels.py, evaluates each
flawed and clean2 take against that speaker's clean baseline, and reports:
  - Per-flaw-type precision, recall, and F1 at IoU >= 0.5 and IoU >= 0.3
  - Per-speaker breakdown tables
  - Detailed listing of every false positive detected on clean2 takes
  - Boundary Mean Absolute Error (MAE) and Mean IoU
Writes eval/real_results.json and prints human-readable formatted tables.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speechlens.analyze import analyze, K_MAD, FLOORS
from speechlens.evaluate import boundary_errors, match
from speechlens.io import load_audio
from speechlens.text import load_alignment


def get_speaker_id(stem: str) -> str:
    """Extract speaker ID from take or baseline stem."""
    return stem.split("__")[0] if "__" in stem else stem.split("_")[0]


def is_clean2_take(lab: dict) -> bool:
    """Check if label dict represents a clean2 take."""
    return lab.get("kind") == "clean2" or "__clean2" in lab.get("id", "") or "_clean2" in lab.get("id", "")


def find_audio_file(base_dir: Path, stem: str) -> Optional[Path]:
    """Find audio file with matching stem under base_dir (searches common audio subdirs)."""
    search_dirs = [
        base_dir,
        base_dir / "flawed",
        base_dir / "baseline",
        ROOT / "data" / "real",
        ROOT / "data" / "real" / "baseline",
        ROOT / "data" / "baseline",
    ]
    exts = [".wav", ".flac", ".mp3"]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in exts:
            cand = d / f"{stem}{ext}"
            if cand.exists():
                return cand
        # Also check glob inside d
        matches = list(d.glob(f"{stem}.*"))
        for m in matches:
            if m.suffix.lower() in exts:
                return m
    return None


def find_alignment_file(base_dir: Path, stem: str) -> Optional[Path]:
    """Find alignment JSON file for stem."""
    search_dirs = [
        base_dir / "alignments",
        base_dir,
        ROOT / "data" / "real" / "alignments",
        ROOT / "data" / "alignments",
    ]
    for d in search_dirs:
        cand = d / f"{stem}.json"
        if cand.exists():
            return cand
    return None


def run_real_eval(
    data_dir: Path,
    labels_dir: Path,
    out_path: Path,
    iou_thrs: tuple[float, float] = (0.5, 0.3),
):
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print(f"No label files found in {labels_dir}. Run scripts/import_audacity_labels.py first.")
        return

    # Fail loudly if constants are missing/invalid
    if K_MAD is None or FLOORS is None:
        raise ImportError("Could not import K_MAD and FLOORS from speechlens.analyze")

    base_cache = {}
    per_file = []

    print(f"Loading {len(label_files)} label files from {labels_dir}...\n")

    for lab_path in label_files:
        lab = json.load(open(lab_path, encoding="utf-8"))
        take_id = lab["id"]
        b_id = lab["baseline"]

        # Cache baseline audio and words
        if b_id not in base_cache:
            base_audio_file = find_audio_file(data_dir, b_id)
            if not base_audio_file:
                # Try finding without __baseline suffix or with speaker ID
                spk = get_speaker_id(b_id)
                base_audio_file = find_audio_file(data_dir, f"{spk}__baseline") or find_audio_file(data_dir, spk)
            if not base_audio_file:
                raise FileNotFoundError(f"Could not locate baseline audio for '{b_id}' (speaker: {get_speaker_id(b_id)}).")

            base_align_file = find_alignment_file(data_dir, b_id)
            if not base_align_file:
                spk = get_speaker_id(b_id)
                base_align_file = find_alignment_file(data_dir, f"{spk}__baseline") or find_alignment_file(data_dir, spk)
            if not base_align_file:
                raise FileNotFoundError(f"Could not locate baseline alignment for '{b_id}'.")

            base_cache[b_id] = (load_audio(base_audio_file), load_alignment(base_align_file))

        yb, wb = base_cache[b_id]

        # Participant audio & alignment
        part_audio_file = find_audio_file(data_dir, take_id)
        if not part_audio_file:
            print(f"Warning: Audio file not found for take '{take_id}'. Skipping.")
            continue

        wp = lab.get("words")
        if not wp:
            part_align_file = find_alignment_file(data_dir, take_id)
            if part_align_file:
                wp = load_alignment(part_align_file)
            else:
                raise ValueError(f"No word alignments available for take '{take_id}'.")

        yp = load_audio(part_audio_file)

        # Run SpeechLens detection
        res = analyze(yp, wp, yb, wb)
        per_file.append((lab, res, wp, wb))

        pred_types = [r["type"] for r in res["regions"]]
        truth_types = [r["type"] for r in lab.get("regions", [])]
        print(f"{take_id:38s} score={res['scores']['overall']:5.1f} | truth={truth_types} | pred={pred_types}")

    if not per_file:
        print("No evaluations could be run.")
        return

    out = {
        "dataset": "real-voice",
        "total_files": len(per_file),
        "clean2_false_positives": [],
        "iou_results": {},
        "per_speaker": {},
    }

    # Evaluate at each IoU threshold
    for thr in iou_thrs:
        t_tp = defaultdict(int)
        t_fn = defaultdict(int)
        t_fp = defaultdict(int)
        spk_tp = defaultdict(int)
        spk_fn = defaultdict(int)
        spk_fp = defaultdict(int)
        spk_takes = defaultdict(int)

        all_ious = []
        all_ious_snapped = []
        all_errs = []

        for lab, res, wp, wb in per_file:
            spk = get_speaker_id(lab["id"])
            if thr == iou_thrs[0]:
                spk_takes[spk] += 1

            # Use raw Audacity seconds as the primary number for IoU evaluation.
            gt_regions_raw = []
            for r in lab.get("regions", []):
                rc = dict(r)
                if "raw_start" in r:
                    rc["start"] = r["raw_start"]
                    rc["end"] = r["raw_end"]
                gt_regions_raw.append(rc)

            pairs, up, ut = match(res["regions"], gt_regions_raw, iou_thr=thr)

            for pi, ti, v in pairs:
                flaw_type = lab["regions"][ti]["type"]
                t_tp[flaw_type] += 1
                spk_tp[spk] += 1
                all_ious.append(v)

            all_errs += boundary_errors(res["regions"], gt_regions_raw, pairs)

            # Secondary column: snapped IoU
            pairs_snapped, _, _ = match(res["regions"], lab["regions"], iou_thr=thr)
            for _, _, v_s in pairs_snapped:
                if thr == iou_thrs[0]:
                    all_ious_snapped.append(v_s)

            for ti in ut:
                flaw_type = lab["regions"][ti]["type"]
                t_fn[flaw_type] += 1
                spk_fn[spk] += 1

            for pi in up:
                flaw_type = res["regions"][pi]["type"]
                t_fp[flaw_type] += 1
                spk_fp[spk] += 1

        tot_tp = sum(t_tp.values())
        tot_fn = sum(t_fn.values())
        tot_fp = sum(t_fp.values())
        micro_r = tot_tp / max(tot_tp + tot_fn, 1)
        micro_p = tot_tp / max(tot_tp + tot_fp, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

        print(f"\n{'='*75}")
        print(f"Per Flaw Type Evaluation (IoU >= {thr:.1f}, flaw type must match)")
        print(f"{'='*75}")
        print(f"Overall Micro: Recall={tot_tp}/{tot_tp+tot_fn} ({micro_r:.2f}), "
              f"Precision={tot_tp}/{tot_tp+tot_fp} ({micro_p:.2f}), F1={micro_f1:.2f}")
        print(f"{'Flaw Type':16s} {'Recall':>8s} {'Precision':>10s} {'F1':>8s} {'TP':>5s} {'FN':>5s} {'FP':>5s}")
        print(f"{'-'*16} {'-'*8} {'-'*10} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")

        thr_key = f"iou_{thr:.1f}"
        out["iou_results"][thr_key] = {
            "micro": {"recall": micro_r, "precision": micro_p, "f1": micro_f1,
                      "tp": tot_tp, "fn": tot_fn, "fp": tot_fp},
            "per_flaw": {},
        }

        all_flaws = sorted(set(t_tp) | set(t_fn) | set(t_fp))
        for flaw in all_flaws:
            has_gt = (t_tp[flaw] + t_fn[flaw]) > 0
            rec = t_tp[flaw] / max(t_tp[flaw] + t_fn[flaw], 1) if has_gt else None
            prec = t_tp[flaw] / max(t_tp[flaw] + t_fp[flaw], 1)
            f1 = (2 * prec * rec / max(prec + rec, 1e-9)) if rec is not None else 0.0

            out["iou_results"][thr_key]["per_flaw"][flaw] = {
                "recall": rec, "precision": prec, "f1": f1,
                "tp": t_tp[flaw], "fn": t_fn[flaw], "fp": t_fp[flaw]
            }

            rec_str = f"{rec:8.2f}" if rec is not None else "     n/a"
            print(f"{flaw:16s} {rec_str} {prec:10.2f} {f1:8.2f} {t_tp[flaw]:5d} {t_fn[flaw]:5d} {t_fp[flaw]:5d}")

        # Per-speaker table
        print(f"\n{'-'*75}")
        print(f"Per Speaker Evaluation (IoU >= {thr:.1f})")
        print(f"{'-'*75}")
        print(f"{'Speaker':12s} {'Takes':>6s} {'Recall':>8s} {'Precision':>10s} {'F1':>8s} {'TP':>5s} {'FN':>5s} {'FP':>5s}")
        print(f"{'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")

        out["per_speaker"][thr_key] = {}
        for spk in sorted(set(spk_takes) | set(spk_tp) | set(spk_fp)):
            s_tp_val = spk_tp[spk]
            s_fn_val = spk_fn[spk]
            s_fp_val = spk_fp[spk]
            has_gt = (s_tp_val + s_fn_val) > 0
            rec = s_tp_val / max(s_tp_val + s_fn_val, 1) if has_gt else None
            prec = s_tp_val / max(s_tp_val + s_fp_val, 1)
            f1 = (2 * prec * rec / max(prec + rec, 1e-9)) if rec is not None else 0.0

            out["per_speaker"][thr_key][spk] = {
                "takes": spk_takes[spk], "recall": rec, "precision": prec, "f1": f1,
                "tp": s_tp_val, "fn": s_fn_val, "fp": s_fp_val
            }
            rec_str = f"{rec:8.2f}" if rec is not None else "     n/a"
            print(f"{spk:12s} {spk_takes[spk]:6d} {rec_str} {prec:10.2f} {f1:8.2f} {s_tp_val:5d} {s_fn_val:5d} {s_fp_val:5d}")

        if thr == 0.5:
            e = np.array(all_errs) if all_errs else np.zeros((0, 2))
            out["boundary_mae_s"] = {
                "start": float(e[:, 0].mean()) if len(e) else None,
                "end": float(e[:, 1].mean()) if len(e) else None,
                "mean_iou_raw": float(np.mean(all_ious)) if all_ious else None,
                "mean_iou_snapped": float(np.mean(all_ious_snapped)) if all_ious_snapped else None,
            }
            print(f"\nBoundary MAE (s): start={out['boundary_mae_s']['start']}, "
                  f"end={out['boundary_mae_s']['end']}, mean IoU (raw)={out['boundary_mae_s']['mean_iou_raw']}, mean IoU (snapped)={out['boundary_mae_s']['mean_iou_snapped']}")

    # Clean2 False Positive Reporting
    print(f"\n{'='*75}")
    print("Clean2 False Positive Audit (Ground truth = no flaws)")
    print(f"{'='*75}")

    clean2_takes = [item for item in per_file if is_clean2_take(item[0])]
    clean2_detections_total = 0

    if not clean2_takes:
        print("No clean2 takes found in evaluated dataset.")
    else:
        for lab, res, wp, _ in clean2_takes:
            take_id = lab["id"]
            spk = get_speaker_id(take_id)
            n_det = len(res["regions"])
            clean2_detections_total += n_det

            if n_det == 0:
                print(f"  [CLEAN] {take_id:34s} (Speaker: {spk}): 0 detections")
            else:
                print(f"  [FLAGGED {n_det} FP] {take_id:28s} (Speaker: {spk}):")
                for r in res["regions"]:
                    record = {
                        "take": take_id,
                        "speaker": spk,
                        "flaw": r["type"],
                        "start": round(float(r["start"]), 3),
                        "end": round(float(r["end"]), 3),
                        "words": f"{r['word_start']}..{r['word_end']}",
                        "text": r.get("text", ""),
                        "metric": r.get("metric", ""),
                        "magnitude": round(float(r.get("magnitude", 0)), 2),
                    }
                    out["clean2_false_positives"].append(record)
                    print(f"      - {r['type']:14s} at {r['start']:6.2f}s - {r['end']:6.2f}s "
                          f"({r.get('text', '')!r}) mag={r.get('magnitude', 0):.2f}")

        print(f"\nTotal clean2 false positives: {clean2_detections_total} across {len(clean2_takes)} clean2 takes.")

    out["clean2_summary"] = {
        "takes_count": len(clean2_takes),
        "false_positive_regions": clean2_detections_total,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nResults saved to {out_path}.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate flaw detection on real-voice recordings.")
    parser.add_argument("--data", default="data/real", help="Directory containing real voice audio/data")
    parser.add_argument("--labels", default=None, help="Directory containing label JSONs (defaults to <data>/labels)")
    parser.add_argument("--out", default="eval/real_results.json", help="Output path for results JSON")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        alt = Path("data")
        if alt.exists():
            data_dir = alt
        else:
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    labels_dir = Path(args.labels) if args.labels else (data_dir / "labels")
    if not labels_dir.exists():
        # Check fallback to data/labels
        if (Path("data") / "labels").exists():
            labels_dir = Path("data") / "labels"

    out_path = Path(args.out)
    run_real_eval(data_dir=data_dir, labels_dir=labels_dir, out_path=out_path)


if __name__ == "__main__":
    main()
