"""Score detected flaw regions against injected ground-truth labels."""
import numpy as np


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0:  # degenerate intervals: fall back to proximity
        return 1.0 if abs(a[0] - b[0]) < 0.3 else 0.0
    return inter / union


def match(pred, truth, iou_thr=0.3, require_type=True):
    """Greedy one-to-one matching by IoU. Returns (pairs, unmatched_pred_idx, unmatched_truth_idx)."""
    cands = []
    for pi, p in enumerate(pred):
        for ti, t in enumerate(truth):
            if require_type and p["type"] != t["type"]:
                continue
            v = iou((p["start"], p["end"]), (t["start"], t["end"]))
            if v >= iou_thr:
                cands.append((v, pi, ti))
    cands.sort(reverse=True)
    used_p, used_t, pairs = set(), set(), []
    for v, pi, ti in cands:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        pairs.append((pi, ti, v))
    return (pairs, [i for i in range(len(pred)) if i not in used_p],
            [i for i in range(len(truth)) if i not in used_t])


def boundary_errors(pred, truth, pairs):
    return [(abs(pred[pi]["start"] - truth[ti]["start"]), abs(pred[pi]["end"] - truth[ti]["end"]))
            for pi, ti, _ in pairs]
