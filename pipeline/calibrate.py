"""Match predictions to GT, sweep the image-score threshold, pick it, and flag images."""
from __future__ import annotations

import numpy as np


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate_image(pred_boxes: list, gt_boxes: list, match_iou: float, min_area: float) -> dict:
    """Greedy (highest conf first) same-class matching at IoU >= match_iou.

    - TP: prediction matched to a required (non-ignored) GT box.
    - Predictions matching an ignored GT (crowd / < min_area) or that are themselves
      smaller than min_area are neutral (COCO-style area-range evaluation).
    - FP: any other unmatched prediction. FN: unmatched required GT box.
    """
    used = [False] * len(gt_boxes)
    tp = fp = 0
    for p in sorted(pred_boxes, key=lambda b: -b["conf"]):
        # candidates: unused same-class GT with IoU >= match_iou; prefer required GT, then IoU
        cands = [(not g["ignore"], iou_xywh(p["bbox"], g["bbox"]), j)
                 for j, g in enumerate(gt_boxes) if not used[j] and g["cls"] == p["cls"]]
        cands = [c for c in cands if c[1] >= match_iou]
        best_j = max(cands)[2] if cands else -1
        if best_j >= 0:
            used[best_j] = True
            if not gt_boxes[best_j]["ignore"]:
                tp += 1
        elif p["bbox"][2] * p["bbox"][3] >= min_area:
            fp += 1
    fn = sum(1 for j, g in enumerate(gt_boxes) if not g["ignore"] and not used[j])
    return {"tp": tp, "fp": fp, "fn": fn, "correct": fp == 0 and fn == 0}


def sweep(rows: list[dict], cfg: dict) -> tuple[list[dict], float]:
    """rows: per-image dicts with min_conf, n_det, correct. Returns (curve, count_cap)."""
    c = cfg["calibration"]
    counts = np.array([r["n_det"] for r in rows])
    count_cap = float(np.percentile(counts, c["count_percentile"]))
    s = c["sweep"]
    thresholds = np.round(np.arange(s["start"], s["stop"] + 1e-9, s["step"]), 2)
    curve = []
    for t in thresholds:
        acc = [r for r in rows if auto_accept(r, t, count_cap)]
        n_ok = sum(r["correct"] for r in acc)
        curve.append({
            "threshold": float(t),
            "n_accepted": len(acc),
            "auto_accept_rate": len(acc) / len(rows),
            "precision": n_ok / len(acc) if acc else None,
            "errors_slipping_through": len(acc) - n_ok,
        })
    return curve, count_cap


def auto_accept(r: dict, t: float, count_cap: float) -> bool:
    return r["n_det"] > 0 and r["n_det"] <= count_cap and r["min_conf"] >= t


def choose_threshold(curve: list[dict], target: float) -> dict | None:
    """Lowest threshold whose auto-accepted set reaches the precision target."""
    for pt in curve:  # ascending thresholds
        if pt["precision"] is not None and pt["precision"] >= target:
            return pt
    return None


def flag_reasons(r: dict, t: float, count_cap: float) -> list[str]:
    reasons = []
    if r["n_det"] == 0:
        reasons.append("zero_detections")
    else:
        if r["min_conf"] < t:
            reasons.append(f"low_min_conf(<{t:.2f})")
        if r["n_det"] > count_cap:
            reasons.append(f"many_detections(>{count_cap:g}=p95)")
    return reasons
