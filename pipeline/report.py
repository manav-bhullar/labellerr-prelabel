"""Write per_image.csv, threshold_curve.png and summary.json; compute the time estimate."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def time_estimate(n_accepted: int, n_flagged: int, n_zero_det: int, tm: dict, manual_s: float) -> dict:
    """Rough model. Zero-detection images have nothing to fix, so they cost a full manual label."""
    baseline = (n_accepted + n_flagged) * manual_s
    pipeline = (n_accepted * tm["verify_s"] + (n_flagged - n_zero_det) * tm["fix_s"]
                + n_zero_det * manual_s)
    return {"manual_label_s": manual_s,
            "baseline_min": round(baseline / 60, 1),
            "pipeline_min": round(pipeline / 60, 1),
            "saved_min": round((baseline - pipeline) / 60, 1),
            "saved_pct": round(100 * (baseline - pipeline) / baseline, 1)}


def plot_curve(curve: list[dict], chosen: dict | None, target: float, out: Path) -> None:
    df = pd.DataFrame(curve).dropna(subset=["precision"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(df["auto_accept_rate"] * 100, df["precision"] * 100, color="#3b6ea5", lw=2,
            marker="o", ms=2.5, label="threshold sweep 0.05 to 0.95")
    for t in (0.3, 0.5, 0.7, 0.9):
        row = df[abs(df["threshold"] - t) < 1e-9]
        if len(row):
            x, y = row["auto_accept_rate"].iloc[0] * 100, row["precision"].iloc[0] * 100
            ax.annotate(f"t={t:.1f}", (x, y), textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color="#555")
    ax.axhline(target * 100, color="#999", ls="--", lw=1, label=f"target precision {target:.0%}")
    if chosen:
        ax.scatter([chosen["auto_accept_rate"] * 100], [chosen["precision"] * 100], s=80,
                   color="#d1495b", zorder=5,
                   label=f"chosen t={chosen['threshold']:.2f}: accept {chosen['auto_accept_rate']:.0%},"
                         f" precision {chosen['precision']:.0%}")
    ax.set_xlabel("Auto-accepted images (%)")
    ax.set_ylabel("Precision of auto-accepted set (%)")
    ax.set_title("Image-level precision vs auto-accept rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_outputs(rows: list[dict], curve: list[dict], chosen: dict | None, count_cap: float,
                  cfg: dict, extra: dict) -> dict:
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    plot_curve(curve, chosen, cfg["calibration"]["target_precision"], out / "threshold_curve.png")

    n = len(rows)
    acc = [r for r in rows if not r["flagged"]]
    flg = [r for r in rows if r["flagged"]]
    n_zero = sum(r["n_det"] == 0 for r in rows)
    tm = cfg["time_model"]
    reasons: dict[str, int] = {}
    for r in flg:
        for k in r["flag_reason"].split(";"):
            key = k.split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
    summary = {
        "n_images": n,
        "classes": cfg["data"]["classes"],
        "model": cfg["model"]["weights"],
        "box_conf": cfg["model"]["box_conf"],
        "target_precision": cfg["calibration"]["target_precision"],
        "threshold": chosen["threshold"] if chosen else None,
        "count_cap_p95": count_cap,
        "n_auto_accepted": len(acc),
        "n_flagged": len(flg),
        "pct_auto_accepted": round(100 * len(acc) / n, 1),
        "pct_flagged": round(100 * len(flg) / n, 1),
        "auto_accepted_precision": round(sum(r["correct"] for r in acc) / len(acc), 4) if acc else None,
        "auto_accepted_errors": sum(not r["correct"] for r in acc),
        "auto_accepted_error_rate": round(sum(not r["correct"] for r in acc) / len(acc), 4) if acc else None,
        "flagged_that_were_actually_correct": sum(r["correct"] for r in flg),
        "flag_reason_counts": reasons,
        "n_zero_detections": n_zero,
        "overall_image_accuracy_no_review": round(sum(r["correct"] for r in rows) / n, 4),
        "time_estimate_note": "rough estimate from assumed per-image times, not a measurement",
        "time_assumptions_s": {k: tm[k] for k in ("manual_label_s", "verify_s", "fix_s")},
        "time_estimate": time_estimate(len(acc), len(flg), n_zero, tm, tm["manual_label_s"]),
        "time_sensitivity": [time_estimate(len(acc), len(flg), n_zero, tm, m)
                             for m in tm["sensitivity_manual_s"]],
        **extra,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "threshold_sweep.csv").write_text(pd.DataFrame(curve).to_csv(index=False))
    return summary
