"""End-to-end: python -m pipeline.run [--skip-labellerr]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from . import calibrate, data, predict, report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "pipeline" / "config.yaml"))
    ap.add_argument("--skip-labellerr", action="store_true", help="run everything except the upload")
    args = ap.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load(Path(args.config).read_text())
    c = cfg["calibration"]

    print("[1/5] dataset")
    sel = data.select_images(cfg)
    image_dir = data.download_images(cfg, sel["images"])
    print(f"      {len(sel['images'])} images in {image_dir}")

    print("[2/5] pre-label with", cfg["model"]["weights"])
    preds = predict.predict(cfg, image_dir, sel["images"])
    cache = ROOT / "data" / "predictions.json"
    cache.write_text(json.dumps(preds))

    print("[3/5] calibrate")
    rows = []
    for im in sel["images"]:
        fn = im["file_name"]
        p, g = preds[fn], sel["gt"][fn]
        ev = calibrate.evaluate_image(p["boxes"], g, c["match_iou"], cfg["data"]["min_gt_area"])
        rows.append({"file_name": fn, "n_gt": sum(not b["ignore"] for b in g),
                     "n_det": p["n_det"], "min_conf": p["min_conf"], "mean_conf": p["mean_conf"],
                     "classes_pred": " ".join(sorted({b["cls"] for b in p["boxes"]})),
                     **ev})
    curve, count_cap = calibrate.sweep(rows, cfg)
    chosen = calibrate.choose_threshold(curve, c["target_precision"])
    if chosen is None:
        out = Path(cfg["results_dir"])
        out.mkdir(parents=True, exist_ok=True)
        report.plot_curve(curve, None, c["target_precision"], out / "threshold_curve.png")
        (out / "threshold_sweep.csv").write_text(report.pd.DataFrame(curve).to_csv(index=False))
        raise SystemExit(f"No threshold reaches precision {c['target_precision']:.0%}; "
                         "see results/threshold_curve.png and adjust the target or image score.")
    t = chosen["threshold"]
    for r in rows:
        reasons = calibrate.flag_reasons(r, t, count_cap)
        r["flagged"] = bool(reasons)
        r["flag_reason"] = ";".join(reasons)
        r["decision"] = "review" if reasons else "auto_accept"
    print(f"      threshold={t:.2f}  accept={chosen['auto_accept_rate']:.0%}  "
          f"precision={chosen['precision']:.1%}  count_cap(p95)={count_cap:g}")

    extra = {}
    if not args.skip_labellerr:
        print("[4/5] Labellerr upload")
        from . import labellerr_io
        extra["labellerr"] = labellerr_io.sync(cfg, image_dir, sel["images"], preds, rows)
    else:
        print("[4/5] Labellerr upload skipped (--skip-labellerr)")
        extra["labellerr"] = {"status": "skipped"}

    print("[5/5] report")
    s = report.write_outputs(rows, curve, chosen, count_cap, cfg, extra)
    te = s["time_estimate"]
    print(f"      auto-accepted {s['n_auto_accepted']}/{s['n_images']}, flagged {s['n_flagged']}, "
          f"errors slipped {s['auto_accepted_errors']}; time {te['baseline_min']} -> "
          f"{te['pipeline_min']} min ({te['saved_pct']}% saved, rough estimate)")


if __name__ == "__main__":
    main()
