"""Reproduce the dataset only: COCO val2017 annotations + the ~100 selected images.

    python scripts/download_data.py
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import data  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "pipeline" / "config.yaml").read_text())
    import os
    os.chdir(root)
    sel = data.select_images(cfg)
    out = data.download_images(cfg, sel["images"])
    print(f"{len(sel['images'])} images in {out}")
