"""Select ~100 COCO val2017 images for a few classes and download only those images."""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def ensure_annotations(cfg: dict) -> Path:
    """Download the COCO annotation zip once and keep only instances_val2017.json."""
    ann_path = ROOT / cfg["data"]["annotations"]
    if ann_path.exists():
        return ann_path
    import zipfile

    zip_path = ann_path.parent.parent / "annotations_trainval2017.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading COCO annotations (~250 MB) -> {zip_path}")
    urllib.request.urlretrieve(ANN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2017.json", zip_path.parent)
    zip_path.unlink()
    return ann_path


def select_images(cfg: dict) -> dict:
    """Return {"images": [...], "gt": {file_name: [ {cls, bbox_xywh, ignore} ]}}.

    Sampling: for each class, pick `per_class` images that contain it, with at most
    `max_gt_per_image` relevant boxes, deterministic by seed. GT boxes that are crowd or
    smaller than `min_gt_area` are kept but marked ignore=True.
    """
    d = cfg["data"]
    coco = json.loads(ensure_annotations(cfg).read_text())
    name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    wanted = {cid for cid, n in name_by_id.items() if n in d["classes"]}

    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        if a["category_id"] in wanted:
            anns_by_img.setdefault(a["image_id"], []).append(a)
    img_by_id = {im["id"]: im for im in coco["images"]}

    rng = random.Random(d["seed"])
    chosen: list[int] = []
    for cls in d["classes"]:
        pool = sorted(
            iid for iid, anns in anns_by_img.items()
            if iid not in chosen
            and len(anns) <= d["max_gt_per_image"]
            and any(name_by_id[a["category_id"]] == cls and not a["iscrowd"]
                    and a["area"] >= d["min_gt_area"] for a in anns)
        )
        chosen += rng.sample(pool, d["per_class"])
    # 3 x 34 = 102 candidates -> keep exactly n_images, deterministically
    chosen = sorted(rng.sample(chosen, min(d["n_images"], len(chosen))))

    images, gt = [], {}
    for iid in chosen:
        im = img_by_id[iid]
        images.append({"id": iid, "file_name": im["file_name"],
                       "width": im["width"], "height": im["height"],
                       "license": im.get("license")})
        gt[im["file_name"]] = [
            {"cls": name_by_id[a["category_id"]], "bbox": a["bbox"],
             "ignore": bool(a["iscrowd"]) or a["area"] < d["min_gt_area"]}
            for a in anns_by_img[iid]
        ]
    licenses = {l["id"]: l for l in coco.get("licenses", [])}
    return {"images": images, "gt": gt, "licenses": licenses}


def download_images(cfg: dict, images: list[dict]) -> Path:
    out = ROOT / cfg["data"]["image_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for im in images:
        dst = out / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(cfg["data"]["image_url"].format(file_name=im["file_name"]), dst)
    # remove stale images from older selections so the upload folder matches exactly
    keep = {im["file_name"] for im in images}
    for p in out.iterdir():
        if p.name not in keep:
            p.unlink()
    return out
