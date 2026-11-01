"""Run YOLOv8 on the selected images and keep only the configured classes."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def predict(cfg: dict, image_dir: Path, images: list[dict]) -> dict:
    """Return {file_name: {"boxes": [{cls, conf, bbox(xywh)}], n_det, min_conf, mean_conf}}."""
    from ultralytics import YOLO

    m = cfg["model"]
    model = YOLO(m["weights"])
    names = model.names  # {idx: name}
    keep_idx = [i for i, n in names.items() if n in cfg["data"]["classes"]]

    preds = {}
    for im in images:
        r = model.predict(str(image_dir / im["file_name"]), conf=m["box_conf"], iou=m["iou_nms"],
                          imgsz=m["imgsz"], device=m["device"], classes=keep_idx, verbose=False)[0]
        boxes = []
        for xyxy, conf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            x1, y1, x2, y2 = xyxy
            boxes.append({"cls": names[int(c)], "conf": round(float(conf), 4),
                          "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]})
        confs = [b["conf"] for b in boxes]
        preds[im["file_name"]] = {
            "boxes": boxes,
            "n_det": len(boxes),
            "min_conf": min(confs) if confs else None,
            "mean_conf": round(sum(confs) / len(confs), 4) if confs else None,
        }
    return preds
