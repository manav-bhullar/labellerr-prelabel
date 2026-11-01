# Labellerr pre-labelling with confidence-based review flagging

Take-home (Option B) for Labellerr: pre-label ~100 images with a detector, upload the
pre-labels to Labellerr, and flag low-confidence images for human review.
See **[NOTE.md](NOTE.md)** for the write-up and **[SDK_FEEDBACK.md](SDK_FEEDBACK.md)** for SDK notes.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # includes labellerr-sdk from GitHub (not on PyPI)
export LABELLERR_API_KEY=...  LABELLERR_API_SECRET=...  LABELLERR_CLIENT_ID=...
python -m pipeline.run                 # add --skip-labellerr to run offline only
```

The first run downloads the COCO annotation zip (~250 MB, only `instances_val2017.json` is kept),
the 100 selected images (into `data/`, git-ignored) and `yolov8n.pt` (~6 MB). YOLO on CPU takes under a minute.
`python scripts/download_data.py` reproduces just the dataset.
If the Labellerr variables are missing, the upload step is skipped with a message.

## Layout

| path | what |
|---|---|
| `pipeline/config.yaml` | every knob: classes, sampling, model, precision target, time assumptions, Labellerr names |
| `pipeline/data.py` | deterministic COCO selection (seed 42) + image download |
| `pipeline/predict.py` | YOLOv8n inference, per-image boxes / min & mean conf / count |
| `pipeline/calibrate.py` | GT matching, threshold sweep, threshold choice, flag reasons |
| `pipeline/labellerr_io.py` | dataset → template → project → pre-annotations (`conf_bucket` low/high) → review trial; idempotent |
| `pipeline/report.py` | `results/` files + time estimate |
| `results/` | `threshold_curve.png`, `threshold_sweep.csv`, `per_image.csv`, `summary.json`, `labellerr_ids.json` |

## Data source and license

COCO 2017 validation set (<https://cocodataset.org>). Annotations are © COCO Consortium under
CC BY 4.0. Images come from Flickr under their individual licenses (listed in the COCO
annotation file). Images are downloaded at run time and **not** redistributed in this repo.
The model is Ultralytics YOLOv8n (AGPL-3.0), and the weights are downloaded at run time.
