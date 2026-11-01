# Labellerr pre-labelling with confidence-based review flagging

## The assignment (Option B)

> Take a small image dataset (about 100 images), use Labellerr to pre-label it with a model,
> and flag the low-confidence images for human review. Include a short note on how you set
> the threshold and a rough estimate of the review time saved.

Also requested: anything about the Labellerr SDK or platform that was hard or could be better.

## Where to find each part

| Asked for | Where |
|---|---|
| Pre-label ~100 images with a model, via Labellerr | `pipeline/` (YOLOv8n on 100 COCO images, uploaded as Labellerr pre-annotations) |
| Flag low-confidence images for human review | `pipeline/calibrate.py`, `results/per_image.csv` (flag + reason per image) |
| How the threshold was set | [NOTE.md](NOTE.md), `results/threshold_curve.png` |
| Rough estimate of review time saved | [NOTE.md](NOTE.md), `results/summary.json` |
| What was hard / could be better | [SDK_FEEDBACK.md](SDK_FEEDBACK.md) |

**Result in one line:** a threshold of 0.77 auto-accepts 32% of images at 90.6% precision and
flags 68% for review, for a rough estimate of about 66% less labelling time.

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
