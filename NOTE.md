# Option B: pre-label with a model, flag low-confidence images for review

**What I built.** A small pipeline that pre-labels 100 COCO val2017 images (person / car / dog)
with YOLOv8n, scores how much each image can be trusted, auto-accepts the confident ones, and
pushes the pre-labels to Labellerr with the uncertain images marked for human review. It is
in the spirit of my earlier project SCALES (a pipeline gated by a confidence check), rebuilt
from scratch for this assignment.

```
COCO val2017 (100 imgs + GT) ─► YOLOv8n (CPU, boxes ≥ 0.25) ─► image score = min box conf
        │                                                              │
        └──── GT used only to calibrate ───► threshold sweep 0.05–0.95 ┘
                                                     │
            auto-accept: score ≥ t, ≥1 box, #boxes ≤ p95      flag: everything else (+ reason)
                       │                                          │
   Labellerr SDK: dataset ─► template (3 bbox classes) ─► project ─► pre-annotations
                       conf_bucket="high"                      conf_bucket="low"
```

**How the threshold was set.** I matched predictions to ground truth (same class, IoU ≥ 0.5;
GT boxes under 32×32 px or marked crowd are ignored). An image counts as *correct* only if every
GT box is found and there are no false positives. Image score = the lowest box confidence in
the image. I swept t from 0.05 to 0.95 and measured, at each value, the auto-accept rate and the
precision of the auto-accepted set (`results/threshold_curve.png`, `threshold_sweep.csv`).
The planned target was 95%, but **no threshold reaches it**: the curve peaks at **90.6%** (t = 0.77).
The high-confidence errors are all *missed* objects, and a min-confidence score can't see a
box that isn't there. Two alternatives didn't fix it: also flagging near-miss boxes
(0.10–0.25) peaked at 94% with only 17 images accepted, and YOLOv8s peaked at 88%. So the target
was set to **90%**, and the lowest threshold meeting it is **t = 0.77**. Images are also flagged
if they have zero detections or more boxes than the 95th percentile (6).

**Results (N = 100, from `results/summary.json`).**

| | |
|---|---|
| Auto-accepted | **32%** (32 images), precision 90.6% |
| Flagged for review | **68%** (62 low min-conf, 6 zero detections, 2 > p95 box count; reasons overlap) |
| Errors among auto-accepted | **3 / 32 (9.4%)**, each one missed object, no wrong boxes |
| Flagged but actually correct | 24 / 68 (review effort that wasn't strictly needed) |
| Without any review | only 53% of images would be fully correct |

**Time saved (rough estimate, not a measurement).** Assumptions are in `config.yaml`: 30 s to label
from scratch, 3 s to verify an auto-accepted pre-label, 12 s to fix a flagged one (zero-detection
images are charged the full 30 s). Baseline **50.0 min** vs pipeline **17.0 min**, so **≈ 66% saved**.
Most of the saving comes from fixing pre-labels instead of drawing boxes, not from auto-accepting.

| manual s / image | baseline | pipeline | saved |
|---|---|---|---|
| 20 | 33.3 min | 16.0 min | 52% |
| 30 | 50.0 min | 17.0 min | 66% |
| 60 | 100.0 min | 20.0 min | 80% |

**Labellerr: verified end to end** (SDK 1.3.2, project `perla_lonely_marten_34549`). The pipeline
creates a dataset (100 files) → annotation template (one bbox class each for person/car/dog) →
project, then uploads COCO pre-annotations in two files: flagged images with `conf_bucket="low"`
(62 images / 196 boxes) and auto-accepted ones with `conf_bucket="high"` (32 / 47). Both jobs
returned `completed`; the 6 zero-detection images have nothing to upload. IDs are in
`results/labellerr_ids.json`, and a re-run creates nothing new. **In the web UI** (checked by hand) the
pre-labels show up in the Label tool's *AI Mode / Model Assisted* panel as ⚡ model suggestions, with a
confidence slider and accept/reject buttons. They are *suggestions*, not annotations: the
Annotations panel stays at 0 until one is accepted, so "verify" in the time model includes those
accept clicks. I couldn't find where `conf_bucket` low/high appears in the UI, so the split by bucket
isn't visible to reviewers yet; the flag and its reason per image live in `results/per_image.csv`.
**Not done:** moving flagged files to a review status. `list_files` first rejected the
SDK-documented dict query, then returned 0 files for a few minutes while the UI already showed 100,
so `bulk_assign_files` was never called (I stopped after 2 attempts). The pipeline changed
no file status. 19 friction points are in `SDK_FEEDBACK.md`, including one security issue (#17):
pre-annotation upload stores the API key in each file's `email_id` field.

**Limitations & next steps.** 100 images is a small calibration set: ±1 error moves precision
by about 3 points, so the threshold needs a held-out set (or bootstrap intervals) before real use.
Min-confidence ignores missed objects. A recall signal (a second model, test-time augmentation, or
an expected object count) is the main lever toward 95%. Next steps: per-class thresholds
(dogs vs small cars behave differently); calibrated scores (temperature scaling) so 0.77 means
the same thing across models; and active learning, where reviewer fixes on flagged images feed
fine-tuning and re-calibration, so the auto-accept share grows over time.
