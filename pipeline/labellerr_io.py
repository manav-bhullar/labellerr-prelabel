"""Labellerr side: dataset -> template -> project -> pre-annotations (bucketed) -> review mark.

Uses labellerr-sdk 1.3.2 (github.com/Labellerr/SDKPython). Idempotent: every created ID is
written to results/labellerr_ids.json immediately, and re-runs reuse them. Pre-annotation
uploads are skipped when the same COCO file (by sha256) was already uploaded.

How flagged images are marked (decided with Manav):
  * flagged images' pre-labels are uploaded with conf_bucket="low",
    auto-accepted ones with conf_bucket="high";
  * then ONE flagged file is moved to status "review" via bulk_assign_files as a trial
    (max 2 attempts). The per-image reason lives in results/per_image.csv.
Credentials come from LABELLERR_API_KEY / LABELLERR_API_SECRET / LABELLERR_CLIENT_ID and
are never printed or written.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ENV = ("LABELLERR_API_KEY", "LABELLERR_API_SECRET", "LABELLERR_CLIENT_ID")
REVIEW_STATUS = "review"  # status name used in the SDK source (constants.LOCAL_EXPORT_STATUS, tests)


def _env(k: str) -> str:
    """Read a credential, tolerating stray whitespace / trailing ';' pasted into env settings."""
    return (os.environ.get(k) or "").strip().rstrip(";").strip()


def _scrub(msg) -> str:
    s = str(msg)
    for k in ENV:
        v = _env(k)
        if v:
            s = s.replace(v, f"<{k}>")
    return s[:500]


def _save(ids: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids, indent=2))


def build_coco(cfg: dict, images: list[dict], preds: dict, subset: set[str]) -> dict:
    """COCO JSON for a subset of images. file_name must equal the uploaded file name exactly."""
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(cfg["data"]["classes"])]
    cat_id = {c["name"]: c["id"] for c in cats}
    out = {"images": [], "annotations": [], "categories": cats}
    aid = 1
    for i, im in enumerate(images, start=1):
        fn = im["file_name"]
        if fn not in subset or not preds[fn]["boxes"]:
            continue
        out["images"].append({"id": i, "file_name": fn, "width": im["width"], "height": im["height"]})
        for b in preds[fn]["boxes"]:
            x, y, w, h = b["bbox"]
            out["annotations"].append({"id": aid, "image_id": i, "category_id": cat_id[b["cls"]],
                                       "bbox": [x, y, w, h], "area": round(w * h, 2), "iscrowd": 0,
                                       "score": b["conf"]})
            aid += 1
    return out


def _find_file_id(project, file_name: str) -> str | None:
    """Page through project files; the response shape isn't documented, so search it."""
    def walk(o):
        if isinstance(o, dict):
            if o.get("file_name") == file_name and o.get("file_id"):
                return o["file_id"]
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r:
                    return r
        return None

    cursor = None
    for _ in range(20):  # 20 x 50 > 100 files
        res = project.list_files(search_queries={}, size=50, next_search_after=cursor)
        fid = walk(res)
        if fid:
            return fid
        resp = res.get("response", {}) if isinstance(res, dict) else {}
        cursor = resp.get("next_search_after") if isinstance(resp, dict) else None
        if not cursor:
            return None
    return None


def sync(cfg: dict, image_dir: Path, images: list[dict], preds: dict, rows: list[dict]) -> dict:
    missing = [k for k in ENV if not _env(k)]
    if missing:
        print(f"      Labellerr skipped: missing env var(s) {missing}")
        return {"status": "skipped_missing_credentials", "missing_env": missing}

    from labellerr.core.annotation_templates import LabellerrAnnotationTemplate, create_template
    from labellerr.core.client import LabellerrClient
    from labellerr.core.datasets import LabellerrDataset, create_dataset_from_local
    from labellerr.core.projects import LabellerrProject, create_project
    from labellerr.core.schemas import (AnnotationQuestion, CreateProjectParams, CreateTemplateParams,
                                        DatasetConfig, DatasetDataType, QuestionType, RotationConfig)

    L = cfg["labellerr"]
    ids_path = Path(L["ids_file"])
    ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    client = LabellerrClient(api_key=_env(ENV[0]), api_secret=_env(ENV[1]), client_id=_env(ENV[2]))
    log = {"status": "started"}
    try:
        # 1. dataset
        if ids.get("dataset_id"):
            dataset = LabellerrDataset(client=client, dataset_id=ids["dataset_id"])
            print(f"      reuse dataset {ids['dataset_id']}")
        else:
            dataset = create_dataset_from_local(
                client=client,
                dataset_config=DatasetConfig(dataset_name=L["dataset_name"], data_type="image",
                                             dataset_description="100 COCO val2017 images (person/car/dog)"),
                folder_to_upload=str(image_dir))
            ids["dataset_id"] = dataset.dataset_id
            _save(ids, ids_path)
            dataset.status()  # waits until files are processed
            print(f"      created dataset {dataset.dataset_id}")

        # 2. annotation template: one bounding-box question per class (names == COCO category names)
        if ids.get("template_id"):
            template = LabellerrAnnotationTemplate(client=client, annotation_template_id=ids["template_id"])
        else:
            qs = [AnnotationQuestion(question_number=i + 1, question=c, question_type=QuestionType.bounding_box,
                                     required=False, color=L["colors"][c])
                  for i, c in enumerate(cfg["data"]["classes"])]
            template = create_template(client, CreateTemplateParams(
                template_name=L["template_name"], data_type=DatasetDataType.image, questions=qs))
            ids["template_id"] = template.annotation_template_id
            _save(ids, ids_path)
            print(f"      created template {ids['template_id']}")

        # 3. project
        if ids.get("project_id"):
            project = LabellerrProject(client, project_id=ids["project_id"])
            print(f"      reuse project {ids['project_id']}")
        else:
            project = create_project(client, CreateProjectParams(
                project_name=L["project_name"], data_type=DatasetDataType.image,
                rotations=RotationConfig(annotation_rotation_count=1, review_rotation_count=1,
                                         client_review_rotation_count=1)),
                datasets=[dataset], annotation_template=template)
            ids["project_id"] = project.project_id
            _save(ids, ids_path)
            print(f"      created project {ids['project_id']}")
            try:
                project.status()  # wait for project setup before uploading answers
            except Exception as e:  # noqa: BLE001 - informational only
                print(f"      project.status() warning: {_scrub(e)}")

        # 4. pre-annotations, split by confidence bucket
        flagged = {r["file_name"] for r in rows if r["flagged"]}
        accepted = {r["file_name"] for r in rows if not r["flagged"]}
        out_dir = Path("data/labellerr")
        out_dir.mkdir(parents=True, exist_ok=True)
        ids.setdefault("preannotations", {})
        for bucket, subset in (("low", flagged), ("high", accepted)):
            coco = build_coco(cfg, images, preds, subset)
            p = out_dir / f"preannotations_{bucket}.json"
            p.write_text(json.dumps(coco))
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            prev = ids["preannotations"].get(bucket, {})
            if prev.get("sha256") == sha and prev.get("status") == "completed":
                print(f"      pre-annotations[{bucket}] already uploaded, skipping")
                continue
            res = project.upload_preannotations(annotation_format=L["annotation_format"],
                                                annotation_file=str(p), conf_bucket=bucket)
            status = (res.get("response") or {}).get("status") if isinstance(res, dict) else None
            ids["preannotations"][bucket] = {"sha256": sha, "status": status,
                                             "n_images": len(coco["images"]),
                                             "n_boxes": len(coco["annotations"])}
            _save(ids, ids_path)
            print(f"      pre-annotations[{bucket}]: {len(coco['images'])} images, "
                  f"{len(coco['annotations'])} boxes -> {status}")

        # 5. trial: move ONE flagged file to the review status (max 2 attempts)
        trial = ids.get("bulk_assign_trial")
        if not trial or (trial.get("outcome") != "ok" and len(trial.get("attempts", [])) < 2):
            trial = trial or {"status": REVIEW_STATUS, "attempts": []}
            target = sorted(flagged)[0] if flagged else None
            trial["file_name"] = target
            while target and len(trial["attempts"]) < 2 and trial.get("outcome") != "ok":
                try:
                    fid = trial.get("file_id") or _find_file_id(project, target)
                    if not fid:
                        raise RuntimeError(f"file_id for {target} not found via list_files")
                    trial["file_id"] = fid
                    r = project.bulk_assign_files([fid], new_status=REVIEW_STATUS)
                    trial["attempts"].append({"ok": True, "response": _scrub(r)})
                    trial["outcome"] = "ok"
                except Exception as e:  # noqa: BLE001 - record and stop after 2
                    trial["attempts"].append({"ok": False, "error": _scrub(e)})
                    trial["outcome"] = "failed"
                ids["bulk_assign_trial"] = trial
                _save(ids, ids_path)
            print(f"      bulk_assign trial on {target}: {trial.get('outcome')}")
        log["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - surface a scrubbed error, keep saved IDs
        log.update(status="error", error=_scrub(e))
        print(f"      Labellerr error: {log['error']}")
    finally:
        client.close()
    log["ids"] = {k: v for k, v in ids.items()}
    return log
