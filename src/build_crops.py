"""Маніфест кропів (JSONL): кроп людини, кроп з контекстом x3, ключові точки, атрибути, split.

  python src/build_crops.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
OUT = PROC / "crops"
MANIFEST = PROC / "manifest/crops.jsonl"
CONTEXT_SCALE = 3.0
MIN_POSE_PX = 32
GUARD = 3
VAL_FRAC = 0.2
SEED = 42

# Тип середовища (розд. 4): rubble / water / fire_smoke / traffic / forest_mountain / other
ENV_C2A = {"collapsed_building": "rubble", "flood": "water", "fire": "fire_smoke",
           "traffic_incident": "traffic"}
ENV_SARD = "forest_mountain"   # SARD: інсценовані пошуки в природній місцевості (рівень датасету)


def load(ds: str, split: str) -> dict:
    return json.loads((PROC / ds / f"{split}.json").read_text(encoding="utf-8"))


def sard_pool_split(pool: dict) -> tuple[dict[int, str], int]:
    """train/val для SARD pool блоками сцен + буфер GUARD кадрів між частинами."""
    scenes = sorted({im["scene"] for im in pool["images"]})
    rng = random.Random(SEED)
    val_scenes = set(rng.sample(scenes, max(1, round(VAL_FRAC * len(scenes)))))
    part = {im["id"]: ("val" if im["scene"] in val_scenes else "train") for im in pool["images"]}
    frames = defaultdict(set)
    for im in pool["images"]:
        frames[part[im["id"]]].add(im["frame_id"])
    dropped = 0
    for im in pool["images"]:
        other = frames["train" if part[im["id"]] == "val" else "val"]
        if any(abs(im["frame_id"] - f) <= GUARD for f in other):
            part[im["id"]] = None
            dropped += 1
    return part, dropped


def hitl_labels() -> dict[int, list[dict]]:
    """Фінальні рамки зображень SARD pool, що пройшли оркестратор."""
    p = ROOT / "data/work/orchestrator/export_sard.json"
    if not p.exists():
        return {}
    coco = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, list] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        out[a["image_id"]].append(a)
    return out


def context_box(b, W, H, scale=CONTEXT_SCALE):
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = max(0.0, cx - scale * w / 2), max(0.0, cy - scale * h / 2)
    x1, y1 = min(float(W), cx + scale * w / 2), min(float(H), cy + scale * h / 2)
    return [x0, y0, x1 - x0, y1 - y0]


def to_int_box(b):
    x, y, w, h = b
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def in_group(i: int, boxes: list) -> bool:
    """«У групі»: центр іншої людини лежить у кропі з контекстом цієї людини."""
    x, y, w, h = boxes[i]
    cx0, cy0, cw, ch = x + w / 2 - 1.5 * w, y + h / 2 - 1.5 * h, 3 * w, 3 * h
    for j, (a, b, c, d) in enumerate(boxes):
        if j != i and cx0 <= a + c / 2 <= cx0 + cw and cy0 <= b + d / 2 <= cy0 + ch:
            return True
    return False


def iou_xywh(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="лише перші N зображень кожного спліту")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--pose-device", default="cpu")
    a = ap.parse_args()

    pose = None
    if not a.no_pose:
        from ultralytics import YOLO
        pose = YOLO(a.pose_model)

    sard_pool = load("sard", "pool")
    pool_part, n_guard = sard_pool_split(sard_pool)
    hitl = hitl_labels()
    sources = [  # (датасет, спліт COCO, функція -> спліт маніфесту)
        ("c2a", "train", lambda im: "train"),
        ("c2a", "val", lambda im: "val"),
        ("sard", "pool", lambda im: pool_part[im["id"]]),
        ("sard", "test", lambda im: "test"),
    ]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with open(MANIFEST, "w", encoding="utf-8") as fout:
        for ds, coco_split, split_of in sources:
            coco = sard_pool if (ds, coco_split) == ("sard", "pool") else load(ds, coco_split)
            by_img = defaultdict(list)
            for ann in coco["annotations"]:
                by_img[ann["image_id"]].append(ann)
            images = sorted(coco["images"], key=lambda im: im["id"])
            if a.limit:
                images = images[:a.limit]
            for im in images:
                split = split_of(im)
                if split is None:
                    stats["images_dropped_guard"] += 1
                    continue
                anns = by_img.get(im["id"], [])
                if ds == "sard" and coco_split == "pool" and im["id"] in hitl:
                    anns = hitl[im["id"]]
                    stats["images_from_hitl"] += 1
                if not anns:
                    continue
                img = Image.open(PROC / ds / "images" / im["file_name"]).convert("RGB")
                W, H = img.size
                assert (W, H) == (im["width"], im["height"]), im["file_name"]
                boxes = [x["bbox"] for x in anns]
                for k, ann in enumerate(anns):
                    crop_id = f"{ds}_{coco_split}_{im['id']:06d}_{k:03d}"
                    sub = OUT / split
                    crop_rel = f"crops/{split}/person/{crop_id}.jpg"
                    ctx_rel = f"crops/{split}/context/{crop_id}.jpg"
                    (sub / "person").mkdir(parents=True, exist_ok=True)
                    (sub / "context").mkdir(parents=True, exist_ok=True)
                    bb = [float(v) for v in ann["bbox"]]
                    cb = context_box(bb, W, H)
                    img.crop(to_int_box(bb)).save(PROC / crop_rel, quality=95)
                    ctx = img.crop(to_int_box(cb))
                    ctx.save(PROC / ctx_rel, quality=95)

                    kp, kp_note = None, "not_run"
                    if pose is not None:
                        if min(bb[2], bb[3]) < MIN_POSE_PX:
                            kp_note = f"too_small_lt_{MIN_POSE_PX}px"
                        else:
                            r = pose.predict(ctx, conf=0.25, verbose=False, device=a.pose_device)[0]
                            ox, oy = to_int_box(cb)[:2]
                            best, best_iou = None, 0.3
                            for d in range(len(r.boxes)):
                                x0, y0, x1, y1 = r.boxes.xyxy[d].tolist()
                                v = iou_xywh([x0 + ox, y0 + oy, x1 - x0, y1 - y0], bb)
                                if v >= best_iou:
                                    best, best_iou = d, v
                            if best is None:
                                kp_note = "no_match"
                            else:
                                xy = r.keypoints.xy[best].tolist()
                                cf = r.keypoints.conf[best].tolist()
                                kp = [[round(x + ox, 1), round(y + oy, 1), round(c, 3)]
                                      for (x, y), c in zip(xy, cf)]
                                kp_note = "yolo11n-pose"
                    stats[f"keypoints_{kp_note}"] += 1

                    attrs = ann.get("attributes", {})
                    env = ENV_C2A.get(im.get("disaster_type"), "other") if ds == "c2a" else ENV_SARD
                    rec = {
                        "crop_id": crop_id, "source_dataset": ds, "source_split": coco_split,
                        "image_id": im["id"],
                        "file_name": im["file_name"], "scene": im.get("scene"),
                        "bbox": [round(v, 2) for v in bb],
                        "keypoints": kp,
                        "crop_path": crop_rel, "context_path": ctx_rel,
                        "context_bbox": [round(v, 2) for v in cb],
                        "priority": "undetermined",
                        "attributes": {
                            "pose": attrs.get("pose", "unknown"),
                            "visibility": attrs.get("visibility", "unknown"),
                            "modality": attrs.get("modality", "rgb"),
                            "environment": env,
                            "group": "group" if in_group(k, boxes) else "single",
                            "gesture": "unknown",
                        },
                        "provenance": attrs.get("provenance", "original_dataset"),
                        "attribute_provenance": {
                            "pose": "original_dataset" if attrs.get("pose", "unknown") != "unknown" else None,
                            "environment": "original_dataset",
                            "group": "rule",
                            "keypoints": "model" if kp else None,
                            "priority": None,
                        },
                        "confidence": float(attrs.get("confidence", 1.0)),
                        "split": split,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats[f"records_{split}"] += 1
                    stats[f"records_{ds}_{split}"] += 1
    stats["sard_pool_guard_dropped_images"] = n_guard
    (MANIFEST.parent / "build_stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
