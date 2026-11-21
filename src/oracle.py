"""Шумний оракул: вносить у розмітку пропуски, зсуви й хибні рамки і журналює запити (JSONL).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SHIFT_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass
class NoiseConfig:
    frac_noisy_images: float = 0.3
    p_miss: float = 0.20        # у «шумному» зображенні: ймовірність видалити рамку
    p_shift: float = 0.15       # ... зсунути рамку
    rate_spurious: float = 0.10  # ... кількість хибних рамок = Binomial(n_true, rate)
    shift_min: float = 0.5      # зсув у частках розміру рамки -> IoU <= 0,33
    shift_max: float = 1.0
    seed: int = 0


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU між рамками a (N,4) і b (M,4) у форматі x,y,w,h -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def make_noisy(coco: dict, cfg: NoiseConfig) -> tuple[dict, list[dict]]:
    """Повертає (зашумлений COCO, список внесених помилок)."""
    rng = np.random.default_rng(cfg.seed)
    images = {im["id"]: im for im in coco["images"]}
    by_img: dict[int, list[dict]] = {i: [] for i in images}
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    img_ids = sorted(images)
    n_noisy = round(cfg.frac_noisy_images * len(img_ids))
    noisy_imgs = set(rng.choice(img_ids, size=n_noisy, replace=False).tolist())

    noisy_anns, errors = [], []
    next_id = max(a["id"] for a in coco["annotations"]) + 1
    for iid in img_ids:
        W, H = images[iid]["width"], images[iid]["height"]
        anns = by_img[iid]
        if iid not in noisy_imgs:
            noisy_anns.extend({**a, "attributes": dict(a["attributes"])} for a in anns)
            continue
        for a in anns:
            u = rng.random()
            if u < cfg.p_miss:
                errors.append({"type": "miss", "image_id": iid, "true_ann_id": a["id"],
                               "true_bbox": a["bbox"], "noisy_bbox": None})
                continue
            b = {**a, "attributes": dict(a["attributes"])}
            if u < cfg.p_miss + cfg.p_shift:
                x, y, w, h = a["bbox"]
                d = rng.uniform(cfg.shift_min, cfg.shift_max)
                # один з 8 напрямків; зсув на d частки ширини/висоти рамки
                cx, cy = SHIFT_DIRS[rng.integers(len(SHIFT_DIRS))]
                nx = float(np.clip(x + cx * d * w, 0, max(W - w, 0)))
                ny = float(np.clip(y + cy * d * h, 0, max(H - h, 0)))
                if iou_xywh(np.array([[nx, ny, w, h]]), np.array([a["bbox"]]))[0, 0] < 0.5:
                    b["bbox"] = [nx, ny, w, h]
                    errors.append({"type": "shift", "image_id": iid, "true_ann_id": a["id"],
                                   "true_bbox": a["bbox"], "noisy_bbox": b["bbox"],
                                   "noisy_ann_id": a["id"]})
            noisy_anns.append(b)
        # хибні рамки
        n_sp = rng.binomial(len(anns), cfg.rate_spurious) if anns else 0
        true_boxes = np.array([a["bbox"] for a in anns]) if anns else np.zeros((0, 4))
        for _ in range(n_sp):
            src = anns[rng.integers(len(anns))]["bbox"] if anns else [0, 0, 10, 20]
            w, h = src[2], src[3]
            for _try in range(30):
                cand = np.array([[rng.uniform(0, max(W - w, 1)), rng.uniform(0, max(H - h, 1)), w, h]])
                if len(true_boxes) == 0 or iou_xywh(cand, true_boxes).max() < 0.1:
                    box = [float(v) for v in cand[0]]
                    noisy_anns.append({"id": next_id, "image_id": iid, "category_id": 1,
                                       "bbox": box, "area": w * h, "iscrowd": 0,
                                       "attributes": {"pose": "unknown", "visibility": "unknown",
                                                      "modality": "rgb",
                                                      "provenance": "original_dataset",
                                                      "confidence": 1.0}})
                    errors.append({"type": "spurious", "image_id": iid, "true_ann_id": None,
                                   "true_bbox": None, "noisy_bbox": box, "noisy_ann_id": next_id})
                    next_id += 1
                    break

    for k, e in enumerate(errors):
        e["error_id"] = k
    noisy = {**coco, "annotations": noisy_anns,
             "info": {**coco.get("info", {}), "noise": asdict(cfg)}}
    return noisy, errors


class Oracle:
    """Відповідає на запити перевірки зображень і веде журнал запитів."""

    def __init__(self, truth: dict, errors: list[dict], log_path: Path, run_tag: str):
        self.truth_by_img: dict[int, list[dict]] = {}
        for a in truth["annotations"]:
            self.truth_by_img.setdefault(a["image_id"], []).append(a)
        self.errors_by_img: dict[int, list[dict]] = {}
        for e in errors:
            self.errors_by_img.setdefault(e["image_id"], []).append(e)
        self.n_errors = len(errors)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_tag = run_tag
        self.n_queries = 0

    def audit(self, image_id: int, strategy: str, score: float | None = None) -> dict:
        errs = self.errors_by_img.get(image_id, [])
        truth = self.truth_by_img.get(image_id, [])
        self.n_queries += 1
        rec = {"ts": time.time(), "run": self.run_tag, "strategy": strategy,
               "query_idx": self.n_queries, "image_id": image_id, "score": score,
               "n_true_boxes": len(truth), "n_errors_found": len(errs),
               "error_ids": [e["error_id"] for e in errs],
               "error_types": [e["type"] for e in errs]}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return {"truth": truth, "errors": errs}
