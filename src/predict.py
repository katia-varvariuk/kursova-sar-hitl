"""Прогнозний модуль: зображення -> рамки людей з рішенням маршрутизації.

  python src/predict.py path/to/image.jpg [--sahi]
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/yolo11n_sard.pt"
MODEL_VERSION = "yolo11n_sard_al_random_seed0_r6"
CONF_MIN, CONF_ACCEPT, IOU_NMS = 0.05, 0.3, 0.6  # пороги з експерименту маршрутизації


@lru_cache(maxsize=2)
def _load(weights: str, sahi: bool):
    if sahi:
        from sahi import AutoDetectionModel
        return AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights,
                                                  confidence_threshold=CONF_MIN, image_size=640)
    from ultralytics import YOLO
    return YOLO(weights)


def predict(image, sahi: bool = False, weights: Path = WEIGHTS) -> dict:
    """image: шлях, PIL.Image або масив HxWx3 (RGB). Повертає словник за контрактом."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("очікується шлях, PIL.Image або numpy-масив")
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 32:
        raise ValueError(f"зображення замале: {w}x{h}")

    det, t0 = _load(str(weights), sahi), time.perf_counter()
    boxes = []
    if sahi:
        from sahi.predict import get_sliced_prediction
        r = get_sliced_prediction(image, det, slice_height=320, slice_width=320,
                                  overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                  postprocess_type="NMS", postprocess_match_metric="IOU",
                                  postprocess_match_threshold=IOU_NMS, verbose=0)
        raw = [(p.bbox.to_xyxy(), float(p.score.value)) for p in r.object_prediction_list]
    else:
        r = det.predict(image, conf=CONF_MIN, iou=IOU_NMS, max_det=300, imgsz=640, verbose=False)[0]
        raw = list(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()))
    for (x1, y1, x2, y2), s in sorted(raw, key=lambda t: -t[1]):
        boxes.append({"bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                      "score": round(s, 4), "category": "person",
                      "route": "auto_accept" if s >= CONF_ACCEPT else "human_check"})
    return {"model_version": MODEL_VERSION, "image_size": [w, h], "sahi": sahi,
            "n_auto_accept": sum(b["route"] == "auto_accept" for b in boxes),
            "n_human_check": sum(b["route"] == "human_check" for b in boxes),
            "boxes": boxes, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--sahi", action="store_true")
    a = ap.parse_args()
    print(json.dumps(predict(a.image, sahi=a.sahi), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
