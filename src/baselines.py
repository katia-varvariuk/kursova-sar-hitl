"""Драбина базових рішень на тесті SARD: від «нічого не знаходити» до моделі після АМН.

  python src/baselines.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from exp_sahi import CONF_LOW, CONF_OP, IOU_MATCH, IOU_NMS, evaluate, iou_matrix

ROOT = Path(__file__).resolve().parents[1]
TEST, IMAGES = ROOT / "data/processed/sard/test.json", ROOT / "data/processed/sard/images"
LADDER = [
    ("0", "Dummy: жодної рамки", None),
    ("0'", "Правило без ML: колірна аномалія відносно фону", "rule"),
    ("1", "YOLO11n COCO, без донавчання", ROOT / "yolo11n.pt"),
    ("2", "YOLO11n, навчена на C2A", ROOT / "runs/baseline_c2a/yolo11n_e30/weights/best.pt"),
    ("3", "YOLO11n, C2A + АМН на SARD (умова А, seed 0)", ROOT / "models/yolo11n_sard.pt"),
]


def run_rule(coco: dict, k: float = 4.0) -> list[dict]:
    """Колірна аномалія відносно локального фону (RX-подібне правило), без навчання."""
    out = []
    for im in coco["images"]:
        lab = cv2.cvtColor(cv2.imread(str(IMAGES / im["file_name"])), cv2.COLOR_BGR2LAB).astype(np.float32)
        mu = cv2.blur(lab, (63, 63))
        sd = np.sqrt(np.maximum(cv2.blur(lab * lab, (63, 63)) - mu * mu, 1.0))
        z = np.sqrt((((lab - mu) / sd) ** 2).sum(2) / 3)
        mask = cv2.morphologyEx((z > k).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if 8 <= max(bw, bh) <= 200 and area >= 20:
                out.append({"image_id": im["id"], "category_id": 1, "bbox": [float(x), float(y), float(bw), float(bh)],
                            "score": float(min(z[lbl == i].mean() / (2 * k), 1.0))})
    return out


def run_yolo(coco: dict, weights: Path) -> list[dict]:
    from ultralytics import YOLO
    det = YOLO(str(weights))
    out = []
    for im in coco["images"]:
        r = det.predict(str(IMAGES / im["file_name"]), conf=CONF_LOW, iou=IOU_NMS, max_det=300,
                        imgsz=640, classes=[0], verbose=False)[0]
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            out.append({"image_id": im["id"], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return out


def per_image_counts(coco: dict, preds: list[dict]) -> np.ndarray:
    """[tp, n_gt, n_pred] для кожного зображення на робочому порозі."""
    gts: dict[int, list] = {}
    for a in coco["annotations"]:
        gts.setdefault(a["image_id"], []).append(a["bbox"])
    dts: dict[int, list] = {}
    for d in preds:
        if d["score"] >= CONF_OP:
            dts.setdefault(d["image_id"], []).append(d)
    rows = []
    for im in coco["images"]:
        g = np.array(gts.get(im["id"], []), float).reshape(-1, 4)
        ds = sorted(dts.get(im["id"], []), key=lambda d: -d["score"])
        b = np.array([d["bbox"] for d in ds], float).reshape(-1, 4)
        matched = np.zeros(len(g), bool)
        if len(g) and len(b):
            m = iou_matrix(b, g)
            for i in range(len(b)):
                m[i, matched] = 0
                j = int(m[i].argmax())
                if m[i, j] >= IOU_MATCH:
                    matched[j] = True
        rows.append([matched.sum(), len(g), len(b)])
    return np.array(rows, float)


def bootstrap(c: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n, len(c)))
    s = c[idx].sum(1)
    rec, prec = s[:, 0] / np.maximum(s[:, 1], 1), s[:, 0] / np.maximum(s[:, 2], 1)
    return {"recall_ci95": [round(float(x), 4) for x in np.percentile(rec, [2.5, 97.5])],
            "precision_ci95": [round(float(x), 4) for x in np.percentile(prec, [2.5, 97.5])]}


def main() -> None:
    coco = json.loads(TEST.read_text(encoding="utf-8"))
    cache_dir = ROOT / "logs/baselines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, name, src in LADDER:
        cache = cache_dir / f"preds_step{step.replace(chr(39), 'p')}.json"
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            preds, sec = c["preds"], c["sec_per_image"]
        else:
            t0 = time.perf_counter()
            preds = [] if src is None else run_rule(coco) if src == "rule" else run_yolo(coco, src)
            sec = (time.perf_counter() - t0) / len(coco["images"])
            cache.write_text(json.dumps({"sec_per_image": sec, "preds": preds}), encoding="utf-8")
        m = evaluate(coco, TEST, preds) if preds else {"AP50": 0.0, "AP50_95": 0.0, "recall": 0.0, "precision": 0.0}
        m.update(bootstrap(per_image_counts(coco, preds)))
        rows.append({"step": step, "model": name, "AP50": m["AP50"], "AP50_95": m["AP50_95"],
                     "recall": m["recall"], "recall_ci95": m["recall_ci95"], "precision": m["precision"],
                     "precision_ci95": m["precision_ci95"], "sec_per_image": sec})
        print(step, name, {k: rows[-1][k] for k in ("AP50", "recall", "recall_ci95", "precision")}, flush=True)
    q2 = pd.read_csv(ROOT / "reports/tables/q2_summary.csv").set_index("condition")
    for cond, label in (("A", "умова А, середнє 5 сідів, recall з ultralytics val"), ("C", "умова В, середнє 5 сідів, recall з ultralytics val")):
        rows.append({"step": "3*", "model": f"YOLO11n, C2A + АМН на SARD ({label})",
                     "AP50": q2.loc[cond, "mAP50_final_mean"], "recall": q2.loc[cond, "test_recall_final_mean"],
                     "recall_ci95": q2.loc[cond, "test_recall_final_ci95"]})
    out = ROOT / "reports/tables/baseline_ladder.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
