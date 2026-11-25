"""Звичайний запуск YOLO проти SAHI (нарізка на шматки) на C2A val і SARD test.

  python src/exp_sahi.py --dataset c2a --modes plain sahi320 sahi256
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tracking

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "c2a": ("data/processed/c2a/val.json", "data/processed/c2a/images",
            "runs/baseline_c2a/yolo11n_e30/weights/best.pt"),
    "sard": ("data/processed/sard/test.json", "data/processed/sard/images",
             "data/work/al/q2_sard_uncertainty_seed0/runs/r6/weights/last.pt"),
}
CONF_LOW, CONF_OP, IOU_NMS, IOU_MATCH = 0.05, 0.25, 0.6, 0.5
SIZE_BINS = [("lt16", 0, 16), ("16to32", 16, 32), ("ge32", 32, 1e9)]


def predict(mode: str, coco: dict, img_dir: Path, weights: Path, device: str) -> tuple[list[dict], float]:
    if mode == "plain":
        from ultralytics import YOLO
        det = YOLO(str(weights))
    else:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        det = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                 confidence_threshold=CONF_LOW, device=device, image_size=640)
        size = int(mode.removeprefix("sahi"))
    out, t0 = [], time.perf_counter()
    for k, im in enumerate(coco["images"]):
        if k == 1:
            t0 = time.perf_counter()  # перше зображення — прогрів, не рахуємо в час
        path = str(img_dir / im["file_name"])
        if mode == "plain":
            r = det.predict(path, conf=CONF_LOW, iou=IOU_NMS, max_det=300, imgsz=640, device=device, verbose=False)[0]
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append({"image_id": im["id"], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
        else:
            r = get_sliced_prediction(path, det, slice_height=size, slice_width=size,
                                      overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                      postprocess_type="NMS", postprocess_match_metric="IOU",
                                      postprocess_match_threshold=IOU_NMS, verbose=0)
            for p in r.object_prediction_list:
                x1, y1, x2, y2 = p.bbox.to_xyxy()
                out.append({"image_id": im["id"], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(p.score.value)})
    return out, (time.perf_counter() - t0) / max(len(coco["images"]) - 1, 1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0][:, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1][:, None], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter + 1e-9)


def evaluate(coco: dict, gt_path: Path, preds: list[dict]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(preds) if preds else None
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 100, 300]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    p = ev.eval["precision"][0, :, 0, 0, 2]  # IoU 0.5, клас person, усі розміри, 300 детекцій
    res = {"AP50": float(p[p > -1].mean()), "AP50_95": float(ev.stats[0])}

    # recall / precision на робочому порозі, жадібне зіставлення за IoU
    by_img: dict[int, list] = {}
    for d in preds:
        if d["score"] >= CONF_OP:
            by_img.setdefault(d["image_id"], []).append(d)
    gts: dict[int, list] = {}
    for a in coco["annotations"]:
        gts.setdefault(a["image_id"], []).append(a["bbox"])
    tp = n_pred = 0
    hit_sides, all_sides = [], []
    for im in coco["images"]:
        g = np.array(gts.get(im["id"], []), dtype=float).reshape(-1, 4)
        ds = sorted(by_img.get(im["id"], []), key=lambda d: -d["score"])
        b = np.array([d["bbox"] for d in ds], dtype=float).reshape(-1, 4)
        n_pred += len(b)
        sides = np.sqrt(g[:, 2] * g[:, 3])
        all_sides += sides.tolist()
        matched = np.zeros(len(g), bool)
        if len(g) and len(b):
            m = iou_matrix(b, g)
            for i in range(len(b)):
                m[i, matched] = 0
                j = int(m[i].argmax())
                if m[i, j] >= IOU_MATCH:
                    matched[j] = True
        tp += int(matched.sum())
        hit_sides += sides[matched].tolist()
    res["recall"] = tp / max(len(all_sides), 1)
    res["precision"] = tp / max(n_pred, 1)
    hs, al = np.array(hit_sides), np.array(all_sides)
    for name, lo, hi in SIZE_BINS:
        n = ((al >= lo) & (al < hi)).sum()
        res[f"recall_{name}"] = float(((hs >= lo) & (hs < hi)).sum() / n) if n else float("nan")
        res[f"n_{name}"] = int(n)
    return res


def plot(tab: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(tab)
    modes = [m for m in ("plain", "sahi320", "sahi256") if m in set(df["mode"])]
    colors = {"plain": "#2a78d6", "sahi320": "#eb6834", "sahi256": "#1baf7a"}
    names = {"plain": "звичайний запуск", "sahi320": "SAHI, шматки 320", "sahi256": "SAHI, шматки 256"}
    bins = [("lt16", "< 16 px"), ("16to32", "16–32 px"), ("ge32", "≥ 32 px")]
    dsets = [d for d in ("c2a", "sard") if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.2 * len(dsets), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        sub = df[df["dataset"] == ds].set_index("mode")
        x, w = np.arange(len(bins)), 0.8 / len(modes)
        for i, m in enumerate(modes):
            v = [sub.loc[m, f"recall_{b}"] for b, _ in bins]
            ax.bar(x + (i - (len(modes) - 1) / 2) * w, v, w * 0.92, color=colors[m], label=names[m])
        ax.set_xticks(x, [f"{lbl}\n(n={int(sub.iloc[0][f'n_{b}'])})" for b, lbl in bins])
        ax.set_ylim(0, 1)
        ax.set_title({"c2a": "C2A val", "sard": "SARD test"}[ds], fontsize=10, color="#1f1e1d")
        ax.set_ylabel("Recall (поріг 0.25, IoU 0.5)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Recall за розміром людини (сторона рамки)", fontsize=11, color="#1f1e1d")
    fig.tight_layout()
    out = ROOT / "reports/figures"
    for ext in ("png", "svg"):
        fig.savefig(out / f"sahi_recall_by_size.{ext}", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATA)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["plain", "sahi320", "sahi256"])
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.plot:
        return plot(ROOT / "reports/tables/sahi_summary.csv")

    gt_rel, img_rel, w_rel = DATA[a.dataset]
    weights = a.weights or ROOT / w_rel
    coco = json.loads((ROOT / gt_rel).read_text(encoding="utf-8"))
    gt_path = ROOT / gt_rel
    if a.limit:
        coco["images"] = coco["images"][: a.limit]
        keep = {im["id"] for im in coco["images"]}
        coco["annotations"] = [x for x in coco["annotations"] if x["image_id"] in keep]
        gt_path = ROOT / f"data/work/sahi/{a.dataset}_limit{a.limit}.json"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(json.dumps(coco), encoding="utf-8")

    out_dir = ROOT / "logs/sahi"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracking.setup("sahi_compare")
    import mlflow
    rows = []
    for mode in a.modes:
        tag = f"{a.dataset}_{mode}" + (f"_limit{a.limit}" if a.limit else "")
        cache = out_dir / f"preds_{tag}.json"
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            preds, sec = c["preds"], c["sec_per_image"]
        else:
            preds, sec = predict(mode, coco, ROOT / img_rel, weights, a.device)
            cache.write_text(json.dumps({"sec_per_image": sec, "preds": preds}), encoding="utf-8")
        m = evaluate(coco, gt_path, preds)
        m["sec_per_image"] = sec
        rows.append({"dataset": a.dataset, "mode": mode, "n_images": len(coco["images"]), **m})
        print(tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({"dataset": a.dataset, "mode": mode, "weights": str(weights),
                               "conf_op": CONF_OP, "iou_nms": IOU_NMS, "n_images": len(coco["images"])})
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float) and not np.isnan(v)})

    if not a.limit:
        tab = ROOT / "reports/tables/sahi_summary.csv"
        tab.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        if tab.exists():
            old = pd.read_csv(tab)
            df = pd.concat([old[~old.set_index(["dataset", "mode"]).index.isin(
                df.set_index(["dataset", "mode"]).index)], df])
        df.to_csv(tab, index=False)


if __name__ == "__main__":
    main()
