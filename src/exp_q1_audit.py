"""Питання 1: скільки перевірок потрібно, щоб знайти помилки розмітки (random / uncertainty / cleanlab).

  python src/exp_q1_audit.py --seeds 0 1 2 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from cleanlab.object_detection.rank import get_label_quality_scores
from ultralytics import YOLO

import tracking
from oracle import NoiseConfig, Oracle, make_noisy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/c2a"
STRATEGIES = ("random", "uncertainty", "cleanlab")
TARGETS = (0.5, 0.8, 0.9)


# ---------- підготовка даних ----------

def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_dataset(coco: dict, work: Path) -> dict[int, Path]:
    """Жорсткі посилання на зображення + YOLO-мітки з (зашумленого) COCO."""
    img_dir, lbl_dir = work / "images" / "train", work / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    paths = {}
    for im in coco["images"]:
        src = DATA / "images" / im["file_name"]
        dst = img_dir / src.name
        link_or_copy(src, dst)
        W, H = im["width"], im["height"]
        lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                 for x, y, w, h in by_img.get(im["id"], [])]
        (lbl_dir / (dst.stem + ".txt")).write_text("\n".join(lines))
        paths[im["id"]] = dst
    return paths


# ---------- out-of-sample передбачення ----------

def kfold_predictions(paths: dict[int, Path], work: Path, k: int, epochs: int,
                      seed: int, imgsz: int, batch: int) -> dict[int, np.ndarray]:
    cache = work / "oos_predictions.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {int(i): np.array(v, dtype=float).reshape(-1, 5) for i, v in raw.items()}

    ids = sorted(paths)
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = [ids[i::k] for i in range(k)]
    preds: dict[int, np.ndarray] = {}
    for f, held in enumerate(folds):
        held_set = set(held)
        train_ids = [i for i in ids if i not in held_set]
        (work / f"fold{f}_train.txt").write_text("\n".join(str(paths[i]) for i in train_ids))
        (work / f"fold{f}_held.txt").write_text("\n".join(str(paths[i]) for i in held))
        data_yaml = work / f"fold{f}.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "path": str(work), "train": str(work / f"fold{f}_train.txt"),
            "val": str(work / f"fold{f}_held.txt"), "names": {0: "person"}}))
        fold_dir = work / "runs" / f"fold{f}"
        last = fold_dir / "weights" / "last.pt"
        res_csv = fold_dir / "results.csv"
        done_epochs = len(pd.read_csv(res_csv)) if res_csv.exists() else 0
        if done_epochs >= epochs and last.exists():
            print(f"fold{f}: вже навчено ({done_epochs} епох) — використовую {last}")
        elif last.exists() and done_epochs > 0:
            print(f"fold{f}: продовжую перерване навчання з епохи {done_epochs + 1}")
            YOLO(str(last)).train(resume=True)
        else:
            YOLO("yolo11n.pt").train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                                     workers=2, seed=seed, deterministic=True, cache=False,
                                     val=False, plots=False, verbose=False,
                                     project=str(work / "runs"), name=f"fold{f}", exist_ok=True)
        # val=False -> best.pt не обирається за метрикою, беремо останню епоху
        det = YOLO(str(last))
        for i in held:
            r = det.predict(str(paths[i]), conf=0.05, iou=0.6, max_det=300, imgsz=imgsz,
                            verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()[:, None]
            preds[i] = np.hstack([xyxy, conf]) if len(conf) else np.zeros((0, 5))
    cache.write_text(json.dumps({str(i): p.tolist() for i, p in preds.items()}))
    return preds


# ---------- оцінки для порядку аудиту ----------

def uncertainty_scores(preds: dict[int, np.ndarray], min_conf: float = 0.1) -> dict[int, float]:
    out = {}
    for i, p in preds.items():
        c = p[:, 4][p[:, 4] >= min_conf] if len(p) else np.array([])
        if len(c) == 0:
            out[i] = 0.0
            continue
        c = np.clip(c, 1e-6, 1 - 1e-6)
        out[i] = float(np.mean(-(c * np.log2(c) + (1 - c) * np.log2(1 - c))))
    return out


def cleanlab_scores(noisy: dict, preds: dict[int, np.ndarray]) -> dict[int, float]:
    by_img: dict[int, list] = {}
    for a in noisy["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    ids = sorted(preds)
    labels, predictions = [], []
    for i in ids:
        b = np.array(by_img.get(i, []), dtype=float).reshape(-1, 4)
        b[:, 2:] += b[:, :2]  # xywh -> xyxy
        labels.append({"bboxes": b, "labels": np.zeros(len(b), dtype=int)})
        per_class = np.empty(1, dtype=object)
        per_class[0] = preds[i].astype(np.float32)
        predictions.append(per_class)
    s = get_label_quality_scores(labels, predictions, verbose=False)
    return {i: float(v) for i, v in zip(ids, s)}


def audit_order(strategy: str, ids: list[int], unc: dict, cl: dict, seed: int) -> list[tuple[int, float | None]]:
    if strategy == "random":
        order = ids[:]
        random.Random(10_000 + seed).shuffle(order)
        return [(i, None) for i in order]
    if strategy == "uncertainty":
        return [(i, unc[i]) for i in sorted(ids, key=lambda i: -unc[i])]
    if strategy == "cleanlab":
        return [(i, cl[i]) for i in sorted(ids, key=lambda i: cl[i])]
    raise ValueError(strategy)


# ---------- метрики з журналу ----------

def curve_from_log(log_path: Path, run_tag: str, strategy: str, n_errors: int) -> pd.DataFrame:
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["run"] == run_tag and r["strategy"] == strategy]
    rows.sort(key=lambda r: r["query_idx"])
    seen: set[int] = set()
    out = []
    for k, r in enumerate(rows, start=1):
        seen.update(r["error_ids"])
        out.append({"checks": k, "found": len(seen), "found_frac": len(seen) / n_errors})
    return pd.DataFrame(out)


def run_seed(seed: int, args) -> None:
    truth = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    if args.limit:
        keep = {im["id"] for im in sorted(truth["images"], key=lambda x: x["id"])[:args.limit]}
        truth = {**truth, "images": [im for im in truth["images"] if im["id"] in keep],
                 "annotations": [a for a in truth["annotations"] if a["image_id"] in keep]}
    cfg = NoiseConfig(seed=seed)
    noisy, errors = make_noisy(truth, cfg)

    work = ROOT / "data/work/q1" / f"{args.tag}_seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "noisy.json").write_text(json.dumps(noisy))
    (work / "errors.json").write_text(json.dumps(errors))
    paths = write_yolo_dataset(noisy, work)

    preds = kfold_predictions(paths, work, args.folds, args.epochs, seed, args.imgsz, args.batch)
    unc = uncertainty_scores(preds)
    cl = cleanlab_scores(noisy, preds)
    ids = sorted(paths)

    log_path = ROOT / "logs/q1" / f"{args.tag}_seed{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    run_tag = f"{args.tag}_seed{seed}"
    for strategy in STRATEGIES:
        oracle = Oracle(truth, errors, log_path, run_tag)
        for i, score in audit_order(strategy, ids, unc, cl, seed):
            oracle.audit(i, strategy, score)

        curve = curve_from_log(log_path, run_tag, strategy, len(errors))
        curve_path = work / f"curve_{strategy}.csv"
        curve.to_csv(curve_path, index=False)
        with mlflow.start_run(run_name=f"{run_tag}_{strategy}"):
            mlflow.log_params({"seed": seed, "strategy": strategy, "tag": args.tag,
                               "epochs": args.epochs, "folds": args.folds,
                               "pool_images": len(ids), "n_errors": len(errors),
                               "git_commit": tracking.git_commit(),
                               **{f"noise_{k}": v for k, v in asdict(cfg).items()}})
            step = max(1, len(curve) // 200)
            for _, r in curve.iloc[::step].iterrows():
                mlflow.log_metric("found_frac", r["found_frac"], step=int(r["checks"]))
            summary = {}
            for t in TARGETS:
                hit = curve[curve["found_frac"] >= t]
                summary[f"checks_to_{int(t * 100)}pct"] = int(hit["checks"].iloc[0]) if len(hit) else len(curve)
            for frac in (0.1, 0.2, 0.3, 0.5):
                k = max(1, round(frac * len(curve)))
                summary[f"found_at_{int(frac * 100)}pct_budget"] = float(curve["found_frac"].iloc[k - 1])
            mlflow.log_metrics(summary)
            mlflow.log_artifact(str(curve_path), "curves")
            mlflow.log_artifact(str(log_path), "oracle_log")
        print(seed, strategy, summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="обмежити пул (для пробного запуску)")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    tracking.setup("q1_error_audit")
    for s in args.seeds:
        run_seed(s, args)


if __name__ == "__main__":
    main()
