"""Підбір порогів маршрутизації на калібрувальній половині й оцінка на іншій.

  python src/exp_routing.py --dataset c2a --q1-tag main
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np

import tracking
from oracle import NoiseConfig, make_noisy
from routing import RoutingConfig, evaluate, grid, route_all

ROOT = Path(__file__).resolve().parents[1]
GRID = {"conf_accept": [0.3, 0.4, 0.5, 0.6, 0.7], "iou_accept": [0.5, 0.6, 0.7],
        "conf_extra": [0.3, 0.5, 0.7], "conf_reject": [0.05, 0.1, 0.2, 0.3]}


def labels_by_img(coco: dict) -> dict[int, tuple[list[int], np.ndarray]]:
    ids: dict[int, list] = {im["id"]: [] for im in coco["images"]}
    boxes: dict[int, list] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        ids[a["image_id"]].append(a["id"])
        boxes[a["image_id"]].append(a["bbox"])
    return {k: (ids[k], np.array(boxes[k], dtype=float).reshape(-1, 4)) for k in ids}


def load_c2a(tag: str, seed: int):
    work = ROOT / "data/work/q1" / f"{tag}_seed{seed}"
    noisy = json.loads((work / "noisy.json").read_text())
    errors = json.loads((work / "errors.json").read_text())
    raw = json.loads((work / "oos_predictions.json").read_text())
    preds = {int(k): np.array(v, dtype=float).reshape(-1, 5) for k, v in raw.items()}
    truth = json.loads((ROOT / "data/processed/c2a/train.json").read_text(encoding="utf-8"))
    return noisy, errors, preds, truth


def load_sard(weights: Path, seed: int, cache_dir: Path):
    truth = json.loads((ROOT / "data/processed/sard/test.json").read_text(encoding="utf-8"))
    noisy, errors = make_noisy(truth, NoiseConfig(seed=seed))
    cache = cache_dir / f"sard_test_preds_{weights.parent.parent.name}.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        from ultralytics import YOLO
        det = YOLO(str(weights))
        raw = {}
        for im in truth["images"]:
            r = det.predict(str(ROOT / "data/processed/sard/images" / im["file_name"]),
                            conf=0.05, iou=0.6, max_det=300, verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()[:, None]
            raw[str(im["id"])] = (np.hstack([xyxy, c]) if len(c) else np.zeros((0, 5))).tolist()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    preds = {int(k): np.array(v, dtype=float).reshape(-1, 5) for k, v in raw.items()}
    return noisy, errors, preds, truth


def subset(ids: set[int], errors: list[dict]) -> list[dict]:
    return [e for e in errors if e["image_id"] in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["c2a", "sard"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--q1-tag", default="main")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/baseline_c2a/yolo11n_e30/weights/best.pt")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    tracking.setup("routing")
    out_dir = ROOT / "logs/routing"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in a.seeds:
        if a.dataset == "c2a":
            noisy, errors, preds, truth = load_c2a(a.q1_tag, seed)
        else:
            noisy, errors, preds, truth = load_sard(a.weights, seed, ROOT / "data/work/routing")
        labels = labels_by_img(noisy)
        truth_boxes = {k: v[1] for k, v in labels_by_img(truth).items()}
        ids = sorted(preds)
        rng = random.Random(500 + seed)
        rng.shuffle(ids)
        calib, evalu = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])

        rows = []
        for cfg in grid(GRID):
            r = route_all({i: labels[i] for i in calib}, {i: preds[i] for i in calib}, cfg)
            m = evaluate(r, subset(calib, errors), truth_boxes, labels)
            rows.append({**asdict(cfg), **m})
        feasible = [r for r in rows if r["errors_to_human_frac"] >= a.target]
        best = min(feasible, key=lambda r: r["human_items"]) if feasible else \
            max(rows, key=lambda r: r["errors_to_human_frac"])
        cfg = RoutingConfig(**{k: best[k] for k in asdict(RoutingConfig())})
        routes = route_all({i: labels[i] for i in evalu}, {i: preds[i] for i in evalu}, cfg)
        m_eval = evaluate(routes, subset(evalu, errors), truth_boxes, labels)

        tag = f"{a.tag}_{a.dataset}_seed{seed}"
        (out_dir / f"eval_{tag}.json").write_text(json.dumps({
            "dataset": a.dataset, "seed": seed, "target": a.target, "feasible": bool(feasible),
            "config": asdict(cfg), "calib": {k: v for k, v in best.items() if k not in asdict(cfg)},
            "eval": m_eval}, indent=2))
        (out_dir / f"grid_{tag}.json").write_text(json.dumps(rows))
        (out_dir / f"routes_{tag}.json").write_text(json.dumps({
            str(k): {"auto_reject": v.auto_reject, "n_human_items": v.n_human_items,
                     "label_decision": dict(zip(map(str, v.label_ids), v.label_decision))}
            for k, v in routes.items()}))
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({"dataset": a.dataset, "seed": seed, "target": a.target,
                               "target_feasible_on_calib": bool(feasible),
                               "weights": str(a.weights) if a.dataset == "sard" else f"q1:{a.q1_tag}",
                               "git_commit": tracking.git_commit(),
                               **{f"chosen_{k}": v for k, v in asdict(cfg).items()}})
            mlflow.log_metrics({f"calib_{k}": float(v) for k, v in best.items()
                                if k not in asdict(cfg)})
            mlflow.log_metrics({f"eval_{k}": float(v) for k, v in m_eval.items()})
            mlflow.log_artifact(str(out_dir / f"grid_{tag}.json"), "grid")
            mlflow.log_artifact(str(out_dir / f"routes_{tag}.json"), "routes")
        print(tag, "cfg", asdict(cfg), "| eval", {k: (round(v, 3) if isinstance(v, float) else v)
                                                  for k, v in m_eval.items()})


if __name__ == "__main__":
    main()
