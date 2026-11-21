"""Цикл активного навчання (питання 2): відбір random / uncertainty, оракул, донавчання YOLO.

  python src/al_loop.py --strategy random uncertainty --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml
from ultralytics import YOLO

import tracking
from exp_q1_audit import uncertainty_scores
from oracle import Oracle, iou_xywh

ROOT = Path(__file__).resolve().parents[1]
PREANN_CONF = 0.25
MATCH_IOU = 0.5


def load_split(coco_path: Path, images_root: Path) -> tuple[dict, dict[int, Path]]:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    paths = {im["id"]: images_root / im["file_name"] for im in coco["images"]}
    return coco, paths


def predict(model: YOLO, paths: list[Path], imgsz: int, conf: float = 0.05) -> list[np.ndarray]:
    out = []
    for i in range(0, len(paths), 16):
        for r in model.predict([str(p) for p in paths[i:i + 16]], conf=conf, iou=0.6,
                               max_det=300, imgsz=imgsz, verbose=False):
            xyxy = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()[:, None]
            out.append(np.hstack([xyxy, c]) if len(c) else np.zeros((0, 5)))
    return out


def preann_stats(pred: np.ndarray, truth_xywh: np.ndarray) -> dict:
    """Скільки рамок пре-анотатора правильні / зайві і скільки людей він пропустив."""
    p = pred[pred[:, 4] >= PREANN_CONF] if len(pred) else np.zeros((0, 5))
    pw = p[:, :4].copy()
    if len(pw):
        pw[:, 2:] -= pw[:, :2]
    iou = iou_xywh(pw, truth_xywh) if len(pw) and len(truth_xywh) else np.zeros((len(pw), len(truth_xywh)))
    tp = 0
    while iou.size and iou.max() >= MATCH_IOU:
        i, j = np.unravel_index(iou.argmax(), iou.shape)
        tp += 1
        iou[i, :] = -1
        iou[:, j] = -1
    return {"n_pred": int(len(pw)), "n_pred_correct": tp,
            "n_pred_spurious": int(len(pw) - tp), "n_missed_by_pred": int(len(truth_xywh) - tp)}


def run(strategy: str, seed: int, a) -> None:
    pool_coco, pool_paths = load_split(a.pool_json, a.pool_images)
    _, test_paths = load_split(a.test_json, a.test_images)
    truth_by_img: dict[int, np.ndarray] = {}
    for x in pool_coco["annotations"]:
        truth_by_img.setdefault(x["image_id"], []).append(x["bbox"])
    truth_by_img = {k: np.array(v, dtype=float) for k, v in truth_by_img.items()}

    pool_ids = sorted(pool_paths)
    rng = random.Random(seed)
    if a.pool_limit:
        pool_ids = sorted(rng.sample(pool_ids, a.pool_limit))
    labeled = rng.sample(pool_ids, a.n0)          # однаковий стартовий набір для обох стратегій
    unlabeled = [i for i in pool_ids if i not in set(labeled)]
    sel_rng = random.Random(1000 + seed)

    tag = f"{a.tag}_{strategy}_seed{seed}"
    work = ROOT / "data/work/al" / tag
    work.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs/al"
    oracle_log = log_dir / f"oracle_{tag}.jsonl"
    rounds_log = log_dir / f"rounds_{tag}.jsonl"
    for f in (oracle_log, rounds_log):
        if f.exists():
            f.unlink()
    oracle = Oracle(pool_coco, [], oracle_log, tag)
    for i in labeled:  # стартовий набір теж розмічає оракул (і теж рахується в бюджеті)
        oracle.audit(i, f"{strategy}:init")

    (work / "test.txt").write_text("\n".join(str(test_paths[i]) for i in sorted(test_paths)))
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"strategy": strategy, "seed": seed, "n0": a.n0,
                           "batch_size": a.batch_size, "rounds": a.rounds, "epochs": a.epochs,
                           "init_weights": str(a.init_weights), "pool": str(a.pool_json),
                           "test": str(a.test_json), "pool_size": len(pool_ids),
                           "git_commit": tracking.git_commit(), "tag": a.tag})
        if a.eval_init:
            init_yaml = work / "data_init.yaml"
            init_yaml.write_text(yaml.safe_dump({"path": str(work), "train": str(work / "test.txt"),
                                                 "val": str(work / "test.txt"), "names": {0: "person"}}))
            m0 = YOLO(str(a.init_weights)).val(data=str(init_yaml), split="val", imgsz=a.imgsz,
                                               batch=a.train_batch, max_det=300, plots=False,
                                               verbose=False, project=str(work / "runs"),
                                               name="init_val", exist_ok=True)
            init_metrics = {"init_test_precision": float(m0.box.mp), "init_test_recall": float(m0.box.mr),
                            "init_test_mAP50": float(m0.box.map50), "init_test_mAP50_95": float(m0.box.map)}
            mlflow.log_metrics(init_metrics)
            with open(rounds_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"round": "init", **init_metrics}) + "\n")
            print(tag, init_metrics)
        for r in range(a.rounds + 1):
            t0 = time.time()
            (work / f"train_r{r}.txt").write_text("\n".join(str(pool_paths[i]) for i in sorted(labeled)))
            data_yaml = work / f"data_r{r}.yaml"
            data_yaml.write_text(yaml.safe_dump({
                "path": str(work), "train": str(work / f"train_r{r}.txt"),
                "val": str(work / "test.txt"), "names": {0: "person"}}))
            model = YOLO(str(a.init_weights))
            model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.train_batch,
                        workers=2, seed=seed, deterministic=True, val=False, plots=False,
                        verbose=False, project=str(work / "runs"), name=f"r{r}", exist_ok=True)
            det = YOLO(Path(model.trainer.save_dir) / "weights" / "last.pt")
            m = det.val(data=str(data_yaml), split="val", imgsz=a.imgsz, batch=a.train_batch,
                        max_det=300, plots=False, verbose=False,
                        project=str(work / "runs"), name=f"r{r}_val", exist_ok=True)
            metrics = {"test_precision": float(m.box.mp), "test_recall": float(m.box.mr),
                       "test_mAP50": float(m.box.map50), "test_mAP50_95": float(m.box.map),
                       "n_labeled": len(labeled), "n_queries": oracle.n_queries}
            rec = {"round": r, **metrics, "train_eval_s": time.time() - t0}

            if r < a.rounds and unlabeled:
                k = min(a.batch_size, len(unlabeled))
                if strategy == "uncertainty":
                    preds = predict(det, [pool_paths[i] for i in unlabeled], a.imgsz)
                    pred_by_id = dict(zip(unlabeled, preds))
                    unc = uncertainty_scores(pred_by_id)
                    chosen = sorted(unlabeled, key=lambda i: -unc[i])[:k]
                    chosen_set = set(chosen)
                    rest = [unc[i] for i in unlabeled if i not in chosen_set]
                    rec.update({"selected_unc_mean": float(np.mean([unc[i] for i in chosen])),
                                "unselected_unc_mean": float(np.mean(rest)) if rest else None,
                                "selected_unc_min": float(min(unc[i] for i in chosen)),
                                "unselected_unc_max": float(max(rest)) if rest else None})
                elif strategy == "random":
                    chosen = sel_rng.sample(unlabeled, k)
                    chosen_set = set(chosen)
                    pred_by_id = dict(zip(chosen, predict(det, [pool_paths[i] for i in chosen], a.imgsz)))
                    unc = uncertainty_scores(pred_by_id)
                    rec.update({"selected_unc_mean": float(np.mean([unc[i] for i in chosen]))})
                else:
                    raise ValueError(strategy)
                for i in chosen:
                    oracle.audit(i, strategy, unc[i])
                    st = preann_stats(pred_by_id[i], truth_by_img.get(i, np.zeros((0, 4))))
                    with open(oracle_log.with_name(f"preann_{tag}.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"round": r + 1, "image_id": i, "uncertainty": unc[i],
                                            "n_true": int(len(truth_by_img.get(i, []))), **st}) + "\n")
                labeled = labeled + chosen
                unlabeled = [i for i in unlabeled if i not in chosen_set]

            with open(rounds_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            mlflow.log_metrics({k: v for k, v in rec.items()
                                if isinstance(v, (int, float)) and k != "round"}, step=r)
            print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        for f in log_dir.glob(f"*_{tag}.jsonl"):
            mlflow.log_artifact(str(f), "logs")


def main() -> None:
    c2a = ROOT / "data/processed/c2a"
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=["random", "uncertainty"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pool-json", type=Path, default=c2a / "train.json")
    ap.add_argument("--pool-images", type=Path, default=c2a / "images")
    ap.add_argument("--test-json", type=Path, default=c2a / "val.json")
    ap.add_argument("--test-images", type=Path, default=c2a / "images")
    ap.add_argument("--init-weights", type=Path, default=Path("yolo11n.pt"))
    ap.add_argument("--n0", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=50, help="зображень на раунд (бюджет запитів)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--pool-limit", type=int, default=0)
    ap.add_argument("--eval-init", action="store_true",
                    help="оцінити стартові ваги на тесті до будь-якого донавчання")
    ap.add_argument("--tag", default="c2a")
    ap.add_argument("--experiment", default="al_loop")
    a = ap.parse_args()
    tracking.setup(a.experiment)
    for seed in a.seeds:
        for s in a.strategy:
            run(s, seed, a)


if __name__ == "__main__":
    main()
