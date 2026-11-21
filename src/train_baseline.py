"""Базова модель YOLO на C2A з логуванням у MLflow.

  python src/train_baseline.py --epochs 30 --name yolo11n_e30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from ultralytics import YOLO

import tracking

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/c2a/c2a.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--experiment", default="baseline_c2a")
    ap.add_argument("--resume", action="store_true",
                    help="продовжити перерване навчання з runs/<experiment>/<name>/weights/last.pt")
    args = ap.parse_args()

    tracking.setup(args.experiment)
    stats_file = args.data.parent / "stats.json"
    with mlflow.start_run(run_name=args.name):
        mlflow.log_params({**{k: str(v) for k, v in vars(args).items()},
                           "git_commit": tracking.git_commit(),
                           "dataset_stats_md5": tracking.file_md5(stats_file),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
        mlflow.log_artifact(str(stats_file), "dataset")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if args.resume:
            last = ROOT / "runs" / args.experiment / args.name / "weights" / "last.pt"
            model = YOLO(last)
            model.train(resume=True)
            mlflow.set_tag("resumed", "true")
        else:
            model = YOLO(args.model)
            model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch, workers=args.workers, seed=args.seed,
                        deterministic=True, cache="ram", max_det=300,
                        project=str(ROOT / "runs" / args.experiment), name=args.name,
                        exist_ok=True, plots=True, verbose=False)
        train_s = time.time() - t0
        save_dir = Path(model.trainer.save_dir)

        res = pd.read_csv(save_dir / "results.csv")
        res.columns = [c.strip() for c in res.columns]
        for _, row in res.iterrows():
            step = int(row["epoch"])
            mlflow.log_metrics({k.replace("(B)", "").replace("/", "_").replace("-", "_"): float(v)
                                for k, v in row.items() if k != "epoch"}, step=step)

        best = YOLO(save_dir / "weights" / "best.pt")
        m = best.val(data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch,
                     max_det=300, plots=False, verbose=False,
                     project=str(ROOT / "runs" / args.experiment), name=f"{args.name}_val",
                     exist_ok=True)
        final = {"val_precision": float(m.box.mp), "val_recall": float(m.box.mr),
                 "val_mAP50": float(m.box.map50), "val_mAP50_95": float(m.box.map),
                 "train_seconds_this_process": train_s,
                 "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
        if "time" in res.columns and len(res) > 1:
            final["seconds_per_epoch"] = float(res["time"].diff().iloc[1:].median())
        mlflow.log_metrics(final)
        mlflow.log_artifact(str(save_dir / "weights" / "best.pt"), "weights")
        mlflow.log_artifact(str(save_dir / "results.csv"), "train")
        for png in save_dir.glob("*.png"):
            mlflow.log_artifact(str(png), "plots")
        print({k: round(v, 4) for k, v in final.items()})


if __name__ == "__main__":
    main()
