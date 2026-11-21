"""Модель людського часу за lead_time Label Studio (Huber-регресія, бутстреп-ДІ).

  python src/cost_model.py --summary logs/labelstudio/project1_summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["n_kept", "n_added", "n_removed"]


def design(df: pd.DataFrame, datasets: list[str]) -> np.ndarray:
    cols = [(df["dataset"] == d).astype(float).to_numpy() for d in datasets]
    cols += [df[f].astype(float).to_numpy() for f in FEATURES]
    return np.column_stack(cols)


def huber_nonneg(X: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
    def loss(w):
        r = y - X @ w
        a = np.abs(r)
        return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta)).sum()
    w0 = np.linalg.lstsq(X, y, rcond=None)[0].clip(0.1)
    res = minimize(loss, w0, method="L-BFGS-B", bounds=[(0, None)] * X.shape[1])
    return res.x


def fit(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    df = df.dropna(subset=["lead_time_s"]).copy()
    datasets = sorted(df["dataset"].unique())
    X, y = design(df, datasets), df["lead_time_s"].to_numpy(float)
    delta = 1.345 * np.median(np.abs(y - np.median(y))) / 0.6745
    w = huber_nonneg(X, y, delta)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        b = df.iloc[idx]
        if b["dataset"].nunique() < len(datasets):
            continue
        boots.append(huber_nonneg(design(b, datasets), b["lead_time_s"].to_numpy(float), delta))
    boots = np.array(boots)
    names = [f"t0_{d}" for d in datasets] + [f"b_{f[2:]}" for f in FEATURES]
    pred = X @ w
    return {
        "coef": {n: float(v) for n, v in zip(names, w)},
        "ci95": {n: [float(np.percentile(boots[:, k], 2.5)), float(np.percentile(boots[:, k], 97.5))]
                 for k, n in enumerate(names)},
        "n_tasks": int(len(df)), "n_bootstrap": int(len(boots)), "huber_delta_s": float(delta),
        "median_abs_error_s": float(np.median(np.abs(y - pred))),
        "tasks_per_dataset_mode": {f"{k[0]}/{k[1]}": int(v)
                                   for k, v in df.groupby(["dataset", "mode"]).size().items()},
        "observed_median_lead_time_s": {f"{k[0]}/{k[1]}": float(v)
                                        for k, v in df.groupby(["dataset", "mode"])["lead_time_s"].median().items()},
    }


def load(path: Path = ROOT / "configs/cost_model.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seconds(model: dict, dataset: str, *, manual: bool, n_true: int = 0, n_correct: int = 0,
            n_missed: int = 0, n_spurious: int = 0, coef: dict | None = None) -> float:
    c = coef or model["coef"]
    t0 = c[f"t0_{dataset}"]
    if manual:
        return t0 + c["b_added"] * n_true
    return t0 + c["b_kept"] * n_correct + c["b_added"] * n_missed + c["b_removed"] * n_spurious


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, nargs="+",
                    default=[ROOT / "logs/labelstudio/project1_summary.csv"])
    ap.add_argument("--out", type=Path, default=ROOT / "configs/cost_model.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.summary], ignore_index=True)
    m = fit(df, a.n_boot, a.seed)
    m["source_logs"] = [str(p.relative_to(ROOT)) for p in a.summary]
    a.out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
