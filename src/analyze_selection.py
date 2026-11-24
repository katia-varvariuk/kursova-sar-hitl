"""Які зображення обирає АМН порівняно з випадковим відбором (питання 2).

  python src/analyze_selection.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG, SEEDS = "q2_sard", (0, 1, 2, 3, 4)
TAB, FIG = ROOT / "reports/tables", ROOT / "reports/figures"
COLORS = {"random": "#2a78d6", "uncertainty": "#eb6834"}
LABELS = {"random": "випадковий відбір (умови А, Б)", "uncertainty": "АМН: невизначеність (умова В)"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    pool = json.loads((ROOT / "data/processed/sard/pool.json").read_text(encoding="utf-8"))
    meta = {im["id"]: im for im in pool["images"]}
    sides = defaultdict(list)
    for a in pool["annotations"]:
        sides[a["image_id"]].append(np.sqrt(a["bbox"][2] * a["bbox"][3]))
    pool_people = np.array([len(sides.get(i, [])) for i in meta])

    rows, curves = [], defaultdict(list)
    for st in ("random", "uncertainty"):
        for s in SEEDS:
            pre = jl(ROOT / f"logs/al/preann_{TAG}_{st}_seed{s}.jsonl")
            ids = [p["image_id"] for p in pre]
            n_true = np.array([p["n_true"] for p in pre])
            frames = [meta[i]["frame_id"] for i in ids]
            fr_count = defaultdict(int)
            for f in frames:
                fr_count[f] += 1
            fr_set = set(frames)
            near = [any((g in fr_set) for g in range(f - 2, f + 3) if g != f) or fr_count[f] > 1
                    for f in frames]
            unc = np.array([p["uncertainty"] for p in pre])
            n_pred = sum(p["n_pred"] for p in pre)
            box = [x for i in ids for x in sides.get(i, [])]
            rows.append({
                "strategy": st, "seed": s, "selected": len(ids),
                "persons_per_tile": n_true.mean(), "persons_total": int(n_true.sum()),
                "share_empty_tiles": float((n_true == 0).mean()),
                "box_side_median_px": float(np.median(box)) if box else None,
                "uncertainty_mean": float(unc.mean()),
                "share_uncertainty_saturated": float((unc > 0.999).mean()),
                "distinct_frames": len(fr_set), "distinct_scenes": len({meta[i]["scene"] for i in ids}),
                "share_same_or_adjacent_frame": float(np.mean(near)),
                "preann_precision": sum(p["n_pred_correct"] for p in pre) / n_pred if n_pred else None,
                "preann_recall": (sum(p["n_pred_correct"] for p in pre) / n_true.sum()) if n_true.sum() else None,
            })
            if st == "random":
                miss = np.array([p["n_missed_by_pred"] for p in pre])
                rows[-1].update({
                    "pool_share_tiles_unc0": float((unc == 0).mean()),
                    "pool_share_model_misses_in_unc0_tiles": float(miss[unc == 0].sum() / max(1, miss.sum())),
                    "pool_share_model_misses_in_saturated_tiles": float(miss[unc > 0.999].sum() / max(1, miss.sum())),
                })
            by_round = defaultdict(int)
            for p in pre:
                by_round[p["round"]] += p["n_true"]
            init = sum(r["n_true_boxes"] for r in jl(ROOT / f"logs/al/oracle_{TAG}_{st}_seed{s}.jsonl")
                       if r["strategy"].endswith(":init"))
            cum = np.cumsum([init] + [by_round[r] for r in sorted(by_round)])
            curves[st].append(cum)

    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "q2_selection_by_seed.csv", index=False)
    summ = df.drop(columns="seed").groupby("strategy").agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.loc["pool_reference", "persons_per_tile_mean"] = pool_people.mean()
    summ.loc["pool_reference", "share_empty_tiles_mean"] = float((pool_people == 0).mean())
    summ.to_csv(TAB / "q2_selection_summary.csv")
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].T.round(3).to_string())

    # рисунок: скільки людей (рамок для навчання) дає бюджет запитів
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    q = np.arange(len(curves["random"][0])) * 50 + 50
    for st, marker in (("random", "o"), ("uncertainty", "s")):
        c = np.array(curves[st])
        ax.fill_between(q, c.min(0), c.max(0), color=COLORS[st], alpha=0.15, linewidth=0)
        ax.plot(q, c.mean(0), color=COLORS[st], marker=marker, markersize=7, linewidth=2, label=LABELS[st])
    ax.set_xlabel("Людських запитів (зображень SARD)")
    ax.set_ylabel("Розмічено людей (рамок для навчання)")
    ax.set_title(f"Що дає бюджет запитів: людей у відібраних тайлах (смуга — min–max за {len(SEEDS)} сідами)",
                 fontsize=10, color="#1f1e1d")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"q2_selection_persons.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
