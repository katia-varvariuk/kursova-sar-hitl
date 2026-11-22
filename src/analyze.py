"""Аналіз результатів: бутстреп-ДІ, графіки й таблиці для питань 1-2, маршрутизації та економії.

  python src/analyze.py --q1-tag main --q2-tag q2_sard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import cost_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
TAB = ROOT / "reports/tables"

COLORS = {"random": "#2a78d6", "uncertainty": "#eb6834", "cleanlab": "#1baf7a",
          "A": "#2a78d6", "B": "#1baf7a", "C": "#eb6834"}
MARKERS = {"random": "o", "uncertainty": "s", "cleanlab": "^", "A": "o", "B": "^", "C": "s"}
LABELS_Q1 = {"random": "випадковий аудит", "uncertainty": "АМН: невизначеність", "cleanlab": "Cleanlab"}
LABELS_Q2 = {"A": "умова А: випадковий відбір, ручна розмітка",
             "B": "умова Б: випадковий відбір + пре-анотація",
             "C": "умова В: АМН (невизначеність) + пре-анотація"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor="#52514e")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def wilcoxon_safe(x, y) -> float | None:
    d = np.asarray(x) - np.asarray(y)
    if len(d) < 2 or np.allclose(d, 0):
        return None
    return float(wilcoxon(x, y).pvalue)


# ---------------- питання 1 ----------------

def q1_load(tag: str, seeds: list[int]):
    data = {}
    for s in seeds:
        log = ROOT / "logs/q1" / f"{tag}_seed{s}.jsonl"
        err = ROOT / "data/work/q1" / f"{tag}_seed{s}" / "errors.json"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        n_err = len(json.loads(err.read_text()))
        per = {}
        for st in ("random", "uncertainty", "cleanlab"):
            r = sorted((x for x in rows if x["strategy"] == st), key=lambda x: x["query_idx"])
            per[st] = {"order": [x["image_id"] for x in r],
                       "found": np.array([x["n_errors_found"] for x in r], dtype=float)}
        data[s] = {"n_errors": n_err, "per": per}
    return data


def q1_curve(found_in_order: np.ndarray, n_err: float, grid: np.ndarray) -> np.ndarray:
    cum = np.concatenate([[0], np.cumsum(found_in_order)]) / n_err
    n = len(found_in_order)
    return np.interp(grid * n, np.arange(n + 1), cum)


def q1_analysis(tag: str, seeds: list[int], n_boot: int, rng) -> dict:
    data = q1_load(tag, seeds)
    if not data:
        return {}
    grid = np.linspace(0, 1, 101)
    out, curves = {}, {}
    for st in ("random", "uncertainty", "cleanlab"):
        per_seed = np.array([q1_curve(d["per"][st]["found"], d["n_errors"], grid) for d in data.values()])
        boots = []
        seeds_l = list(data.values())
        for _ in range(n_boot):
            d = seeds_l[rng.integers(len(seeds_l))]
            f = d["per"][st]["found"]
            w = rng.multinomial(len(f), np.ones(len(f)) / len(f)).astype(float)
            fw = f * w
            cum = np.concatenate([[0], np.cumsum(fw)]) / max(fw.sum(), 1e-9)
            x = np.concatenate([[0], np.cumsum(w)]) / w.sum()
            boots.append(np.interp(grid, x, cum))
        boots = np.array(boots)
        curves[st] = (per_seed.mean(0), np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0))

        def checks_to(t):
            v = []
            for c in per_seed:
                hit = np.nonzero(c >= t)[0]
                v.append(grid[hit[0]] if len(hit) else 1.0)
            return v
        out[st] = {"share_checked_to_50pct": checks_to(0.5), "share_checked_to_80pct": checks_to(0.8),
                   "share_checked_to_90pct": checks_to(0.9),
                   "found_at_20pct_budget": per_seed[:, 20].tolist(),
                   "found_at_30pct_budget": per_seed[:, 30].tolist(),
                   "found_at_20pct_ci95": [float(np.percentile(boots[:, 20], 2.5)),
                                           float(np.percentile(boots[:, 20], 97.5))]}
    for st in ("uncertainty", "cleanlab"):
        out[f"wilcoxon_{st}_vs_random_found_at_20pct"] = wilcoxon_safe(
            out[st]["found_at_20pct_budget"], out["random"]["found_at_20pct_budget"])
    out["seeds"] = list(data)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    for st, (m, lo, hi) in curves.items():
        ax.fill_between(grid * 100, lo * 100, hi * 100, color=COLORS[st], alpha=0.15, linewidth=0)
        ax.plot(grid * 100, m * 100, color=COLORS[st], linewidth=2, label=LABELS_Q1[st],
                marker=MARKERS[st], markevery=10, markersize=6)
        k = {"cleanlab": 22, "random": 50, "uncertainty": 72}[st]   # рознесені підписи
        ax.annotate(LABELS_Q1[st], (grid[k] * 100, m[k] * 100), xytext=(6, -12),
                    textcoords="offset points", color="#52514e", fontsize=8)
    ax.set_xlabel("Перевірено зображень, % пулу")
    ax.set_ylabel("Знайдено внесених помилок, %")
    ax.set_title(f"Питання 1: аудит розмітки C2A (сіди: {len(data)}, смуга — 95 % ДІ)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "q1_error_audit_curve")

    rows = []
    for st in ("random", "uncertainty", "cleanlab"):
        for k in ("share_checked_to_50pct", "share_checked_to_80pct", "share_checked_to_90pct",
                  "found_at_20pct_budget", "found_at_30pct_budget"):
            v = np.array(out[st][k])
            rows.append({"strategy": st, "metric": k, "mean": v.mean(), "min": v.min(), "max": v.max(),
                         "per_seed": json.dumps([round(float(x), 4) for x in v])})
    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TAB / "q1_summary.csv", index=False)
    return out


# ---------------- питання 2 ----------------

def q2_load(tag: str, seeds: list[int]):
    res = {}
    for st in ("random", "uncertainty"):
        for s in seeds:
            base = f"{tag}_{st}_seed{s}"
            rl = ROOT / "logs/al" / f"rounds_{base}.jsonl"
            if not rl.exists():
                continue
            rounds = [json.loads(l) for l in rl.read_text(encoding="utf-8").splitlines()]
            if not rounds or "n_queries" not in rounds[-1] or "selected_unc_mean" in rounds[-1]:
                print(f"[q2] пропускаю неповний прогін {base}")
                continue
            pre = ROOT / "logs/al" / f"preann_{base}.jsonl"
            pre_rows = [json.loads(l) for l in pre.read_text(encoding="utf-8").splitlines()] if pre.exists() else []
            orc = ROOT / "logs/al" / f"oracle_{base}.jsonl"
            orc_rows = [json.loads(l) for l in orc.read_text(encoding="utf-8").splitlines()]
            res[(st, s)] = {"rounds": rounds, "preann": pre_rows, "oracle": orc_rows}
    return res


def q2_costs(run: dict, model: dict, dataset: str) -> dict[str, list[float]]:
    """Кумулятивні людські години після кожного раунду для умов А/Б/В."""
    init = [r for r in run["oracle"] if r["strategy"].endswith(":init")]
    init_s = sum(cost_model.seconds(model, dataset, manual=True, n_true=r["n_true_boxes"]) for r in init)
    by_round: dict[int, list[dict]] = {}
    for p in run["preann"]:
        by_round.setdefault(p["round"], []).append(p)
    rounds = [r for r in run["rounds"] if r["round"] != "init"]
    manual, pre = [init_s], [init_s]
    for r in rounds[1:]:
        items = by_round.get(r["round"], [])
        manual.append(manual[-1] + sum(cost_model.seconds(model, dataset, manual=True, n_true=p["n_true"])
                                       for p in items))
        pre.append(pre[-1] + sum(cost_model.seconds(model, dataset, manual=False,
                                                    n_correct=p["n_pred_correct"],
                                                    n_missed=p["n_missed_by_pred"],
                                                    n_spurious=p["n_pred_spurious"]) for p in items))
    return {"manual_h": [x / 3600 for x in manual], "preann_h": [x / 3600 for x in pre]}


def q2_analysis(tag: str, seeds: list[int], dataset: str, n_boot: int, rng) -> dict:
    runs = q2_load(tag, seeds)
    if not runs:
        return {}
    model = cost_model.load()
    metric = "test_recall"
    cond = {"A": ("random", "manual_h"), "B": ("random", "preann_h"), "C": ("uncertainty", "preann_h")}
    table, curves = [], {}
    init_vals = [r["rounds"][0][f"init_{metric}"] for r in runs.values() if r["rounds"][0]["round"] == "init"]
    for c, (st, cost_key) in cond.items():
        if not any((st, s) in runs for s in seeds):
            print(f"[q2] немає повних прогонів для {st}")
            return {}
        ys, xs, qs = [], [], []
        for s in seeds:
            if (st, s) not in runs:
                continue
            run = runs[(st, s)]
            rounds = [r for r in run["rounds"] if r["round"] != "init"]
            ys.append([r[metric] for r in rounds])
            qs.append([r["n_queries"] for r in rounds])
            xs.append(q2_costs(run, model, dataset)[cost_key])
        ys, xs, qs = np.array(ys), np.array(xs), np.array(qs)
        boot = np.array([ys[rng.integers(len(ys), size=len(ys))].mean(0) for _ in range(n_boot)])
        curves[c] = {"y": ys.mean(0), "lo": np.percentile(boot, 2.5, 0), "hi": np.percentile(boot, 97.5, 0),
                     "x_h": xs.mean(0), "q": qs.mean(0), "per_seed_final": ys[:, -1].tolist(),
                     "per_seed_hours": xs[:, -1].tolist(), "ys": ys, "xs": xs}
        table.append({"condition": c, "selection": st, "queries_final": float(qs[:, -1].mean()),
                      "human_hours_final_mean": float(xs[:, -1].mean()),
                      f"{metric}_final_mean": float(ys[:, -1].mean()),
                      f"{metric}_final_ci95": [float(curves[c]["lo"][-1]), float(curves[c]["hi"][-1])],
                      "per_seed": [round(float(v), 4) for v in ys[:, -1]],
                      "mAP50_final_mean": float(np.mean([[r["test_mAP50"] for r in runs[(st, s)]["rounds"]
                                                          if r["round"] != "init"][-1]
                                                         for s in seeds if (st, s) in runs]))})
    out = {"table": table,
           "init_recall_c2a_only": float(np.mean(init_vals)) if init_vals else None,
           "wilcoxon_C_vs_A_final_recall": wilcoxon_safe(curves["C"]["per_seed_final"],
                                                         curves["A"]["per_seed_final"]),
           "seeds": seeds,
           "economy": q2_economy(curves, rng, n_boot)}

    # (а) якість від кількості запитів — А і Б тут збігаються (той самий відбір)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    for c, lab in (("A", "випадковий відбір (умови А, Б)"), ("C", "АМН: невизначеність (умова В)")):
        cv = curves[c]
        ax.fill_between(cv["q"], cv["lo"], cv["hi"], color=COLORS[c], alpha=0.15, linewidth=0)
        ax.plot(cv["q"], cv["y"], color=COLORS[c], marker=MARKERS[c], markersize=7, linewidth=2, label=lab)
        ax.annotate(lab, (cv["q"][-1], cv["y"][-1]), xytext=(-8, 8), textcoords="offset points",
                    ha="right", color="#52514e", fontsize=8)
    if out["init_recall_c2a_only"] is not None:
        ax.axhline(out["init_recall_c2a_only"], color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("лише C2A, без донавчання", (ax.get_xlim()[0], out["init_recall_c2a_only"]),
                    xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xlabel("Людських запитів (зображень SARD)")
    ax.set_ylabel("Recall на реальному тесті SARD")
    ax.set_title(f"Питання 2: синтетика → реальність (сіди: {len(seeds)}, смуга — 95 % ДІ)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, "q2_recall_vs_queries")

    # (б) якість від людських годин — умови А/Б/В
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    for c in ("A", "B", "C"):
        cv = curves[c]
        ax.fill_between(cv["x_h"], cv["lo"], cv["hi"], color=COLORS[c], alpha=0.12, linewidth=0)
        ax.plot(cv["x_h"], cv["y"], color=COLORS[c], marker=MARKERS[c], markersize=7, linewidth=2,
                label=LABELS_Q2[c])
    ax.set_xlabel("Людські години (модель часу з lead_time Label Studio)")
    ax.set_ylabel("Recall на реальному тесті SARD")
    ax.set_title("Питання 2: якість від людського часу, умови А–В", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "q2_recall_vs_hours")
    pd.DataFrame(table).to_csv(TAB / "q2_summary.csv", index=False)
    return out


# ---------------- економія ----------------

def boot_ci(v, rng, n_boot: int) -> list[float]:
    v = np.asarray(v, dtype=float)
    b = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def hours_to_reach(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """Перші людські години, коли крива якості досягла target (лінійна інтерполяція)."""
    for k in range(len(y)):
        if y[k] >= target:
            if k == 0:
                return float(x[0])
            return float(x[k - 1] + (target - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1]))
    return None


def q2_economy(curves: dict, rng, n_boot: int) -> dict:
    """Економія людського часу в умовах Б, В відносно А (ручна розмітка)."""
    out = {}
    # (1) за рівного бюджету запитів: години на весь бюджет
    for c in ("B", "C"):
        save = 1 - np.array(curves[c]["per_seed_hours"]) / np.array(curves["A"]["per_seed_hours"])
        out[f"hours_saved_vs_A_equal_queries_{c}"] = {"mean": float(save.mean()),
                                                     "ci95": boot_ci(save, rng, n_boot),
                                                     "per_seed": save.round(4).tolist()}
    # (2) за рівної якості: години до recall = target
    for target in (0.45, 0.50):
        row = {}
        for c in ("A", "B", "C"):
            row[c] = [hours_to_reach(x, y, target) for x, y in zip(curves[c]["xs"], curves[c]["ys"])]
        for c in ("B", "C"):
            pairs = [(a, b) for a, b in zip(row["A"], row[c]) if a is not None and b is not None]
            save = np.array([1 - b / a for a, b in pairs])
            out[f"hours_to_recall_{target:.2f}"] = {k: v for k, v in row.items()}
            out[f"hours_saved_vs_A_at_recall_{target:.2f}_{c}"] = (
                {"mean": float(save.mean()), "ci95": boot_ci(save, rng, n_boot),
                 "seeds_reached": len(pairs)} if len(pairs) else None)
    return out


def routing_economy(model: dict, n_boot: int, rng) -> list[dict]:
    """Економія від маршрутизації (журнали logs/routing/routes_*.json)."""
    c = model["coef"]
    groups: dict[str, list] = {}
    for f in sorted((ROOT / "logs/routing").glob("routes_*.json")):
        if "smoke" in f.name:
            continue
        run = f.stem.replace("routes_", "")
        ds = "sard" if "sard" in run else "c2a"
        routes = json.loads(f.read_text())
        ev = json.loads((f.parent / f"eval_{run}.json").read_text())["eval"]
        n_lab = n_items = base_s = routed_s = 0.0
        n_auto = 0
        for r in routes.values():
            k = len(r["label_decision"])
            n_lab += k
            n_items += r["n_human_items"]
            base_s += c[f"t0_{ds}"] + c["b_kept"] * k
            if not r["auto_reject"] and r["n_human_items"] > 0:
                routed_s += c[f"t0_{ds}"] + c["b_kept"] * r["n_human_items"]
            else:
                n_auto += 1
        groups.setdefault(run.rsplit("_seed", 1)[0], []).append({
            "items_saved": 1 - n_items / n_lab if n_lab else None,
            "images_fully_auto": n_auto / len(routes),
            "hours_saved": 1 - routed_s / base_s,
            "baseline_h": base_s / 3600, "routed_h": routed_s / 3600,
            "errors_to_human_frac": ev["errors_to_human_frac"], "errors_leaked": ev["errors_leaked"]})
    rows = []
    for g, runs in groups.items():
        row = {"run": g, "seeds": len(runs)}
        for k in runs[0]:
            v = [r[k] for r in runs]
            row[f"{k}_mean"] = float(np.mean(v))
            if k in ("hours_saved", "errors_to_human_frac"):
                row[f"{k}_ci95"] = boot_ci(v, rng, n_boot)
        rows.append(row)
    return rows


def economy_table(summary: dict) -> None:
    """Зведена таблиця «що зекономлено і якою ціною» (reports/tables/economy_summary.csv)."""
    rows = []
    q1 = pd.read_csv(TAB / "q1_summary.csv")
    for st in ("uncertainty", "cleanlab"):
        r_st = np.array(json.loads(q1[(q1.strategy == st) & (q1.metric == "share_checked_to_80pct")].per_seed.iloc[0]))
        r_rn = np.array(json.loads(q1[(q1.strategy == "random") & (q1.metric == "share_checked_to_80pct")].per_seed.iloc[0]))
        save = 1 - r_st / r_rn
        rows.append({"source": "Q1 аудит C2A", "what": f"{st} vs random: перевірок до 80 % помилок",
                     "saving_mean": float(save.mean()), "saving_min": float(save.min()),
                     "saving_max": float(save.max()), "price": "—"})
    eco = summary["q2"].get("economy", {})
    for c, name in (("B", "Б: random + пре-анотація"), ("C", "В: АМН + пре-анотація")):
        e = eco.get(f"hours_saved_vs_A_equal_queries_{c}")
        if e:
            rows.append({"source": "Q2 C2A→SARD", "what": f"{name} vs А: години за 350 запитів",
                         "saving_mean": e["mean"], "saving_min": min(e["per_seed"]),
                         "saving_max": max(e["per_seed"]), "price": "recall: див. q2_summary"})
        e = eco.get(f"hours_saved_vs_A_at_recall_0.50_{c}")
        if e:
            rows.append({"source": "Q2 C2A→SARD", "what": f"{name} vs А: години до recall 0.50",
                         "saving_mean": e["mean"], "saving_min": e["ci95"][0], "saving_max": e["ci95"][1],
                         "price": f"сідів, що досягли: {e['seeds_reached']}"})
    for r in summary.get("routing_economy", []):
        rows.append({"source": f"Маршрутизація {r['run']}", "what": "години верифікації vs перевірка всіх рамок",
                     "saving_mean": r["hours_saved_mean"], "saving_min": r["hours_saved_ci95"][0],
                     "saving_max": r["hours_saved_ci95"][1],
                     "price": f"до людини {r['errors_to_human_frac_mean']:.1%} помилок, "
                              f"пропущено {r['errors_leaked_mean']:.0f}"})
    pd.DataFrame(rows).to_csv(TAB / "economy_summary.csv", index=False)


# ---------------- маршрутизація ----------------

def routing_table() -> list[dict]:
    rows = []
    for f in sorted((ROOT / "logs/routing").glob("eval_*.json")):
        if "smoke" in f.name:
            continue
        d = json.loads(f.read_text())
        rows.append({"run": f.stem.replace("eval_", ""), "dataset": d["dataset"], "seed": d["seed"],
                     "target_errors_to_human": d["target"], "feasible_on_calib": d["feasible"],
                     **{f"cfg_{k}": v for k, v in d["config"].items()},
                     **{f"calib_{k}": d["calib"][k] for k in ("errors_to_human_frac", "human_items")},
                     **{f"eval_{k}": v for k, v in d["eval"].items()}})
    if rows:
        TAB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(TAB / "routing_summary.csv", index=False)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1-tag", default="main")
    ap.add_argument("--q2-tag", default="q2_sard")
    ap.add_argument("--q2-dataset", default="sard")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--q2-seeds", type=int, nargs="+", default=None,
                    help="сіди питання 2, якщо їх більше, ніж у питанні 1 (типово = --seeds)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    summary = {"q1": q1_analysis(a.q1_tag, a.seeds, a.n_boot, rng),
               "q2": q2_analysis(a.q2_tag, a.q2_seeds or a.seeds, a.q2_dataset, a.n_boot, rng),
               "routing": routing_table(),
               "cost_model": cost_model.load()}
    summary["routing_economy"] = routing_economy(summary["cost_model"], a.n_boot, rng)
    economy_table(summary)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    print(json.dumps({k: (v if k != "cost_model" else "...") for k, v in summary.items()},
                     indent=1, ensure_ascii=False, default=float)[:4000])


if __name__ == "__main__":
    main()
