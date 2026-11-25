"""Пояснюваність YOLO11n: карта оцінок голови і оклюзія на прикладах з тесту SARD (влучання, пропуск, хибна рамка).

  python src/explain.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

from exp_sahi import iou_matrix
from predict import CONF_ACCEPT, WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
TEST, IMAGES = ROOT / "data/processed/sard/test.json", ROOT / "data/processed/sard/images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRIDS = (80, 40, 20)  # сітки детект-голови для 640 px (кроки 8, 16, 32)


def _tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img.resize((640, 640)), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    return x.to(DEVICE)


def score_map(net, img: Image.Image) -> np.ndarray:
    """Оцінка «людина» в кожній клітинці сітки до NMS (максимум за трьома масштабами)."""
    with torch.no_grad():
        s = net(_tensor(img))[0][0, 4].cpu().numpy()
    out, k = np.zeros((640, 640), np.float32), 0
    for g in GRIDS:
        m = s[k:k + g * g].reshape(g, g)
        k += g * g
        out = np.maximum(out, np.asarray(Image.fromarray(m).resize((640, 640), Image.NEAREST)))
    return np.asarray(Image.fromarray(out).resize(img.size, Image.BILINEAR))


def occlusion(net, img: Image.Image, box: np.ndarray, patch: int = 32) -> np.ndarray:
    """На скільки падає оцінка рамки, якщо закрити ділянку сірим квадратом."""
    x = _tensor(img)
    sx, sy = 640 / img.size[0], 640 / img.size[1]
    cx, cy = (box[0] + box[2] / 2) * sx, (box[1] + box[3] / 2) * sy

    def box_score(xx):
        with torch.no_grad():
            o = net(xx)[0][0]
        d = (o[0] - cx) ** 2 + (o[1] - cy) ** 2
        near = d < (max(box[2] * sx, box[3] * sy) / 2) ** 2
        return float(o[4][near].max()) if near.any() else 0.0

    base, n = box_score(x), 640 // patch
    heat = np.zeros((n, n), np.float32)
    for i in range(n):
        batch = x.repeat(n, 1, 1, 1)
        for j in range(n):
            batch[j, :, i * patch:(i + 1) * patch, j * patch:(j + 1) * patch] = 0.5
        with torch.no_grad():
            o = net(batch)[0]
        d = (o[:, 0] - cx) ** 2 + (o[:, 1] - cy) ** 2
        near = d < (max(box[2] * sx, box[3] * sy) / 2) ** 2
        sc = torch.where(near, o[:, 4], torch.zeros_like(o[:, 4])).max(1).values.cpu().numpy()
        heat[i] = np.maximum(base - sc, 0)
    heat = np.asarray(Image.fromarray(heat).resize(img.size, Image.NEAREST))
    return heat / (base + 1e-9)


def pick_examples(model: YOLO, coco: dict) -> list[tuple[str, dict, np.ndarray, np.ndarray]]:
    gts: dict[int, list] = {}
    for a in coco["annotations"]:
        gts.setdefault(a["image_id"], []).append(a["bbox"])
    best = {"влучання": (-1, None), "пропуск": (-1, None), "хибна рамка": (-1, None)}
    for im in coco["images"]:
        r = model.predict(str(IMAGES / im["file_name"]), conf=CONF_ACCEPT, imgsz=640, verbose=False)[0]
        p = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]).reshape(-1, 4)
        g = np.array(gts.get(im["id"], []), float).reshape(-1, 4)
        m = iou_matrix(p, g) if len(p) and len(g) else np.zeros((len(p), len(g)))
        tp = int((m.max(0) >= 0.5).sum()) if len(p) else 0
        fn, fp = len(g) - tp, int((m.max(1) < 0.5).sum()) if len(g) else len(p)
        score = {"влучання": tp if fn == 0 and fp == 0 else -1,
                 "пропуск": fn if tp == 0 else -1, "хибна рамка": fp if not len(g) else -1}
        for k, s in score.items():
            if s > best[k][0]:
                best[k] = (s, (im, g, p))
    return [(k, *v[1]) for k, v in best.items() if v[1] is not None]


def main() -> None:
    model = YOLO(str(WEIGHTS))
    net = model.model.to(DEVICE).eval()
    coco = json.loads(TEST.read_text(encoding="utf-8"))
    ex = pick_examples(model, coco)
    fig, axes = plt.subplots(3, len(ex), figsize=(4.2 * len(ex), 12.4), squeeze=False)
    for j, (kind, im, g, p) in enumerate(ex):
        img = Image.open(IMAGES / im["file_name"]).convert("RGB")
        top = p[0] if len(p) else None
        maps = [None, score_map(net, img), occlusion(net, img, top) if top is not None else None]
        for row, ax in enumerate(axes[:, j]):
            ax.imshow(img)
            if maps[row] is not None:
                ax.imshow(maps[row], cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            for b in g:
                ax.add_patch(Rectangle(b[:2], b[2], b[3], fill=False, ec="white", ls="--", lw=1.5))
            for k, b in enumerate(p):
                if row < 2 or k == 0:
                    ax.add_patch(Rectangle(b[:2], b[2], b[3], fill=False, ec="#1baf7a", lw=2))
            ax.set_axis_off()
        axes[0, j].set_title(f"{kind}: розмітка {len(g)}, модель {len(p)}", fontsize=10)
    for row, txt in enumerate(["знімок", "оцінка «людина» до NMS", "оклюзія: що важливо для рамки"]):
        axes[row, 0].text(-0.04, 0.5, txt, transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=10)
    fig.suptitle("Пояснення прогнозів. Біла пунктирна — розмітка, зелена — модель (≥ 0.3)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"explain_sard.{ext}", dpi=160)
    print([(k, im["file_name"]) for k, im, _, _ in ex])


if __name__ == "__main__":
    main()
