"""Гармонізація C2A: COCO + YOLO, пози, перерозбиття за сценами.

  python src/harmonize_c2a.py
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SPLITS_IN_ZIP = ("train", "val", "test")
POSE_DIR = "new_dataset3/All labels with Pose information/labels/"
NAME_RE = re.compile(r"^(?P<dtype>.+)_image(?P<bg>\d+)_(?P<var>\d+)$")


def load_schema() -> dict:
    with open(ROOT / "configs" / "schema.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_label(text: str, where: str) -> list[tuple[float, float, float, float, int]]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"{where}: очікувалось 6 значень, отримано '{line}'")
        xc, yc, w, h = map(float, parts[1:5])
        rows.append((xc, yc, w, h, int(float(parts[5]))))
    return rows


def index_zip(zf: zipfile.ZipFile) -> list[dict]:
    """Список усіх зображень з метаданими (тип лиха, сцена, оригінальний спліт)."""
    pose_files = {Path(n).stem: n for n in zf.namelist()
                  if n.startswith(POSE_DIR) and n.endswith(".txt")}
    items = []
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) != 4 or parts[1] not in SPLITS_IN_ZIP or parts[2] != "images":
            continue
        if not name.lower().endswith(".png"):
            continue
        stem = Path(name).stem
        m = NAME_RE.match(stem)
        if not m:
            raise ValueError(f"Неочікуване ім'я файлу: {name}")
        if stem not in pose_files:
            raise ValueError(f"Немає мітки пози для {name}")
        items.append({
            "stem": stem, "img": name, "pose_label": pose_files[stem],
            "dtype": m["dtype"], "scene": f"{m['dtype']}_image{m['bg']}",
            "variant": int(m["var"]), "orig_split": parts[1],
        })
    return items


def split_scenes(items: list[dict], val_frac: float, n_scenes: int | None,
                 rng: random.Random) -> dict[str, str]:
    """scene -> train/val, стратифіковано за типом лиха."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for s in sorted({(it["dtype"], it["scene"]) for it in items}):
        by_type[s[0]].append(s[1])
    total = sum(len(v) for v in by_type.values())
    assign = {}
    for dtype, scenes in sorted(by_type.items()):
        rng.shuffle(scenes)
        if n_scenes is not None:
            scenes = scenes[:round(n_scenes * len(scenes) / total)]
        n_val = round(len(scenes) * val_frac)
        for i, s in enumerate(scenes):
            assign[s] = "val" if i < n_val else "train"
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=ROOT / "data/c2a/raw/C2A_Dataset.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/c2a")
    ap.add_argument("--n-scenes", type=int, default=None,
                    help="скільки сцен узяти (None = усі)")
    ap.add_argument("--per-scene", type=int, default=1,
                    help="максимум варіантів накладання людей з однієї сцени")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    schema = load_schema()
    pose_map = {int(k): v for k, v in schema["pose_maps"]["c2a"].items()}
    defaults = schema["defaults"]["c2a"]
    rng = random.Random(args.seed)

    zf = zipfile.ZipFile(args.zip)
    items = index_zip(zf)
    print(f"зображень у zip: {len(items)}, сцен: {len({i['scene'] for i in items})}")

    assign = split_scenes(items, args.val_frac, args.n_scenes, rng)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["scene"] in assign:
            by_scene[it["scene"]].append(it)
    chosen: dict[str, list[dict]] = {"train": [], "val": []}
    for scene in sorted(by_scene):
        variants = sorted(by_scene[scene], key=lambda x: x["variant"])
        rng.shuffle(variants)
        chosen[assign[scene]].extend(variants[:args.per_scene])

    stats: dict = {"seed": args.seed, "max_side": args.max_side,
                   "per_scene": args.per_scene, "val_frac": args.val_frac}
    for split, sel in chosen.items():
        sel.sort(key=lambda x: x["stem"])
        out_img = args.out / "images" / split
        out_lbl = args.out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        coco = {"info": {"description": f"C2A harmonized subset ({split})"},
                "categories": schema["categories"], "images": [], "annotations": []}
        ann_id = 1
        for img_id, it in enumerate(tqdm(sel, desc=split), start=1):
            im = Image.open(io.BytesIO(zf.read(it["img"]))).convert("RGB")
            orig_w, orig_h = im.size
            scale = min(1.0, args.max_side / max(im.size))
            if scale < 1.0:
                im = im.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
            W, H = im.size
            fname = f"c2a_{it['stem']}.jpg"
            im.save(out_img / fname, quality=92)
            rows = parse_label(zf.read(it["pose_label"]).decode(), it["pose_label"])
            coco["images"].append({
                "id": img_id, "file_name": f"{split}/{fname}", "width": W, "height": H,
                "source_dataset": "c2a", "source_file": it["img"],
                "orig_size": [orig_w, orig_h], "scene": it["scene"],
                "disaster_type": it["dtype"], "c2a_orig_split": it["orig_split"],
            })
            lines = []
            for xc, yc, w, h, pose in rows:
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                bw, bh = w * W, h * H
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [(xc - w / 2) * W, (yc - h / 2) * H, bw, bh],
                    "area": bw * bh, "iscrowd": 0,
                    "attributes": {
                        "pose": pose_map.get(pose, "unknown"),
                        "visibility": defaults["visibility"],
                        "modality": defaults["modality"],
                        "provenance": "original_dataset", "confidence": 1.0,
                    },
                })
                ann_id += 1
            (out_lbl / f"c2a_{it['stem']}.txt").write_text("\n".join(lines))
        with open(args.out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(coco, f)
        stats[split] = {
            "images": len(coco["images"]), "persons": len(coco["annotations"]),
            "scenes": len({i["scene"] for i in sel}),
            "disaster_type": dict(Counter(i["dtype"] for i in sel)),
            "pose": dict(Counter(a["attributes"]["pose"] for a in coco["annotations"])),
        }

    tr = {i["scene"] for i in chosen["train"]}
    va = {i["scene"] for i in chosen["val"]}
    assert not tr & va, "перетин сцен між train і val"
    stats["scene_overlap_train_val"] = 0

    (args.out / "c2a.yaml").write_text(yaml.safe_dump({
        "path": str(args.out.resolve()), "train": "images/train", "val": "images/val",
        "names": {0: "person"}}, allow_unicode=True))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
