"""Гармонізація SARD (копія Roboflow): перерозбиття блоками кадрів з буфером.

  python src/harmonize_sard.py
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^gss(?P<fid>\d+)_(jpg|jpeg|png)\.rf\.(?P<h>[0-9a-f]+)$", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=ROOT / "data/sard/archive (4).zip")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/sard")
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--guard", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    schema = yaml.safe_load((ROOT / "configs/schema.yaml").read_text(encoding="utf-8"))
    defaults = schema["defaults"]["sard"]
    zf = zipfile.ZipFile(a.zip)
    names = set(zf.namelist())

    tiles = []
    for n in sorted(names):
        parts = n.split("/")
        if len(parts) == 4 and parts[2] == "images" and parts[3]:
            stem = Path(parts[3]).stem
            m = NAME_RE.match(stem)
            if not m:
                raise ValueError(f"неочікуване ім'я: {n}")
            tiles.append({"img": n, "lbl": f"{parts[0]}/{parts[1]}/labels/{stem}.txt",
                          "fid": int(m["fid"]), "hash": m["h"], "rf_split": parts[1]})

    fids = sorted({t["fid"] for t in tiles})
    blocks = sorted({f // a.block for f in fids})
    rng = random.Random(a.seed)
    test_blocks = set(rng.sample(blocks, round(a.test_frac * len(blocks))))
    test_fids = {f for f in fids if f // a.block in test_blocks}
    tf_sorted = sorted(test_fids)

    def near_test(f: int) -> bool:
        import bisect
        i = bisect.bisect_left(tf_sorted, f)
        return any(0 <= j < len(tf_sorted) and abs(tf_sorted[j] - f) <= a.guard for j in (i - 1, i))

    split_of = {}
    dropped = 0
    for f in fids:
        if f in test_fids:
            split_of[f] = "test"
        elif near_test(f):
            split_of[f] = None
            dropped += 1
        else:
            split_of[f] = "pool"

    stats = {"seed": a.seed, "block": a.block, "test_frac": a.test_frac, "guard": a.guard,
             "frames_dropped_by_guard": dropped,
             "source": "Kaggle nikolasgegenava/sard-search-and-rescue (Roboflow Sard v17 TILES_3x3)"}
    for split in ("pool", "test"):
        sel = [t for t in tiles if split_of[t["fid"]] == split]
        out_img = a.out / "images" / split
        out_lbl = a.out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        coco = {"info": {"description": f"SARD tiles harmonized ({split})"},
                "categories": schema["categories"], "images": [], "annotations": []}
        ann_id = 1
        for img_id, t in enumerate(sel, start=1):
            im = Image.open(io.BytesIO(zf.read(t["img"]))).convert("RGB")
            W, H = im.size
            fname = f"sard_gss{t['fid']:04d}_{t['hash'][:10]}.jpg"
            im.save(out_img / fname, quality=95)
            txt = zf.read(t["lbl"]).decode() if t["lbl"] in names else ""
            rows = [l.split() for l in txt.splitlines() if l.strip()]
            coco["images"].append({
                "id": img_id, "file_name": f"{split}/{fname}", "width": W, "height": H,
                "source_dataset": "sard", "source_file": t["img"], "frame_id": t["fid"],
                "scene": f"sard_block{t['fid'] // a.block}", "roboflow_split": t["rf_split"],
                "disaster_type": "none"})
            lines = []
            for r in rows:
                if len(r) != 5:
                    raise ValueError(f"{t['lbl']}: '{' '.join(r)}'")
                xc, yc, w, h = map(float, r[1:])
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                bw, bh = w * W, h * H
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [(xc - w / 2) * W, (yc - h / 2) * H, bw, bh], "area": bw * bh,
                    "iscrowd": 0,
                    "attributes": {"pose": "unknown", "visibility": defaults["visibility"],
                                   "modality": defaults["modality"],
                                   "provenance": "original_dataset", "confidence": 1.0}})
                ann_id += 1
            (out_lbl / f"{Path(fname).stem}.txt").write_text("\n".join(lines))
        (a.out / f"{split}.json").write_text(json.dumps(coco))
        counts = Counter(x["image_id"] for x in coco["annotations"])
        stats[split] = {"tiles": len(sel), "frames": len({t["fid"] for t in sel}),
                        "persons": len(coco["annotations"]),
                        "tiles_without_persons": len(sel) - len(counts),
                        "blocks": len({t["fid"] // a.block for t in sel})}

    (a.out / "sard.yaml").write_text(yaml.safe_dump({
        "path": str(a.out.resolve()), "train": "images/pool", "val": "images/test",
        "names": {0: "person"}}))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
