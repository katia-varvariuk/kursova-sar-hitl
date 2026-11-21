"""Завантаження гармонізованих наборів у FiftyOne.

  python src/load_fiftyone.py [--launch]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fiftyone as fo

ROOT = Path(__file__).resolve().parents[1]


def build(data_dir: Path, name: str, splits: list[str]) -> fo.Dataset:
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    for split in splits:
        coco = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
        anns: dict[int, list] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        samples = []
        for im in coco["images"]:
            W, H = im["width"], im["height"]
            dets = []
            for a in anns.get(im["id"], []):
                x, y, w, h = a["bbox"]
                dets.append(fo.Detection(
                    label="person", bounding_box=[x / W, y / H, w / W, h / H],
                    **a["attributes"]))
            samples.append(fo.Sample(
                filepath=str((data_dir / "images" / im["file_name"]).resolve()),
                split=split, tags=[split], source_dataset=im["source_dataset"],
                scene=im["scene"], disaster_type=im["disaster_type"],
                ground_truth=fo.Detections(detections=dets)))
        ds.add_samples(samples)
    ds.compute_metadata()
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/c2a")
    ap.add_argument("--name", default="kursova_c2a")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    ds = build(args.data, args.name, args.splits)
    print(ds)
    print("рамок:", ds.count("ground_truth.detections"))
    print("пози:", ds.count_values("ground_truth.detections.pose"))
    print("тип лиха:", ds.count_values("disaster_type"))
    if args.launch:
        session = fo.launch_app(ds)
        session.wait()


if __name__ == "__main__":
    main()
