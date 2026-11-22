"""Перевірка формату маніфесту кропів: схема, файли, правила розбиття.

  python src/validate_manifest.py [--no-files] [--write-schema]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
DEFAULT = PROC / "manifest/crops.jsonl"
SCHEMA_PATH = ROOT / "configs/crop_manifest.schema.json"
SYNTHETIC = {"c2a"}
REAL = {"sard"}

Provenance = Literal["original_dataset", "model", "vlm", "annotator", "expert", "consensus"]


class Attributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose: Literal["standing", "walking", "sitting", "bent", "kneeling", "lying", "unknown"]
    visibility: Literal["full", "partial", "partially_buried", "unknown"]
    modality: Literal["rgb", "thermal"]
    environment: Literal["rubble", "water", "fire_smoke", "traffic", "forest_mountain", "other"]
    group: Literal["single", "group"]
    gesture: Literal["yes", "no", "unknown"]


class AttributeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose: Optional[Provenance]
    environment: Optional[Provenance]
    group: Optional[Literal["rule", "annotator", "expert", "vlm", "model"]]
    keypoints: Optional[Provenance]
    priority: Optional[Provenance]


class CropRecord(BaseModel):
    """Один запис маніфесту = одна підтверджена людина."""
    model_config = ConfigDict(extra="forbid")
    crop_id: str = Field(pattern=r"^[a-z0-9]+_[a-z]+_\d{6}_\d{3}$")
    source_dataset: Literal["c2a", "sard"]
    source_split: Literal["train", "val", "pool", "test"] = Field(
        description="файл гармонізованого COCO, у якому шукати image_id")
    image_id: int = Field(ge=0)
    file_name: str
    scene: str
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x, y, w, h], px вихідного зображення")
    keypoints: Optional[list[list[float]]] = Field(
        None, description="17 точок COCO [x, y, conf] у координатах вихідного зображення; null — не визначено")
    crop_path: str
    context_path: str
    context_bbox: list[float] = Field(min_length=4, max_length=4)
    priority: Literal["red", "yellow", "green", "undetermined"]
    attributes: Attributes
    provenance: Provenance
    attribute_provenance: AttributeProvenance
    confidence: float = Field(ge=0.0, le=1.0)
    split: Literal["train", "val", "test"]

    @field_validator("bbox")
    @classmethod
    def positive_size(cls, v):
        if v[2] <= 0 or v[3] <= 0:
            raise ValueError("ширина і висота рамки мають бути > 0")
        return v

    @field_validator("keypoints")
    @classmethod
    def seventeen(cls, v):
        if v is not None and (len(v) != 17 or any(len(p) != 3 for p in v)):
            raise ValueError("очікується 17 точок по [x, y, conf]")
        return v


def image_sizes() -> dict[tuple[str, str, int], tuple[int, int]]:
    sizes = {}
    for ds, splits in {"c2a": ("train", "val"), "sard": ("pool", "test")}.items():
        for sp in splits:
            for im in json.loads((PROC / ds / f"{sp}.json").read_text(encoding="utf-8"))["images"]:
                sizes[(ds, sp, im["id"])] = (im["width"], im["height"])
    return sizes


def check_record(r: CropRecord, size, check_files: bool) -> list[str]:
    err = []
    W, H = size
    x, y, w, h = r.bbox
    if x < -0.5 or y < -0.5 or x + w > W + 0.5 or y + h > H + 0.5:
        err.append(f"bbox {r.bbox} виходить за межі {W}x{H}")
    cx, cy = x + w / 2, y + h / 2
    exp = [max(0, cx - 1.5 * w), max(0, cy - 1.5 * h), min(W, cx + 1.5 * w), min(H, cy + 1.5 * h)]
    got = [r.context_bbox[0], r.context_bbox[1], r.context_bbox[0] + r.context_bbox[2],
           r.context_bbox[1] + r.context_bbox[3]]
    if max(abs(a - b) for a, b in zip(exp, got)) > 0.05:
        err.append(f"context_bbox {r.context_bbox} не дорівнює рамці x3 {exp}")
    if r.keypoints is not None and r.attribute_provenance.keypoints is None:
        err.append("є keypoints, але не вказано їх provenance")
    if r.source_dataset in SYNTHETIC and r.split == "test":
        err.append("синтетичний набір у test")
    if check_files:
        from PIL import Image
        for rel, box in ((r.crop_path, r.bbox), (r.context_path, r.context_bbox)):
            p = PROC / rel
            if not p.exists():
                err.append(f"немає файлу {rel}")
                continue
            with Image.open(p) as im:
                bw, bh = im.size
            if not (abs(bw - box[2]) <= 2 and abs(bh - box[3]) <= 2):
                err.append(f"{rel}: розмір {bw}x{bh} не відповідає рамці {box[2]:.1f}x{box[3]:.1f}")
    return err


def validate(path: Path, check_files: bool = True) -> tuple[int, list[str]]:
    sizes = image_sizes()
    errors: list[str] = []
    seen_ids = set()
    img_split: dict[tuple[str, int], set] = defaultdict(set)
    scene_split: dict[tuple[str, str], set] = defaultdict(set)
    n = 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            n += 1
            try:
                r = CropRecord.model_validate_json(line)
            except ValidationError as e:
                errors.append(f"рядок {ln}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
                continue
            if r.crop_id in seen_ids:
                errors.append(f"рядок {ln}: повторний crop_id {r.crop_id}")
            seen_ids.add(r.crop_id)
            key = (r.source_dataset, r.source_split, r.image_id)
            size = sizes.get(key)
            if size is None:
                errors.append(f"рядок {ln}: невідоме зображення {'/'.join(map(str, key))}")
                continue
            if not r.crop_id.startswith(f"{r.source_dataset}_{r.source_split}_{r.image_id:06d}_"):
                errors.append(f"рядок {ln}: crop_id {r.crop_id} не відповідає джерелу {key}")
            if r.source_split == "test" and r.split != "test" or r.split == "test" and r.source_split != "test":
                errors.append(f"рядок {ln}: test маніфесту має збігатися з test джерела")
            errors += [f"рядок {ln} ({r.crop_id}): {m}" for m in check_record(r, size, check_files)]
            img_split[key].add(r.split)
            scene_split[(r.source_dataset, r.scene)].add(r.split)
    errors += [f"зображення {k} у кількох частинах: {sorted(v)}" for k, v in img_split.items() if len(v) > 1]
    errors += [f"сцена {k} у кількох частинах: {sorted(v)}" for k, v in scene_split.items() if len(v) > 1]
    test_sources = {k[0] for k, v in img_split.items() if "test" in v}
    if test_sources - REAL:
        errors.append(f"у test є нереальні джерела: {sorted(test_sources - REAL)}")
    return n, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--no-files", action="store_true")
    ap.add_argument("--write-schema", action="store_true")
    a = ap.parse_args()
    if a.write_schema:
        SCHEMA_PATH.write_text(json.dumps(CropRecord.model_json_schema(), ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"схема -> {SCHEMA_PATH.relative_to(ROOT)}")
        return
    n, errors = validate(a.path, check_files=not a.no_files)
    if errors:
        print(f"ПОМИЛКИ: {len(errors)} (записів {n})")
        for e in errors[:20]:
            print(" ", e)
        sys.exit(1)
    print(f"OK: {n} записів відповідають схемі розд. 6 і правилам розбиття")


if __name__ == "__main__":
    main()
