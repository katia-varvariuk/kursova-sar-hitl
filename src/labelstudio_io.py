"""Label Studio через REST API: create / push / export (з lead_time).

  python src/labelstudio_io.py push --project 1 --mode preann --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/c2a"
LOG_DIR = ROOT / "logs/labelstudio"
DOC_ROOT_IN_CONTAINER = "/label-studio/files"   # LOCAL_FILES_DOCUMENT_ROOT


def env(key: str, default: str | None = None) -> str | None:
    if os.environ.get(key):
        return os.environ[key]
    f = ROOT / "docker/.env"
    if f.exists():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return default


class LS:
    def __init__(self) -> None:
        self.url = env("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
        key = env("LABEL_STUDIO_API_KEY")
        if not key:
            raise SystemExit("Немає LABEL_STUDIO_API_KEY (docker/.env або змінна середовища)")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Token {key}"

    def req(self, method: str, path: str, **kw):
        r = self.s.request(method, self.url + path, timeout=120, **kw)
        if r.status_code >= 400:
            raise SystemExit(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else None


def rel_url(p: Path) -> str:
    """Шлях для local-files: відносно kursova/ (у контейнері змонтовано ../data)."""
    rel = p.resolve().relative_to(ROOT).as_posix()
    return f"/data/local-files/?d={rel}"


def cmd_create(a) -> None:
    ls = LS()
    cfg = (ROOT / "labelstudio/verify_boxes.xml").read_text(encoding="utf-8")
    proj = ls.req("POST", "/api/projects", json={
        "title": a.title, "label_config": cfg,
        "description": "Шар 1: верифікація рамок «людина» (курсова, етап I)",
        "show_collab_predictions": True})
    ls.req("POST", "/api/storages/localfiles", json={
        "project": proj["id"], "title": "kursova data",
        "path": f"{DOC_ROOT_IN_CONTAINER}/data", "use_blob_urls": False})
    print(f"project id={proj['id']}  {ls.url}/projects/{proj['id']}")


DATASETS = {  # набір -> (каталог, спліт, з якого беруться задачі)
    "c2a": (ROOT / "data/processed/c2a", "val"),
    "sard": (ROOT / "data/processed/sard", "pool"),
}


def choose_images(dataset: str, n: int, offset: int, seed: int, max_persons: int) -> list[dict]:
    """Фіксоване перемішування (seed) і зріз [offset, offset+n): різні виклики з"""
    root, split = DATASETS[dataset]
    coco = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    counts: dict[int, int] = {}
    for x in coco["annotations"]:
        counts[x["image_id"]] = counts.get(x["image_id"], 0) + 1
    pool = sorted((im for im in coco["images"] if 0 < counts.get(im["id"], 0) <= max_persons),
                  key=lambda im: im["id"])
    random.Random(seed).shuffle(pool)
    sel = pool[offset:offset + n]
    for im in sel:
        im["n_persons_gt"] = counts[im["id"]]
        im["path"] = root / "images" / im["file_name"]
    return sel


def cmd_push(a) -> None:
    ls = LS()
    images = choose_images(a.dataset, a.n, a.offset, a.seed, a.max_persons)
    det = None
    if a.mode == "preann":
        from ultralytics import YOLO
        det = YOLO(a.model)
    tasks = []
    for im in images:
        path = im["path"]
        task = {"data": {"image": rel_url(path), "image_id": im["id"], "dataset": a.dataset,
                         "file_name": im["file_name"], "mode": a.mode,
                         "n_persons_gt": im["n_persons_gt"],
                         "hint": ("Перевірте рамки моделі: підтвердьте (1) або виправте (2)."
                                  if a.mode == "preann" else
                                  "Намалюйте рамку на кожну людину (p), потім вердикт.")}}
        if det is not None:
            r = det.predict(str(path), conf=a.conf, max_det=300, verbose=False)[0]
            W, H = im["width"], im["height"]
            result, confs = [], []
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                result.append({"from_name": "person_box", "to_name": "image",
                               "type": "rectanglelabels", "original_width": W,
                               "original_height": H, "score": c,
                               "value": {"x": 100 * x1 / W, "y": 100 * y1 / H,
                                         "width": 100 * (x2 - x1) / W,
                                         "height": 100 * (y2 - y1) / H,
                                         "rotation": 0, "rectanglelabels": ["person"]}})
                confs.append(c)
            # score задачі = невизначеність (для сортування черги в Data Manager)
            cc = np.clip(np.array(confs), 1e-6, 1 - 1e-6)
            unc = float(np.mean(-(cc * np.log2(cc) + (1 - cc) * np.log2(1 - cc)))) if len(cc) else 0.0
            task["predictions"] = [{"model_version": Path(a.model).as_posix(), "score": unc,
                                    "result": result}]
        tasks.append(task)
    ls.req("POST", f"/api/projects/{a.project}/import", json=tasks)
    print(f"імпортовано {len(tasks)} задач ({a.mode}) у проєкт {a.project}")


def boxes_from_result(result: list[dict]) -> np.ndarray:
    b = [[r["value"]["x"], r["value"]["y"], r["value"]["width"], r["value"]["height"]]
         for r in result if r.get("type") == "rectanglelabels"]
    return np.array(b, dtype=float).reshape(-1, 4)


def match_count(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> int:
    from oracle import iou_xywh
    if len(a) == 0 or len(b) == 0:
        return 0
    iou = iou_xywh(a, b)
    n = 0
    while iou.size and iou.max() >= thr:
        i, j = np.unravel_index(iou.argmax(), iou.shape)
        n += 1
        iou[i, :] = -1
        iou[:, j] = -1
    return n


def cmd_export(a) -> None:
    ls = LS()
    data = ls.req("GET", f"/api/projects/{a.project}/export?exportType=JSON&download_all_tasks=false")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = LOG_DIR / f"project{a.project}_export.json"
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    pred_by_task: dict[int, list] = {}
    for t in data:
        if t.get("predictions"):
            p = t["predictions"][0]
            if isinstance(p, int):
                p = ls.req("GET", f"/api/predictions/{p}")
            pred_by_task[t["id"]] = p["result"]
    (LOG_DIR / f"project{a.project}_predictions.json").write_text(
        json.dumps({str(k): v for k, v in pred_by_task.items()}), encoding="utf-8")
    rows = []
    for t in data:
        pred = boxes_from_result(pred_by_task[t["id"]]) if t["id"] in pred_by_task else np.zeros((0, 4))
        for ann in t.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            final = boxes_from_result(ann["result"])
            verdict = next((r["value"]["choices"][0] for r in ann["result"]
                            if r.get("from_name") == "verdict"), None)
            kept = match_count(pred, final)
            rows.append({
                "task_id": t["id"], "annotation_id": ann["id"],
                "annotator": ann.get("completed_by"), "mode": t["data"].get("mode"),
                "dataset": t["data"].get("dataset"),
                "file_name": t["data"].get("file_name"),
                "n_persons_gt": t["data"].get("n_persons_gt"),
                "lead_time_s": ann.get("lead_time"), "verdict": verdict,
                "n_pred": len(pred), "n_final": len(final), "n_kept": kept,
                "n_added": len(final) - kept, "n_removed": len(pred) - kept,
                "accepted_unchanged": int(len(pred) > 0 and kept == len(pred) == len(final)),
                "created_at": ann.get("created_at"),
            })
    df = pd.DataFrame(rows)
    csv = LOG_DIR / f"project{a.project}_summary.csv"
    df.to_csv(csv, index=False)
    print(f"анотацій: {len(df)} -> {csv}")
    if len(df):
        print(df.groupby(["dataset", "mode"])[["lead_time_s", "n_final", "n_added", "n_removed"]].median())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("--title", required=True)
    p = sub.add_parser("push")
    p.add_argument("--project", type=int, required=True)
    p.add_argument("--mode", choices=["preann", "blind"], required=True)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="c2a")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-persons", type=int, default=15)
    p.add_argument("--model", default=str(ROOT / "runs/baseline_c2a/yolo11n_e30/weights/best.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    e = sub.add_parser("export"); e.add_argument("--project", type=int, required=True)
    a = ap.parse_args()
    {"create": cmd_create, "push": cmd_push, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
