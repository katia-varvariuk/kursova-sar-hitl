"""Оркестратор (FastAPI): відбір АМН -> маршрутизація -> Label Studio -> донавчання.

  python -m uvicorn orchestrator:app --app-dir src --port 8000
"""
from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

import tracking  # noqa: F401 — вимикає підказку MLflow до його імпорту
from exp_q1_audit import uncertainty_scores
from labelstudio_io import LS, rel_url
from oracle import NoiseConfig, iou_xywh, make_noisy
from routing import RoutingConfig, route_one

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("ORCH_WORK", ROOT / "data/work/orchestrator"))
STATE_PATH = WORK / "state.json"
EVENTS = Path(os.environ.get("ORCH_EVENTS", ROOT / "logs/orchestrator/events.jsonl"))
SCHEMA = yaml.safe_load((ROOT / "configs/schema.yaml").read_text(encoding="utf-8"))

DATASETS = {  # набір -> (каталог, пул для розмітки, тест)
    "c2a": (ROOT / "data/processed/c2a", "train", "val"),
    "sard": (ROOT / "data/processed/sard", "pool", "test"),
}
LABELED = ("auto_accepted", "auto_rejected", "human_done")
MATCH_IOU = 0.5

app = FastAPI(title="kursova orchestrator", version="1.0",
              description="HITL-розмітка шару 1 (людина/поза/видимість): АМН + маршрутизація")
_lock = threading.RLock()
_train = {"running": False, "error": None, "started": None, "finished": None}


# ---------------- стан і журнал ----------------

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(kind: str, **kw) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now(), "event": kind, **kw}, ensure_ascii=False) + "\n")


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise HTTPException(409, "Оркестратор не ініціалізовано: POST /init")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(st: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


@contextmanager
def state():
    with _lock:
        st = load_state()
        yield st
        save_state(st)


def ls_call(method: str, path: str, **kw):
    """Виклик REST API Label Studio; помилки -> HTTP 502 замість SystemExit."""
    try:
        return LS().req(method, path, **kw)
    except SystemExit as e:
        raise HTTPException(502, f"Label Studio: {e}") from None


def rel(p: Path) -> str:
    """Шляхи в стані — відносно кореня репозиторію (однаково на хості й у контейнері)."""
    return Path(p).resolve().relative_to(ROOT).as_posix()


def coco_split(dataset: str, split: str) -> dict:
    root = DATASETS[dataset][0]
    return json.loads((root / f"{split}.json").read_text(encoding="utf-8"))


def image_path(st: dict, img: dict) -> Path:
    return DATASETS[st["dataset"]][0] / "images" / img["file_name"]


# ---------------- моделі запитів ----------------

class InitReq(BaseModel):
    dataset: Literal["c2a", "sard"] = "sard"
    project_id: int = Field(..., description="id проєкту Label Studio з шаблоном verify_boxes.xml")
    init_weights: str = "runs/baseline_c2a/yolo11n_e30/weights/best.pt"
    seed: int = 0
    noise_seed: int | None = Field(None, description="якщо задано — оракул вносить помилки в наявну "
                                                    "розмітку (як в експериментах), інакше розмітка як є")
    pool_limit: int = Field(0, description="обмежити пул (0 — увесь)")
    routing: dict = Field(default_factory=lambda: asdict(RoutingConfig(0.3, 0.5, 0.3, 0.05)),
                          description="пороги маршрутизації; типово — обрані в exp_routing на SARD")
    webhook_url: str | None = Field(None, description="адреса вебхука, яку бачить контейнер LS, напр. "
                                                     "http://host.docker.internal:8000/webhook/labelstudio")
    reset: bool = False


class SelectReq(BaseModel):
    k: int = 20
    strategy: Literal["uncertainty", "random"] = "uncertainty"
    candidates: int = Field(0, description="для uncertainty: оцінювати випадкову підвибірку пулу (0 — весь)")
    conf: float = 0.05


class TrainReq(BaseModel):
    epochs: int = 15
    imgsz: int = 640
    batch: int = 8


# ---------------- ініціалізація ----------------

@app.get("/health")
def health():
    return {"ok": True, "initialized": STATE_PATH.exists(), "training": _train["running"]}


@app.post("/init")
def init(req: InitReq):
    with _lock:
        if STATE_PATH.exists() and not req.reset:
            raise HTTPException(409, "Стан уже існує; передайте reset=true, щоб почати заново")
        weights = (ROOT / req.init_weights).resolve()
        if not weights.exists():
            raise HTTPException(400, f"Немає ваг {weights}")
        ls_call("GET", f"/api/projects/{req.project_id}")
        _, pool_split, _ = DATASETS[req.dataset]
        pool = coco_split(req.dataset, pool_split)
        if req.pool_limit:
            keep = set(random.Random(req.seed).sample([im["id"] for im in pool["images"]], req.pool_limit))
            pool = {**pool, "images": [im for im in pool["images"] if im["id"] in keep],
                    "annotations": [a for a in pool["annotations"] if a["image_id"] in keep]}
        labels, n_err = pool, 0
        if req.noise_seed is not None:
            labels, errors = make_noisy(pool, NoiseConfig(seed=req.noise_seed))
            n_err = len(errors)
            (WORK).mkdir(parents=True, exist_ok=True)
            (WORK / "injected_errors.json").write_text(json.dumps(errors), encoding="utf-8")
        by_img: dict[int, list] = {}
        for a in labels["annotations"]:
            by_img.setdefault(a["image_id"], []).append(
                {"ann_id": a["id"], "bbox": a["bbox"], "attributes": a.get("attributes", {})})
        images = {str(im["id"]): {"file_name": im["file_name"], "width": im["width"],
                                  "height": im["height"], "status": "unlabeled",
                                  "orig_labels": by_img.get(im["id"], [])}
                  for im in pool["images"]}
        if req.webhook_url:
            ls_call("POST", "/api/webhooks", json={
                "project": req.project_id, "url": req.webhook_url, "send_payload": True,
                "send_for_all_actions": False, "is_active": True,
                "actions": ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]})
        if req.reset and WORK.exists():
            for p in WORK.glob("train_r*"):
                shutil.rmtree(p, ignore_errors=True)
        st = {"created": now(), "dataset": req.dataset, "project_id": req.project_id,
              "seed": req.seed, "noise_seed": req.noise_seed, "n_injected_errors": n_err,
              "routing": req.routing, "init_weights": rel(weights), "current_weights": rel(weights),
              "model_version": "init", "round": 0, "rounds": [], "trainings": [], "images": images}
        save_state(st)
    log_event("init", dataset=req.dataset, project_id=req.project_id, pool=len(images),
              noise_seed=req.noise_seed, n_injected_errors=n_err, weights=str(weights))
    return {"pool": len(images), "n_injected_errors": n_err}


# ---------------- раунд: відбір + маршрутизація ----------------

def _predict(weights: str, paths: list[Path], conf: float) -> list[np.ndarray]:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / weights))
    out = []
    for i in range(0, len(paths), 16):
        for r in model.predict([str(p) for p in paths[i:i + 16]], conf=conf, iou=0.6,
                               max_det=300, verbose=False):
            xyxy = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()[:, None]
            out.append(np.hstack([xyxy, c]) if len(c) else np.zeros((0, 5)))
    return out


def _region(box_xywh, W: int, H: int, text: str, score: float | None = None) -> dict:
    x, y, w, h = (float(v) for v in box_xywh)
    r = {"from_name": "person_box", "to_name": "image", "type": "rectanglelabels",
         "original_width": W, "original_height": H, "meta": {"text": [text]},
         "value": {"x": 100 * x / W, "y": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H,
                   "rotation": 0, "rectanglelabels": ["person"]}}
    if score is not None:
        r["score"] = float(score)
    return r


@app.post("/rounds/select")
def select_round(req: SelectReq):
    if _train["running"]:
        raise HTTPException(409, "Іде навчання — дочекайтеся GET /train/status")
    st = load_state()
    unl = [i for i, im in st["images"].items() if im["status"] == "unlabeled"]
    if not unl:
        raise HTTPException(409, "Пул вичерпано")
    rng = random.Random(f"{st['seed']}-{st['round']}")
    k = min(req.k, len(unl))
    t0 = time.time()
    if req.strategy == "uncertainty":
        cand = rng.sample(unl, min(req.candidates, len(unl))) if req.candidates else unl
        preds = dict(zip(cand, _predict(st["current_weights"],
                                        [image_path(st, st["images"][i]) for i in cand], req.conf)))
        unc = uncertainty_scores(preds)
        chosen = sorted(cand, key=lambda i: -unc[i])[:k]
    else:
        chosen = rng.sample(unl, k)
        preds = dict(zip(chosen, _predict(st["current_weights"],
                                          [image_path(st, st["images"][i]) for i in chosen], req.conf)))
        unc = uncertainty_scores(preds)
    cfg = RoutingConfig(**st["routing"])
    rnd = st["round"] + 1
    tasks, counts = [], {"auto_accepted": 0, "auto_rejected": 0, "to_human": 0}
    decisions = {}
    for i in chosen:
        im = st["images"][i]
        labs = im["orig_labels"]
        lab_xywh = np.array([b["bbox"] for b in labs], dtype=float).reshape(-1, 4)
        route = route_one([b["ann_id"] for b in labs], lab_xywh, preds[i], cfg)
        dec = {"auto_reject": route.auto_reject, "label_decision": route.label_decision,
               "extra_boxes": route.extra_boxes.tolist(), "uncertainty": unc[i], "round": rnd}
        decisions[i] = dec
        if route.auto_reject:
            counts["auto_rejected"] += 1
        elif not route.to_human:
            counts["auto_accepted"] += 1
        else:
            counts["to_human"] += 1
            W, H = im["width"], im["height"]
            result = [_region(b["bbox"], W, H, "ПЕРЕВІРИТИ: модель не підтверджує" if d == "human"
                              else "модель підтверджує")
                      for b, d in zip(labs, route.label_decision)]
            result += [_region(b, W, H, "пропущена людина? (бачить лише модель)")
                       for b in route.extra_boxes]
            n_items = route.n_human_items
            tasks.append({
                "data": {"image": rel_url(image_path(st, im)), "image_id": int(i),
                         "dataset": st["dataset"], "file_name": im["file_name"], "mode": "routed",
                         "round": rnd, "n_human_items": n_items,
                         "hint": (f"Перевірте {n_items} позначених елемент(и): рамки «ПЕРЕВІРИТИ» і "
                                  "«пропущена людина?». Решту модель уже підтвердила. "
                                  "Потім вердикт 1/2/3.")},
                "predictions": [{"model_version": st["model_version"], "score": unc[i],
                                 "result": result}]})
    task_ids = {}
    if tasks:
        ls_call("POST", f"/api/projects/{st['project_id']}/import", json=tasks)
        want = {str(t["data"]["image_id"]) for t in tasks}
        page, seen, total = 1, 0, None
        while want - set(task_ids) and (total is None or seen < total):
            resp = ls_call("GET", f"/api/tasks?project={st['project_id']}&page={page}&page_size=500")
            items = resp.get("tasks", []) if isinstance(resp, dict) else resp
            total = resp.get("total", len(items)) if isinstance(resp, dict) else len(items)
            if not items:
                break
            seen += len(items)
            for t in items:
                iid = str(t["data"].get("image_id"))
                if iid in want and t["data"].get("round") == rnd:
                    task_ids[iid] = t["id"]
            page += 1
    with state() as st2:
        for i in chosen:
            im, dec = st2["images"][i], decisions[i]
            im["route"] = dec
            if dec["auto_reject"]:
                im.update(status="auto_rejected", final=[], decided_by="routing")
            elif "human" not in dec["label_decision"] and not dec["extra_boxes"]:
                im.update(status="auto_accepted", decided_by="routing",
                          final=[{"bbox": b["bbox"], "provenance": "consensus",
                                  "attributes": b["attributes"]} for b in im["orig_labels"]])
            else:
                im.update(status="pending_human", task_id=task_ids.get(i))
        st2["round"] = rnd
        st2["rounds"].append({"round": rnd, "ts": now(), "strategy": req.strategy, "k": k,
                              "model_version": st2["model_version"], **counts,
                              "seconds": round(time.time() - t0, 1),
                              "selected_unc_mean": float(np.mean([unc[i] for i in chosen]))})
    log_event("round_selected", round=rnd, strategy=req.strategy, images=[int(i) for i in chosen],
              **counts, missing_task_ids=len(tasks) - len(task_ids))
    return {"round": rnd, "selected": len(chosen), **counts,
            "label_studio": f"{os.environ.get('LABEL_STUDIO_PUBLIC_URL', LS().url)}/projects/{st['project_id']}"}


# ---------------- прийом анотацій ----------------

def _parse_annotation(ann: dict, W: int, H: int) -> tuple[list[dict], str | None]:
    attrs: dict[str, dict] = {}
    boxes, verdict = {}, None
    for r in ann.get("result", []):
        if r.get("type") == "rectanglelabels":
            v = r["value"]
            boxes[r["id"]] = [v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100]
        elif r.get("from_name") in ("pose", "visibility"):
            attrs.setdefault(r["id"], {})[r["from_name"]] = r["value"]["choices"][0]
        elif r.get("from_name") == "verdict":
            verdict = r["value"]["choices"][0]
    out = [{"bbox": b, "attributes": {"pose": attrs.get(rid, {}).get("pose", "unknown"),
                                      "visibility": attrs.get(rid, {}).get("visibility", "unknown")}}
           for rid, b in boxes.items()]
    return out, verdict


def _provenance(final: list[dict], im: dict) -> None:
    """consensus — рамка розмітки, яку підтвердила модель і людина лишила;"""
    labs = im["orig_labels"]
    dec = im.get("route", {}).get("label_decision", ["human"] * len(labs))
    src = [(b["bbox"], "consensus" if d == "auto_accept" else "original_dataset") for b, d in zip(labs, dec)]
    src += [(b, "model") for b in im.get("route", {}).get("extra_boxes", [])]
    used = set()
    for f in final:
        f["provenance"] = "annotator"
        if not src:
            continue
        ious = iou_xywh(np.array([f["bbox"]]), np.array([s[0] for s in src]))[0]
        for j in np.argsort(-ious):
            if ious[j] < MATCH_IOU:
                break
            if j not in used:
                used.add(int(j))
                f["provenance"] = src[j][1]
                break


def _apply_annotation(task: dict, ann: dict, source: str) -> dict | None:
    if ann.get("was_cancelled"):
        return None
    iid = str(task["data"].get("image_id"))
    with state() as st:
        im = st["images"].get(iid)
        if im is None:
            return None
        final, verdict = _parse_annotation(ann, im["width"], im["height"])
        if verdict == "no_person":
            final = []
        _provenance(final, im)
        im.update(status="human_done", final=final, decided_by="annotator", verdict=verdict,
                  lead_time_s=ann.get("lead_time"), annotation_id=ann.get("id"),
                  task_id=task.get("id"), annotator=ann.get("completed_by"))
        rec = {"image_id": int(iid), "task_id": task.get("id"), "annotation_id": ann.get("id"),
               "verdict": verdict, "lead_time_s": ann.get("lead_time"), "n_final": len(final),
               "n_orig": len(im["orig_labels"]), "round": im.get("route", {}).get("round"),
               "provenance": [f["provenance"] for f in final], "source": source}
    log_event("annotation", **rec)
    return rec


@app.post("/webhook/labelstudio")
def webhook(payload: dict):
    if payload.get("action") not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
        return {"ignored": payload.get("action")}
    rec = _apply_annotation(payload.get("task", {}), payload.get("annotation", {}), "webhook")
    return {"applied": rec is not None}


@app.post("/sync")
def sync():
    """Забрати анотації для всіх задач, що чекають людину (якщо вебхук не доходить)."""
    st = load_state()
    pending = [(i, im["task_id"]) for i, im in st["images"].items()
               if im["status"] == "pending_human" and im.get("task_id")]
    applied = 0
    for _, tid in pending:
        task = ls_call("GET", f"/api/tasks/{tid}")
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if anns:
            if _apply_annotation(task, anns[-1], "sync"):
                applied += 1
    return {"pending_checked": len(pending), "applied": applied}


# ---------------- навчання ----------------

def _write_yolo(st: dict, out: Path) -> int:
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in st["images"].values():
        if im["status"] not in LABELED:
            continue
        src = image_path(st, im)
        dst = img_dir / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        W, H = im["width"], im["height"]
        (lbl_dir / f"{dst.stem}.txt").write_text("\n".join(
            f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
            for x, y, w, h in (f["bbox"] for f in im["final"])))
        n += 1
    return n


def _train_job(req: TrainReq) -> None:
    import mlflow
    from ultralytics import YOLO

    try:
        st = load_state()
        r = st["round"]
        out = WORK / f"train_r{r}"
        n = _write_yolo(st, out)
        root, _, test_split = DATASETS[st["dataset"]]
        test = coco_split(st["dataset"], test_split)
        (out / "test.txt").write_text("\n".join(str(root / "images" / im["file_name"])
                                                for im in test["images"]))
        data_yaml = out / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({"path": str(out), "train": str(out / "images"),
                                             "val": str(out / "test.txt"), "names": {0: "person"}}))
        tracking.setup("orchestrator")
        t0 = time.time()
        with mlflow.start_run(run_name=f"orch_r{r}"):
            mlflow.log_params({"round": r, "n_train_images": n, "dataset": st["dataset"],
                               "init_weights": st["init_weights"], **req.model_dump(),
                               "git_commit": tracking.git_commit()})
            model = YOLO(str(ROOT / st["init_weights"]))
            model.train(data=str(data_yaml), epochs=req.epochs, imgsz=req.imgsz, batch=req.batch,
                        workers=2, seed=st["seed"], deterministic=True, val=False, plots=False,
                        verbose=False, project=str(out / "runs"), name="train", exist_ok=True)
            weights = Path(model.trainer.save_dir) / "weights" / "last.pt"
            m = YOLO(str(weights)).val(data=str(data_yaml), split="val", imgsz=req.imgsz,
                                       batch=req.batch, max_det=300, plots=False, verbose=False,
                                       project=str(out / "runs"), name="val", exist_ok=True)
            metrics = {"test_precision": float(m.box.mp), "test_recall": float(m.box.mr),
                       "test_mAP50": float(m.box.map50), "test_mAP50_95": float(m.box.map),
                       "train_seconds": time.time() - t0}
            mlflow.log_metrics(metrics)
        with state() as st2:
            st2["current_weights"] = rel(weights)
            st2["model_version"] = f"r{r}"
            st2["trainings"].append({"round": r, "ts": now(), "n_train_images": n,
                                     "weights": rel(weights), **metrics})
        log_event("trained", round=r, n_train_images=n, **metrics)
    except Exception as e:  # noqa: BLE001 — помилку показуємо в /train/status
        _train["error"] = repr(e)
        log_event("train_failed", error=repr(e))
    finally:
        _train.update(running=False, finished=now())


@app.post("/train")
def train(req: TrainReq, bg: BackgroundTasks):
    st = load_state()
    n = sum(im["status"] in LABELED for im in st["images"].values())
    if n == 0:
        raise HTTPException(409, "Ще немає прийнятих зображень")
    with _lock:
        if _train["running"]:
            raise HTTPException(409, "Навчання вже йде")
        _train.update(running=True, error=None, started=now(), finished=None)
    bg.add_task(_train_job, req)
    return {"started": True, "n_train_images": n}


@app.get("/train/status")
def train_status():
    st = load_state()
    return {**_train, "model_version": st["model_version"],
            "last": st["trainings"][-1] if st["trainings"] else None}


# ---------------- стан, метрики, експорт ----------------

@app.get("/metrics")
def metrics():
    st = load_state()
    ims = st["images"].values()
    status: dict[str, int] = {}
    prov: dict[str, int] = {}
    for im in ims:
        status[im["status"]] = status.get(im["status"], 0) + 1
        for f in im.get("final", []):
            prov[f["provenance"]] = prov.get(f["provenance"], 0) + 1
    lead = [im["lead_time_s"] for im in ims if im.get("lead_time_s") is not None]
    decided = sum(status.get(s, 0) for s in LABELED) + status.get("pending_human", 0)
    n_box = sum(prov.values())
    return {"dataset": st["dataset"], "round": st["round"], "model_version": st["model_version"],
            "pool": len(st["images"]), "status": status,
            "share_auto": ((status.get("auto_accepted", 0) + status.get("auto_rejected", 0)) / decided
                           if decided else None),
            "provenance_boxes": prov,
            "provenance_share": {k: v / n_box for k, v in prov.items()} if n_box else {},
            "human_tasks_done": len(lead),
            "lead_time_median_s": median(lead) if lead else None,
            "lead_time_total_h": sum(lead) / 3600 if lead else 0.0,
            "rounds": st["rounds"], "trainings": st["trainings"]}


@app.get("/images/{image_id}")
def image(image_id: int):
    im = load_state()["images"].get(str(image_id))
    if im is None:
        raise HTTPException(404, "Немає такого зображення")
    return im


@app.post("/export")
def export():
    """COCO з прийнятими зображеннями: атрибути шару 1 + provenance кожної рамки."""
    st = load_state()
    pose_ok, vis_ok = set(SCHEMA["attributes"]["pose"]), set(SCHEMA["attributes"]["visibility"])
    images, anns = [], []
    for i, im in sorted(st["images"].items(), key=lambda kv: int(kv[0])):
        if im["status"] not in LABELED:
            continue
        images.append({"id": int(i), "file_name": im["file_name"], "width": im["width"],
                       "height": im["height"], "decided_by": im.get("decided_by"),
                       "verdict": im.get("verdict"), "lead_time_s": im.get("lead_time_s"),
                       "round": im.get("route", {}).get("round")})
        for f in im["final"]:
            a = dict(f.get("attributes", {}))
            a["pose"] = a.get("pose") if a.get("pose") in pose_ok else "unknown"
            a["visibility"] = a.get("visibility") if a.get("visibility") in vis_ok else "unknown"
            a["provenance"] = f["provenance"]
            x, y, w, h = f["bbox"]
            anns.append({"id": len(anns) + 1, "image_id": int(i), "category_id": 1,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "attributes": a})
    coco = {"info": {"description": f"kursova layer-1 HITL export ({st['dataset']})",
                     "date_created": now(), "model_version": st["model_version"]},
            "categories": SCHEMA["categories"], "images": images, "annotations": anns}
    out = WORK / f"export_{st['dataset']}.json"
    out.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    log_event("export", path=str(out), images=len(images), annotations=len(anns))
    return {"path": str(out.relative_to(ROOT)), "images": len(images), "annotations": len(anns)}
