"""Маршрутизація рамок шару 1 з фіксованими порогами: auto_accept / auto_reject / human.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from oracle import iou_xywh

MISS_CATCH_IOU = 0.3


@dataclass
class RoutingConfig:
    conf_accept: float = 0.5
    iou_accept: float = 0.5
    conf_extra: float = 0.5
    conf_reject: float = 0.1


@dataclass
class ImageRoute:
    auto_reject: bool
    label_ids: list[int]                  # id анотацій розмітки
    label_decision: list[str]             # auto_accept | human для кожної
    extra_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # xywh -> human

    @property
    def to_human(self) -> bool:
        return (not self.auto_reject) and ("human" in self.label_decision or len(self.extra_boxes) > 0)

    @property
    def n_human_items(self) -> int:
        return self.label_decision.count("human") + len(self.extra_boxes)


def _xywh(p: np.ndarray) -> np.ndarray:
    b = p[:, :4].copy()
    b[:, 2:] -= b[:, :2]
    return b


def route_one(label_ids: list[int], labels_xywh: np.ndarray, pred_xyxyc: np.ndarray,
              cfg: RoutingConfig) -> ImageRoute:
    pred = pred_xyxyc if len(pred_xyxyc) else np.zeros((0, 5))
    if len(labels_xywh) == 0 and not (pred[:, 4] >= cfg.conf_reject).any():
        return ImageRoute(True, [], [])
    strong = pred[pred[:, 4] >= min(cfg.conf_accept, cfg.conf_extra)]
    decision = ["human"] * len(labels_xywh)
    matched_p: set[int] = set()
    if len(strong) and len(labels_xywh):
        acc = strong[:, 4] >= cfg.conf_accept
        iou = iou_xywh(labels_xywh, _xywh(strong))
        iou[:, ~acc] = -1
        while iou.size and iou.max() >= cfg.iou_accept:
            i, j = np.unravel_index(iou.argmax(), iou.shape)
            decision[i] = "auto_accept"
            matched_p.add(j)
            iou[i, :] = -1
            iou[:, j] = -1
    extra_idx = [j for j in range(len(strong))
                 if j not in matched_p and strong[j, 4] >= cfg.conf_extra]
    extra = _xywh(strong[extra_idx]) if extra_idx else np.zeros((0, 4))
    if len(extra) and "human" in decision:
        hum = labels_xywh[[k for k, d in enumerate(decision) if d == "human"]]
        keep = iou_xywh(extra, hum).max(axis=1) < MISS_CATCH_IOU
        extra = extra[keep]
    return ImageRoute(False, list(label_ids), decision, extra)


def route_all(labels: dict[int, tuple[list[int], np.ndarray]], preds: dict[int, np.ndarray],
              cfg: RoutingConfig) -> dict[int, ImageRoute]:
    empty = ([], np.zeros((0, 4)))
    return {i: route_one(*labels.get(i, empty), preds.get(i, np.zeros((0, 5))), cfg) for i in preds}


def evaluate(routes: dict[int, ImageRoute], errors: list[dict], truth_by_img: dict[int, np.ndarray],
             labels: dict[int, tuple[list[int], np.ndarray]]) -> dict:
    n_img = len(routes)
    n_lab = sum(len(r.label_ids) for r in routes.values())
    n_lab_auto = sum(r.label_decision.count("auto_accept") for r in routes.values())
    n_items = sum(r.n_human_items for r in routes.values())

    caught = {"miss": 0, "shift": 0, "spurious": 0}
    total = {"miss": 0, "shift": 0, "spurious": 0}
    leaked_auto_reject = 0
    for e in errors:
        r = routes.get(e["image_id"])
        if r is None:
            continue
        total[e["type"]] += 1
        if e["type"] in ("shift", "spurious"):
            dec = dict(zip(r.label_ids, r.label_decision))
            if dec.get(e["noisy_ann_id"]) == "human":
                caught[e["type"]] += 1
        else:  # miss
            if r.auto_reject:
                leaked_auto_reject += 1
                continue
            t = np.array([e["true_bbox"]])
            near_extra = len(r.extra_boxes) and iou_xywh(t, r.extra_boxes).max() >= MISS_CATCH_IOU
            lab_ids, lab_boxes = labels.get(e["image_id"], ([], np.zeros((0, 4))))
            hum = [k for k, d in enumerate(r.label_decision) if d == "human"]
            near_label = len(hum) and iou_xywh(t, lab_boxes[hum]).max() >= MISS_CATCH_IOU
            if near_extra or near_label:
                caught["miss"] += 1

    n_err = sum(total.values())
    persons_rejected = sum(len(truth_by_img.get(i, [])) for i, r in routes.items() if r.auto_reject)
    persons_total = sum(len(truth_by_img.get(i, [])) for i in routes)
    return {
        "n_images": n_img,
        "share_images_auto_reject": sum(r.auto_reject for r in routes.values()) / n_img,
        "share_images_to_human": sum(r.to_human for r in routes.values()) / n_img,
        "share_images_fully_auto": sum(not r.to_human for r in routes.values()) / n_img,
        "share_label_boxes_auto_accept": n_lab_auto / n_lab if n_lab else 0.0,
        "human_items": n_items,
        "human_items_per_label_box": n_items / n_lab if n_lab else 0.0,
        "errors_total": n_err,
        "errors_to_human_frac": sum(caught.values()) / n_err if n_err else 1.0,
        **{f"errors_to_human_frac_{k}": caught[k] / total[k] if total[k] else 1.0 for k in total},
        "errors_leaked": n_err - sum(caught.values()),
        "miss_leaked_in_auto_reject": leaked_auto_reject,
        "person_recall_outside_auto_reject": 1 - persons_rejected / persons_total if persons_total else 1.0,
    }


def grid(cfg_values: dict[str, list[float]]):
    keys = list(cfg_values)
    for combo in np.array(np.meshgrid(*[cfg_values[k] for k in keys])).T.reshape(-1, len(keys)):
        yield RoutingConfig(**{k: float(v) for k, v in zip(keys, combo)})


__all__ = ["RoutingConfig", "ImageRoute", "route_one", "route_all", "evaluate", "grid", "asdict"]
