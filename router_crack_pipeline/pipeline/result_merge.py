"""Merge downstream crack/damage detections after mapping them to original coordinates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    grade: str
    source_model: str
    source_router_class: str


GRADE_RANK = {"B": 1, "C": 2, "D": 3}


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def area_xyxy(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def ioa_min_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    denom = min(area_xyxy(a), area_xyxy(b))
    return 0.0 if denom <= 0 else intersection_xyxy(a, b) / denom


def ioa_over_first_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    denom = area_xyxy(a)
    return 0.0 if denom <= 0 else intersection_xyxy(a, b) / denom


def center_in_xyxy(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def grade_level(value: str) -> str:
    if value in GRADE_RANK:
        return value
    match = re.search(r"程度([BCD])", value)
    if match:
        return match.group(1)
    match = re.search(r"\b([BCD])\b", value)
    return match.group(1) if match else str(value)


def detection_priority(det: Detection, prefer_higher_grade: bool) -> tuple[float, float]:
    grade_score = float(GRADE_RANK.get(grade_level(det.grade), 0)) if prefer_higher_grade else 0.0
    return (grade_score, det.confidence)


def nms_detections(
    detections: Iterable[Detection],
    same_grade_iou_threshold: float = 0.50,
    cross_grade_iou_threshold: float = 0.60,
    prefer_higher_grade: bool = True,
) -> list[Detection]:
    """Conflict-aware NMS for downstream crack/damage detections."""
    remaining = sorted(detections, key=lambda d: detection_priority(d, prefer_higher_grade), reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        next_remaining = []
        for candidate in remaining:
            threshold = same_grade_iou_threshold if candidate.grade == current.grade else cross_grade_iou_threshold
            if iou_xyxy(candidate.xyxy, current.xyxy) <= threshold:
                next_remaining.append(candidate)
        remaining = next_remaining
    return kept


def prod_like_merge_detections(
    detections: Iterable[Detection],
    same_model_ioa_threshold: float = 0.70,
    cross_model_iou_threshold: float = 0.55,
) -> list[Detection]:
    """Merge detections using the previous prod API's postprocess semantics.

    Step 1: within each source model, higher damage grade suppresses lower/equal
    grade boxes when IoA over the smaller box is high.
    Step 2: across source models, overlapping boxes are clustered by IoU and
    coordinates are confidence-weighted. The highest grade/confidence detection
    supplies the final grade and source_model.
    """
    return _merge_across_models(
        _merge_levels_within_model(list(detections), same_model_ioa_threshold),
        cross_model_iou_threshold,
    )


def _merge_levels_within_model(detections: list[Detection], ioa_threshold: float) -> list[Detection]:
    grouped: dict[str, list[Detection]] = {}
    for det in detections:
        grouped.setdefault(det.source_model, []).append(det)

    merged: list[Detection] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda d: (GRADE_RANK.get(grade_level(d.grade), 0), d.confidence), reverse=True)
        kept: list[Detection] = []
        for current in ordered:
            current_rank = GRADE_RANK.get(grade_level(current.grade), 0)
            suppress = False
            for previous in kept:
                previous_rank = GRADE_RANK.get(grade_level(previous.grade), 0)
                if previous_rank >= current_rank and ioa_min_xyxy(previous.xyxy, current.xyxy) >= ioa_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(current)
        merged.extend(kept)
    return merged


def _merge_across_models(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    remaining = sorted(detections, key=lambda d: d.confidence, reverse=True)
    clusters: list[list[Detection]] = []
    for det in remaining:
        for cluster in clusters:
            if any(iou_xyxy(det.xyxy, other.xyxy) >= iou_threshold for other in cluster):
                cluster.append(det)
                break
        else:
            clusters.append([det])

    merged: list[Detection] = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        weights = [max(1e-6, det.confidence) for det in cluster]
        total = sum(weights)
        coords = []
        for index in range(4):
            coords.append(sum(det.xyxy[index] * weight for det, weight in zip(cluster, weights)) / total)
        best = max(cluster, key=lambda d: (GRADE_RANK.get(grade_level(d.grade), 0), d.confidence))
        merged.append(
            Detection(
                xyxy=tuple(coords),  # type: ignore[arg-type]
                confidence=max(det.confidence for det in cluster),
                grade=best.grade,
                source_model=best.source_model,
                source_router_class=best.source_router_class,
            )
        )
    return merged
