"""Final product-display suppression for overlapping detections."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .result_merge import GRADE_RANK, area_xyxy, grade_level, ioa_min_xyxy, iou_xyxy


def suppress_overlapping_display_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.45,
    ioa_threshold: float = 0.70,
    cross_class_ioa_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reduce final UI clutter by suppressing overlapping display boxes.

    This runs after wall business rules have converted inner_wall/rc_wall
    candidates into ``壁-B/C/D`` display items. It intentionally works on the
    display records, not raw model outputs.
    """
    items = [deepcopy(det) for det in detections if _valid_box(det)]
    remaining = sorted(range(len(items)), key=lambda idx: _display_priority(items[idx]), reverse=True)
    kept_indices: list[int] = []
    suppressed: list[dict[str, Any]] = []

    while remaining:
        current_idx = remaining.pop(0)
        current = items[current_idx]
        kept_indices.append(current_idx)
        next_remaining: list[int] = []
        members: list[dict[str, Any]] = []

        for candidate_idx in remaining:
            candidate = items[candidate_idx]
            overlap = _display_overlap(current, candidate)
            if _should_suppress(current, candidate, overlap, iou_threshold, ioa_threshold, cross_class_ioa_threshold):
                suppressed_record = deepcopy(candidate)
                suppressed_record["suppressed_by_display_index"] = len(kept_indices) - 1
                suppressed_record["display_suppression_overlap"] = overlap
                suppressed.append(suppressed_record)
                members.append(_member(candidate, overlap))
            else:
                next_remaining.append(candidate_idx)

        if members:
            current.setdefault("display_merge_members", [])
            current["display_merge_members"].extend(members)
            current["display_suppressed_count"] = len(current["display_merge_members"])
            current["display_suppression_status"] = "kept_after_suppressing_overlap"
        remaining = next_remaining

    kept = [items[idx] for idx in kept_indices]
    return kept, suppressed


def _should_suppress(
    keeper: dict[str, Any],
    candidate: dict[str, Any],
    overlap: dict[str, float],
    iou_threshold: float,
    ioa_threshold: float,
    cross_class_ioa_threshold: float,
) -> bool:
    if overlap["iou"] >= iou_threshold:
        return True
    if _display_family(keeper) == _display_family(candidate):
        return overlap["ioa_min"] >= ioa_threshold
    return overlap["ioa_min"] >= cross_class_ioa_threshold and _grade_rank(keeper) >= _grade_rank(candidate)


def _display_priority(det: dict[str, Any]) -> tuple[int, float, float]:
    # Higher damage grade is a stronger user-facing signal than confidence.
    # Within the same grade, prefer the broader box to avoid keeping tiny
    # duplicate fragments over a more readable region.
    return (_grade_rank(det), _area(det), float(det.get("confidence") or 0.0))


def _grade_rank(det: dict[str, Any]) -> int:
    return GRADE_RANK.get(grade_level(str(det.get("damage_grade", ""))), 0)


def _area(det: dict[str, Any]) -> float:
    return area_xyxy(_box(det))


def _display_family(det: dict[str, Any]) -> str:
    structure = str(det.get("structure_type") or "")
    if structure in {"壁類", "壁类"}:
        return "wall"
    router_class = str(det.get("source_router_class") or "")
    if router_class in {"壁類", "壁类"}:
        return "wall"
    source_model = str(det.get("source_model") or "")
    if source_model in {"inner_wall", "rc_wall", "wall_merged"}:
        return "wall"
    return source_model or router_class or structure


def _display_overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_box = _box(left)
    right_box = _box(right)
    return {
        "iou": round(iou_xyxy(left_box, right_box), 4),
        "ioa_min": round(ioa_min_xyxy(left_box, right_box), 4),
    }


def _member(det: dict[str, Any], overlap: dict[str, float]) -> dict[str, Any]:
    return {
        "structure_type": det.get("structure_type"),
        "damage_grade": det.get("damage_grade"),
        "confidence": det.get("confidence"),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "source_model": det.get("source_model"),
        "source_router_class": det.get("source_router_class"),
        "overlap": overlap,
    }


def _valid_box(det: dict[str, Any]) -> bool:
    values = det.get("bbox_xyxy") or []
    if len(values) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in values]
    return x2 > x1 and y2 > y1


def _box(det: dict[str, Any]) -> tuple[float, float, float, float]:
    values = det.get("bbox_xyxy") or [0, 0, 0, 0]
    return tuple(float(v) for v in values)  # type: ignore[return-value]
