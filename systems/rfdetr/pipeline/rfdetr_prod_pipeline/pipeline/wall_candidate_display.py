"""Display-oriented grouping for wall-class downstream outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .result_merge import grade_level, ioa_min_xyxy, iou_xyxy


WALL_MODEL_LABELS = {
    "inner_wall": "内壁",
    "rc_wall": "RC壁",
}

WALL_DISPLAY_RULE = {
    ("B", "B"): "B",
    ("B", "C"): "C",
    ("B", "D"): "D",
    ("C", "B"): "B",
    ("C", "C"): "C",
    ("C", "D"): "D",
    ("D", "B"): "D",
    ("D", "C"): "D",
    ("D", "D"): "D",
}


def build_wall_candidate_display(
    raw_records: list[dict[str, Any]],
    iou_threshold: float = 0.50,
    ioa_threshold: float = 0.70,
    min_single_confidence: float = 0.05,
    min_single_confidence_by_model: dict[str, float] | None = None,
    max_single_groups_per_model: int = 4,
    use_union_bbox_for_pairs: bool = True,
) -> dict[str, Any]:
    """Build one-photo wall display groups from raw wall model outputs.

    The output is intentionally separate from the existing merged crack
    detections. It preserves Matsumoto's UX requirement: one image remains one
    image record, while wall subtype alternatives are displayed inside that
    record.
    """
    wall_records = [
        record
        for record in raw_records
        if record.get("source_router_class") == "壁类" and record.get("source_model") in WALL_MODEL_LABELS
    ]
    inner = [record for record in wall_records if record.get("source_model") == "inner_wall"]
    rc = [record for record in wall_records if record.get("source_model") == "rc_wall"]

    pairs = _pair_wall_records(inner, rc, iou_threshold, ioa_threshold)
    used_inner = {pair[0] for pair in pairs}
    used_rc = {pair[1] for pair in pairs}

    groups: list[dict[str, Any]] = []
    display_detections: list[dict[str, Any]] = []

    for inner_index, rc_index, overlap in pairs:
        group = _paired_group(
            inner[inner_index],
            rc[rc_index],
            len(groups),
            overlap,
            use_union_bbox_for_pairs=use_union_bbox_for_pairs,
        )
        groups.append(group)
        display_detections.extend(group["display_detections"])

    remaining_inner = [
        record
        for index, record in enumerate(inner)
        if index not in used_inner
        and float(record.get("confidence") or 0.0)
        >= _min_single_confidence(record, min_single_confidence, min_single_confidence_by_model)
    ]
    remaining_rc = [
        record
        for index, record in enumerate(rc)
        if index not in used_rc
        and float(record.get("confidence") or 0.0)
        >= _min_single_confidence(record, min_single_confidence, min_single_confidence_by_model)
    ]

    for record in sorted(remaining_inner, key=_single_priority, reverse=True)[:max_single_groups_per_model]:
        group = _single_group(record, len(groups))
        groups.append(group)
        display_detections.extend(group["display_detections"])

    for record in sorted(remaining_rc, key=_single_priority, reverse=True)[:max_single_groups_per_model]:
        group = _single_group(record, len(groups))
        groups.append(group)
        display_detections.extend(group["display_detections"])

    status_counts: dict[str, int] = {}
    for group in groups:
        status = str(group["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "groups": groups,
        "display_detections": display_detections,
        "summary": {
            "groups": len(groups),
            "display_detections": len(display_detections),
            "status_counts": status_counts,
            "has_grade_conflict": False,
            "display_rule": "wall_single_display_inner_wall_x_rc_wall_matrix",
        },
    }


def _pair_wall_records(
    inner: list[dict[str, Any]],
    rc: list[dict[str, Any]],
    iou_threshold: float,
    ioa_threshold: float,
) -> list[tuple[int, int, dict[str, float]]]:
    candidates = []
    for inner_index, inner_record in enumerate(inner):
        inner_box = _box(inner_record)
        for rc_index, rc_record in enumerate(rc):
            rc_box = _box(rc_record)
            iou = iou_xyxy(inner_box, rc_box)
            ioa = ioa_min_xyxy(inner_box, rc_box)
            if iou < iou_threshold and ioa < ioa_threshold:
                continue
            candidates.append((max(iou, ioa), inner_index, rc_index, {"iou": round(iou, 4), "ioa_min": round(ioa, 4)}))

    candidates.sort(reverse=True, key=lambda item: item[0])
    used_inner: set[int] = set()
    used_rc: set[int] = set()
    pairs: list[tuple[int, int, dict[str, float]]] = []
    for _, inner_index, rc_index, overlap in candidates:
        if inner_index in used_inner or rc_index in used_rc:
            continue
        used_inner.add(inner_index)
        used_rc.add(rc_index)
        pairs.append((inner_index, rc_index, overlap))
    return pairs


def _paired_group(
    inner_record: dict[str, Any],
    rc_record: dict[str, Any],
    group_index: int,
    overlap: dict[str, float],
    use_union_bbox_for_pairs: bool = True,
) -> dict[str, Any]:
    inner_grade = grade_level(str(inner_record.get("damage_grade", "")))
    rc_grade = grade_level(str(rc_record.get("damage_grade", "")))
    candidates = [_candidate(inner_record), _candidate(rc_record)]
    display_grade = wall_display_grade(inner_grade, rc_grade)
    representative = representative_record_for_display(inner_record, rc_record, display_grade)
    display_record = (
        display_record_for_pair(inner_record, rc_record, representative)
        if use_union_bbox_for_pairs
        else representative
    )
    display = [
        _display_detection(
            display_record,
            group_index,
            "wall_rule_merged",
            "壁類",
            f"壁-{display_grade}",
            wall_display_reason(inner_grade, rc_grade, display_grade),
            candidates,
        )
    ]
    status = "wall_rule_merged"

    return {
        "group_index": group_index,
        "status": status,
        "overlap": overlap,
        "candidates": candidates,
        "display_detections": display,
        "reason": display[0]["reason"],
    }


def _single_group(record: dict[str, Any], group_index: int) -> dict[str, Any]:
    grade = grade_level(str(record.get("damage_grade", "")))
    candidate = _candidate(record)
    display = [
        _display_detection(
            record,
            group_index,
            "single_model",
            "壁類",
            f"壁-{grade}",
            "壁類 router 区域只得到一个壁模型候选，因此按该候选等级作为 PC 显示结果。",
            [candidate],
        )
    ]
    return {
        "group_index": group_index,
        "status": "single_model",
        "overlap": None,
        "candidates": [candidate],
        "display_detections": display,
        "reason": display[0]["reason"],
    }


def _candidate(record: dict[str, Any]) -> dict[str, Any]:
    source_model = str(record.get("source_model"))
    return {
        "structure_type": WALL_MODEL_LABELS.get(source_model, source_model),
        "source_model": source_model,
        "damage_grade": grade_level(str(record.get("damage_grade", ""))),
        "raw_damage_grade": record.get("damage_grade"),
        "confidence": record.get("confidence"),
        "bbox_xyxy": record.get("bbox_xyxy"),
    }


def _display_detection(
    record: dict[str, Any],
    group_index: int,
    status: str,
    structure_type: str,
    grade: str,
    reason: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    output = {
        "group_index": group_index,
        "status": status,
        "structure_type": structure_type,
        "damage_grade": grade,
        "raw_damage_grade": record.get("damage_grade"),
        "confidence": record.get("confidence"),
        "bbox_xyxy": record.get("bbox_xyxy"),
        "source_model": record.get("source_model"),
        "source_router_class": record.get("source_router_class"),
        "reason": reason,
        "candidates": candidates,
    }
    if "display_bbox_source" in record:
        output["display_bbox_source"] = record["display_bbox_source"]
    return output


def wall_display_grade(inner_grade: str, rc_grade: str) -> str:
    return WALL_DISPLAY_RULE.get((inner_grade, rc_grade), higher_grade(inner_grade, rc_grade))


def representative_record_for_display(
    inner_record: dict[str, Any],
    rc_record: dict[str, Any],
    display_grade: str,
) -> dict[str, Any]:
    inner_grade = grade_level(str(inner_record.get("damage_grade", "")))
    rc_grade = grade_level(str(rc_record.get("damage_grade", "")))
    if rc_grade == display_grade:
        return rc_record
    if inner_grade == display_grade:
        return inner_record
    return rc_record


def display_record_for_pair(
    inner_record: dict[str, Any],
    rc_record: dict[str, Any],
    representative: dict[str, Any],
) -> dict[str, Any]:
    """Keep business grade provenance but use the paired wall geometry.

    A low-confidence/high-grade candidate can correctly decide the final wall
    grade, but its box is often a small fragment inside a better wall candidate.
    The customer-facing rectangle should cover the paired wall evidence rather
    than shrink to that fragment.
    """
    record = deepcopy(representative)
    inner_box = _box(inner_record)
    rc_box = _box(rc_record)
    record["bbox_xyxy"] = [
        round(min(inner_box[0], rc_box[0]), 3),
        round(min(inner_box[1], rc_box[1]), 3),
        round(max(inner_box[2], rc_box[2]), 3),
        round(max(inner_box[3], rc_box[3]), 3),
    ]
    record["display_bbox_source"] = "paired_wall_union"
    return record


def wall_display_reason(inner_grade: str, rc_grade: str, display_grade: str) -> str:
    if (inner_grade, rc_grade) == ("C", "B"):
        return "内壁=C、RC壁=B の例外ルールにより、PC上は 壁-B として表示します。"
    if (inner_grade, rc_grade) == ("D", "C"):
        return "内壁=D、RC壁=C の組み合わせは、PC上は 壁-D として表示します。"
    return f"内壁={inner_grade}、RC壁={rc_grade} の組み合わせ表に従い、PC上は 壁-{display_grade} として表示します。"


def higher_grade(left: str, right: str) -> str:
    rank = {"B": 1, "C": 2, "D": 3}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _confidence(record: dict[str, Any]) -> float:
    return float(record.get("confidence") or 0.0)


def _single_priority(record: dict[str, Any]) -> tuple[float, float]:
    box = _box(record)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return (area, _confidence(record))


def _min_single_confidence(
    record: dict[str, Any],
    default_threshold: float,
    per_model_thresholds: dict[str, float] | None,
) -> float:
    if not per_model_thresholds:
        return default_threshold
    model = str(record.get("source_model") or "")
    value = per_model_thresholds.get(model)
    return float(value) if value is not None else default_threshold


def _box(record: dict[str, Any]) -> tuple[float, float, float, float]:
    values = record.get("bbox_xyxy") or [0, 0, 0, 0]
    return tuple(float(v) for v in values)  # type: ignore[return-value]
