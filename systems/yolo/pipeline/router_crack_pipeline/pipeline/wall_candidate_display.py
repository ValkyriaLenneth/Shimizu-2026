"""Display-oriented grouping for wall-class downstream outputs."""

from __future__ import annotations

from typing import Any

from .result_merge import grade_level, ioa_min_xyxy, iou_xyxy


WALL_MODEL_LABELS = {
    "inner_wall": "内壁",
    "rc_wall": "RC壁",
}


def build_wall_candidate_display(
    raw_records: list[dict[str, Any]],
    iou_threshold: float = 0.50,
    ioa_threshold: float = 0.70,
    min_single_confidence: float = 0.05,
    max_single_groups_per_model: int = 1,
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
        group = _paired_group(inner[inner_index], rc[rc_index], len(groups), overlap)
        groups.append(group)
        display_detections.extend(group["display_detections"])

    remaining_inner = [
        record
        for index, record in enumerate(inner)
        if index not in used_inner and float(record.get("confidence") or 0.0) >= min_single_confidence
    ]
    remaining_rc = [
        record
        for index, record in enumerate(rc)
        if index not in used_rc and float(record.get("confidence") or 0.0) >= min_single_confidence
    ]

    for record in sorted(remaining_inner, key=_confidence, reverse=True)[:max_single_groups_per_model]:
        group = _single_group(record, len(groups))
        groups.append(group)
        display_detections.extend(group["display_detections"])

    for record in sorted(remaining_rc, key=_confidence, reverse=True)[:max_single_groups_per_model]:
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
            "has_grade_conflict": any(group["status"] == "grade_conflict" for group in groups),
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
) -> dict[str, Any]:
    inner_grade = grade_level(str(inner_record.get("damage_grade", "")))
    rc_grade = grade_level(str(rc_record.get("damage_grade", "")))
    candidates = [_candidate(inner_record), _candidate(rc_record)]
    same_grade = inner_grade == rc_grade
    if same_grade:
        representative = _best_record([inner_record, rc_record])
        representative_candidate = _candidate(representative)
        display = [
            _display_detection(
                representative,
                group_index,
                "same_grade_merged",
                str(representative_candidate["structure_type"]),
                inner_grade,
                "内壁模型与RC壁模型输出等级一致，取置信度更高的候选作为表示结果。",
                candidates,
            )
        ]
        status = "same_grade_merged"
    else:
        display = [
            _display_detection(
                inner_record,
                group_index,
                "grade_conflict",
                "内壁",
                inner_grade,
                "内壁模型与RC壁模型输出等级不同，保留内壁候选。",
                candidates,
            ),
            _display_detection(
                rc_record,
                group_index,
                "grade_conflict",
                "RC壁",
                rc_grade,
                "内壁模型与RC壁模型输出等级不同，保留RC壁候选。",
                candidates,
            ),
        ]
        status = "grade_conflict"

    return {
        "group_index": group_index,
        "status": status,
        "overlap": overlap,
        "candidates": candidates,
        "display_detections": display,
        "reason": display[0]["reason"],
    }


def _single_group(record: dict[str, Any], group_index: int) -> dict[str, Any]:
    structure_type = WALL_MODEL_LABELS.get(str(record.get("source_model")), str(record.get("source_model")))
    grade = grade_level(str(record.get("damage_grade", "")))
    candidate = _candidate(record)
    display = [
        _display_detection(
            record,
            group_index,
            "single_model",
            structure_type,
            grade,
            f"只有{structure_type}模型在该区域输出结果，因此作为单独候选显示。",
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
    return {
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


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=lambda record: float(record.get("confidence") or 0.0))


def _confidence(record: dict[str, Any]) -> float:
    return float(record.get("confidence") or 0.0)


def _box(record: dict[str, Any]) -> tuple[float, float, float, float]:
    values = record.get("bbox_xyxy") or [0, 0, 0, 0]
    return tuple(float(v) for v in values)  # type: ignore[return-value]
