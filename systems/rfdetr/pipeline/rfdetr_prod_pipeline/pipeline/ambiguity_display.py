"""Cross-class ambiguity display groups.

When a fallback detection (e.g. RC柱) spatially overlaps a main-branch
detection of a different router class (e.g. 壁类), the two answers must be
shown to the auditor as *coexisting candidates*, never merged. Different
classes use different damage-grade rubrics; silently picking one would
corrupt the safety-critical semantics.

This module identifies such cross-class overlaps after merging and groups
them into ``ambiguous_class_candidate`` records that the display layer
exposes alongside (and instead of) the regular per-class outputs.
"""

from __future__ import annotations

from typing import Any

from .result_merge import Detection, grade_level, iou_xyxy

CLASS_DISPLAY_LABEL = {
    "天井": "天井",
    "壁类": "壁类",
    "RC柱": "RC柱",
}


def build_ambiguity_candidate_groups(
    merged: list[Detection],
    iou_threshold: float = 0.50,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Identify cross-class overlapping detections and group them.

    Returns ``(groups, used_indices)`` where ``used_indices`` are the indices
    of ``merged`` consumed by some ambiguity group; the caller is expected
    to skip these detections in the regular display path so they are not
    double-rendered.

    A detection participates in at most one group. Grouping is performed
    greedily ordered by confidence so the highest-confidence anchor seeds
    each group.
    """
    n = len(merged)
    used: set[int] = set()
    groups: list[dict[str, Any]] = []

    order = sorted(range(n), key=lambda i: -merged[i].confidence)
    for anchor_idx in order:
        if anchor_idx in used:
            continue
        anchor = merged[anchor_idx]
        partner_indices: list[int] = []
        for j in order:
            if j == anchor_idx or j in used:
                continue
            other = merged[j]
            if other.source_router_class == anchor.source_router_class:
                continue
            if iou_xyxy(anchor.xyxy, other.xyxy) >= iou_threshold:
                partner_indices.append(j)
        if not partner_indices:
            continue
        members = [anchor] + [merged[j] for j in partner_indices]
        groups.append(_build_group(group_index=len(groups), members=members))
        used.add(anchor_idx)
        used.update(partner_indices)

    return groups, used


def _build_group(group_index: int, members: list[Detection]) -> dict[str, Any]:
    candidates = [_candidate(det) for det in members]
    classes_present = sorted({det.source_router_class for det in members})
    reason = (
        "router 判定与兜底候选类别不一致，"
        f"同位置存在 {' / '.join(CLASS_DISPLAY_LABEL.get(c, c) for c in classes_present)} 两种解读，"
        "请人工确认实际构件类别。"
    )
    bbox = _representative_bbox(members)
    display_detections = [
        {
            "group_index": group_index,
            "status": "ambiguous_class_candidate",
            "structure_type": CLASS_DISPLAY_LABEL.get(cand["source_router_class"], cand["source_router_class"]),
            "damage_grade": cand["damage_grade"],
            "raw_damage_grade": cand["raw_damage_grade"],
            "confidence": cand["confidence"],
            "bbox_xyxy": cand["bbox_xyxy"],
            "source_model": cand["source_model"],
            "source_router_class": cand["source_router_class"],
            "reason": reason,
            "candidates": candidates,
        }
        for cand in candidates
    ]
    return {
        "group_index": group_index,
        "status": "ambiguous_class_candidate",
        "overlap": _pairwise_overlap_summary(members),
        "candidates": candidates,
        "display_detections": display_detections,
        "reason": reason,
        "bbox_xyxy": bbox,
        "classes": classes_present,
    }


def _candidate(det: Detection) -> dict[str, Any]:
    if det.source_router_class in {"壁类", "壁類"}:
        grade = grade_level(str(det.grade))
        return {
            "structure_type": "壁類",
            "source_model": "wall",
            "raw_source_model": det.source_model,
            "source_router_class": "壁类",
            "damage_grade": f"壁-{grade}",
            "raw_damage_grade": det.grade,
            "confidence": float(det.confidence),
            "bbox_xyxy": [round(float(v), 3) for v in det.xyxy],
        }
    return {
        "structure_type": CLASS_DISPLAY_LABEL.get(det.source_router_class, det.source_router_class),
        "source_model": det.source_model,
        "source_router_class": det.source_router_class,
        "damage_grade": grade_level(str(det.grade)),
        "raw_damage_grade": det.grade,
        "confidence": float(det.confidence),
        "bbox_xyxy": [round(float(v), 3) for v in det.xyxy],
    }


def _representative_bbox(members: list[Detection]) -> list[float]:
    x1 = min(d.xyxy[0] for d in members)
    y1 = min(d.xyxy[1] for d in members)
    x2 = max(d.xyxy[2] for d in members)
    y2 = max(d.xyxy[3] for d in members)
    return [round(float(v), 3) for v in (x1, y1, x2, y2)]


def _pairwise_overlap_summary(members: list[Detection]) -> dict[str, float]:
    pairs: list[float] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            pairs.append(iou_xyxy(members[i].xyxy, members[j].xyxy))
    if not pairs:
        return {"min_iou": 0.0, "max_iou": 0.0}
    return {"min_iou": round(min(pairs), 4), "max_iou": round(max(pairs), 4)}
