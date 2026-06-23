"""Engineering fallback policy for router → downstream pipeline.

Plans the set of detection tasks executed for a single image. Fallback tasks
are triggered only when the router output carries a specific suspicious
signal, never as a blanket "always-on" rescue. The supported triggers are:

- **A. Morphology** — a router 壁类 box that is unusually small and
  low-confidence (likely a misidentified RC柱).
- **B. Main detector dropout** — after the main detectors have run for a
  router region, the maximum confidence inside the region is below a
  threshold. This is the strongest evidence that the router routed to the
  wrong specialist and is the only *dynamic* trigger (planned after main
  inference). It is symmetric: low rc_column confidence in a router RC柱
  region brings in the wall models, and vice versa.
- **C. Parallel walls** — two or more router 壁类 boxes side-by-side with
  near-equal confidence and tall aspect ratio (likely a pair of columns).
- **D. Low-confidence router** — keep the previous behaviour: when the
  router only produced low-confidence boxes, run all detectors.
- **E. Empty router** — same, when the router produced nothing.

Tasks carry the *missing* class as ``source_router_class`` so the display
layer routes their outputs through the correct (rescue) branch rather than
the present-but-wrong branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .region_view import padded_xyxy
from .result_merge import area_xyxy

DETECTOR_NAME_BY_ROUTER_CLASS = {
    "天井": ["ceiling"],
    "壁类": ["inner_wall", "rc_wall"],
    "RC柱": ["rc_column"],
}

ROUTER_CLASS_BY_DETECTOR_NAME = {
    detector_name: router_class
    for router_class, names in DETECTOR_NAME_BY_ROUTER_CLASS.items()
    for detector_name in names
}

# For Trigger B, what fallback detectors to run when a router region's
# main detectors collectively fall silent.
SISTER_DETECTORS_BY_ROUTER_CLASS = {
    "天井": [],
    "壁类": ["rc_column"],
    "RC柱": ["inner_wall", "rc_wall"],
}

SISTER_ROUTER_CLASS = {
    "壁类": "RC柱",
    "RC柱": "壁类",
}


@dataclass(frozen=True)
class Task:
    detector_name: str
    filter_box: tuple[float, float, float, float]
    source_router_class: str
    router_region_indices: tuple[int, ...]
    is_fallback: bool
    fallback_reason: str
    min_confidence: float
    extra_meta: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


def plan_main_tasks(
    router_detections: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    region_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    available_detector_names: Iterable[str],
) -> list[Task]:
    """Always-on tasks driven directly by router output."""
    available = set(available_detector_names)
    main_padding = float(region_cfg.get("region_padding_ratio", 0.10))
    main_min_conf = float(fallback_cfg.get("main_min_confidence", 0.0))
    tasks: list[Task] = []
    for index, det in enumerate(router_detections):
        router_class = str(det["class_name"])
        filter_box = tuple(
            float(v) for v in padded_xyxy(det["bbox_xyxy"], image_shape, padding_ratio=main_padding)
        )
        for detector_name in DETECTOR_NAME_BY_ROUTER_CLASS.get(router_class, []):
            if detector_name not in available:
                continue
            tasks.append(
                Task(
                    detector_name=detector_name,
                    filter_box=filter_box,
                    source_router_class=router_class,
                    router_region_indices=(index,),
                    is_fallback=False,
                    fallback_reason="",
                    min_confidence=main_min_conf,
                    extra_meta=(("router_confidence", float(det.get("confidence", 0.0))),),
                )
            )
    return tasks


def plan_static_fallback_tasks(
    router_detections: list[dict[str, Any]],
    router_status: str,
    image_shape: tuple[int, ...],
    region_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    available_detector_names: Iterable[str],
) -> list[Task]:
    """Triggers that depend on router output alone (A, C, plus low-conf/empty)."""
    if not bool(fallback_cfg.get("enabled", False)):
        return []
    available = set(available_detector_names)
    tasks: list[Task] = []
    tasks.extend(_morphology_trigger_tasks(router_detections, image_shape, fallback_cfg, available))
    tasks.extend(_parallel_walls_trigger_tasks(router_detections, image_shape, fallback_cfg, available))
    tasks.extend(_low_confidence_trigger_tasks(router_detections, router_status, image_shape, fallback_cfg, available))
    tasks.extend(_empty_router_trigger_tasks(router_detections, image_shape, fallback_cfg, available))
    return tasks


def plan_dynamic_fallback_tasks(
    router_detections: list[dict[str, Any]],
    max_main_conf_by_region: dict[int, float],
    image_shape: tuple[int, ...],
    region_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
    available_detector_names: Iterable[str],
) -> list[Task]:
    """Trigger B: per-region main-detector dropout."""
    if not bool(fallback_cfg.get("enabled", False)):
        return []
    cfg = fallback_cfg.get("trigger_main_dropout", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []
    threshold = float(cfg.get("max_main_confidence", 0.15))
    min_conf = float(cfg.get("rescue_min_confidence", 0.20))
    padding = float(cfg.get("region_padding_ratio", 0.05))
    available = set(available_detector_names)
    tasks: list[Task] = []
    for index, det in enumerate(router_detections):
        router_class = str(det["class_name"])
        sister_router_class = SISTER_ROUTER_CLASS.get(router_class)
        sister_detectors = SISTER_DETECTORS_BY_ROUTER_CLASS.get(router_class, [])
        if not sister_router_class or not sister_detectors:
            continue
        max_main_conf = float(max_main_conf_by_region.get(index, 0.0))
        if max_main_conf >= threshold:
            continue
        filter_box = tuple(
            float(v) for v in padded_xyxy(det["bbox_xyxy"], image_shape, padding_ratio=padding)
        )
        for detector_name in sister_detectors:
            if detector_name not in available:
                continue
            tasks.append(
                Task(
                    detector_name=detector_name,
                    filter_box=filter_box,
                    source_router_class=sister_router_class,
                    router_region_indices=(index,),
                    is_fallback=True,
                    fallback_reason=f"main_dropout:{router_class}_region_max_conf={round(max_main_conf,3)}",
                    min_confidence=min_conf,
                )
            )
    return tasks


def _morphology_trigger_tasks(
    router_detections: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    fallback_cfg: dict[str, Any],
    available: set[str],
) -> list[Task]:
    cfg = fallback_cfg.get("trigger_morphology", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []
    area_max = float(cfg.get("wall_area_ratio_max", 0.05))
    conf_max = float(cfg.get("wall_confidence_max", 0.50))
    min_conf = float(cfg.get("rescue_min_confidence", 0.20))
    padding = float(cfg.get("region_padding_ratio", 0.05))
    tasks: list[Task] = []
    for index, det in enumerate(router_detections):
        if str(det.get("class_name")) != "壁类":
            continue
        if float(det.get("area_ratio", 1.0)) > area_max:
            continue
        if float(det.get("confidence", 1.0)) > conf_max:
            continue
        filter_box = tuple(
            float(v) for v in padded_xyxy(det["bbox_xyxy"], image_shape, padding_ratio=padding)
        )
        for detector_name in SISTER_DETECTORS_BY_ROUTER_CLASS.get("壁类", []):
            if detector_name not in available:
                continue
            tasks.append(
                Task(
                    detector_name=detector_name,
                    filter_box=filter_box,
                    source_router_class="RC柱",
                    router_region_indices=(index,),
                    is_fallback=True,
                    fallback_reason=(
                        f"morphology:壁类_area={round(float(det.get('area_ratio', 0.0)), 3)}_"
                        f"conf={round(float(det.get('confidence', 0.0)), 3)}"
                    ),
                    min_confidence=min_conf,
                )
            )
    return tasks


def _parallel_walls_trigger_tasks(
    router_detections: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    fallback_cfg: dict[str, Any],
    available: set[str],
) -> list[Task]:
    cfg = fallback_cfg.get("trigger_parallel_walls", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []
    min_count = int(cfg.get("min_wall_count", 2))
    conf_ratio_max = float(cfg.get("conf_ratio_max", 1.20))
    aspect_min = float(cfg.get("aspect_ratio_min", 1.50))
    min_conf = float(cfg.get("rescue_min_confidence", 0.20))
    padding = float(cfg.get("region_padding_ratio", 0.05))
    wall_dets = [
        (index, det)
        for index, det in enumerate(router_detections)
        if str(det.get("class_name")) == "壁类"
    ]
    if len(wall_dets) < min_count:
        return []
    confs = [float(det.get("confidence", 0.0)) for _, det in wall_dets]
    aspects = [_aspect_ratio(det.get("bbox_xyxy")) for _, det in wall_dets]
    if any(a < aspect_min for a in aspects):
        return []
    if min(confs) <= 0:
        return []
    if max(confs) / min(confs) > conf_ratio_max:
        return []
    indices = tuple(i for i, _ in wall_dets)
    union = _union_box([_xyxy(det.get("bbox_xyxy")) for _, det in wall_dets], image_shape, padding)
    if union is None:
        return []
    tasks: list[Task] = []
    for detector_name in SISTER_DETECTORS_BY_ROUTER_CLASS.get("壁类", []):
        if detector_name not in available:
            continue
        tasks.append(
            Task(
                detector_name=detector_name,
                filter_box=union,
                source_router_class="RC柱",
                router_region_indices=indices,
                is_fallback=True,
                fallback_reason=(
                    f"parallel_walls:n={len(wall_dets)}_"
                    f"conf_ratio={round(max(confs) / min(confs), 3)}_"
                    f"min_aspect={round(min(aspects), 3)}"
                ),
                min_confidence=min_conf,
            )
        )
    return tasks


def _low_confidence_trigger_tasks(
    router_detections: list[dict[str, Any]],
    router_status: str,
    image_shape: tuple[int, ...],
    fallback_cfg: dict[str, Any],
    available: set[str],
) -> list[Task]:
    cfg = fallback_cfg.get("trigger_low_confidence_router", {}) or fallback_cfg.get("low_confidence", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []
    if router_status != "low_confidence":
        return []
    min_conf = float(cfg.get("rescue_min_confidence", cfg.get("min_confidence", 0.30)))
    full_box = _full_image_box(image_shape)
    tasks: list[Task] = []
    for detector_name in sorted(available):
        tasks.append(
            Task(
                detector_name=detector_name,
                filter_box=full_box,
                source_router_class=ROUTER_CLASS_BY_DETECTOR_NAME.get(detector_name, "未知"),
                router_region_indices=(),
                is_fallback=True,
                fallback_reason="low_confidence_router",
                min_confidence=min_conf,
            )
        )
    return tasks


def _empty_router_trigger_tasks(
    router_detections: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    fallback_cfg: dict[str, Any],
    available: set[str],
) -> list[Task]:
    cfg = fallback_cfg.get("trigger_empty_router", {}) or fallback_cfg.get("empty_router", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []
    if router_detections:
        return []
    min_conf = float(cfg.get("rescue_min_confidence", cfg.get("min_confidence", 0.30)))
    full_box = _full_image_box(image_shape)
    tasks: list[Task] = []
    for detector_name in sorted(available):
        tasks.append(
            Task(
                detector_name=detector_name,
                filter_box=full_box,
                source_router_class=ROUTER_CLASS_BY_DETECTOR_NAME.get(detector_name, "未知"),
                router_region_indices=(),
                is_fallback=True,
                fallback_reason="empty_router",
                min_confidence=min_conf,
            )
        )
    return tasks


def _aspect_ratio(bbox: Any) -> float:
    box = _xyxy(bbox)
    if box is None:
        return 0.0
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    return h / w


def _xyxy(bbox: Any) -> tuple[float, float, float, float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    return tuple(float(v) for v in bbox)  # type: ignore[return-value]


def _union_box(
    boxes: list[tuple[float, float, float, float] | None],
    image_shape: tuple[int, ...],
    padding_ratio: float,
) -> tuple[float, float, float, float] | None:
    valid = [b for b in boxes if b is not None and area_xyxy(b) > 0]
    if not valid:
        return None
    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)
    return tuple(
        float(v) for v in padded_xyxy((x1, y1, x2, y2), image_shape, padding_ratio=padding_ratio)
    )


def _full_image_box(image_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    h, w = image_shape[:2]
    return (0.0, 0.0, float(w), float(h))
