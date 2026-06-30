"""End-to-end router + crack detection pipeline runner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .ambiguity_display import build_ambiguity_candidate_groups
from .crack_detector_registry import build_detector_registry
from .display_merge import suppress_overlapping_display_detections
from .fallback_policy import (
    Task,
    plan_dynamic_fallback_tasks,
    plan_main_tasks,
    plan_static_fallback_tasks,
)
from .rfdetr_router_infer import RfdetrRouterConfig, RfdetrRouterInfer
from .region_view import make_region_view, map_region_xyxy_to_original, padded_xyxy
from .result_merge import Detection, center_in_xyxy, grade_level, ioa_min_xyxy, ioa_over_first_xyxy, iou_xyxy, nms_detections, prod_like_merge_detections
from .router_infer import RouterConfig, RouterInfer
from .wall_candidate_display import build_wall_candidate_display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml")
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-crack", action="store_true", help="use mock downstream crack detector")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def iter_images(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if source.is_file():
        return [source]
    return sorted([p for p in source.rglob("*") if p.suffix.lower() in exts])


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    config_path = resolve_path(args.config, repo_root)
    config = load_config(config_path)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else resolve_path(config["outputs"]["root"], config_path.parent) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = config["pipeline"]
    router = build_router(pipeline_cfg, config, config_path.parent, args.device)

    detector_cfg = {**config, "device": args.device}
    registry = build_detector_registry(detector_cfg, config_path.parent, mock=args.mock_crack)
    images = iter_images(resolve_path(args.source, repo_root))
    if args.limit:
        images = images[: args.limit]

    results_path = output_dir / "results.jsonl"
    summaries = []
    with results_path.open("w", encoding="utf-8") as f:
        for image_path in images:
            result = run_one_safe(image_path, router, registry, config)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            summaries.append(result)
            if not args.skip_visualization:
                save_visualization(image_path, result, vis_dir / f"{image_path.stem}_pipeline.jpg")

    summary = {
        "run_id": run_id,
        "images": len(images),
        "results_jsonl": str(results_path),
        "visualizations": str(vis_dir),
        "router_status_counts": count_values(router_status(r) for r in summaries),
        "crack_detections": sum(len(r["crack_detections"]) for r in summaries),
        "warning_counts": count_many(w for r in summaries for w in r.get("warnings", [])),
        "error_count": sum(1 for r in summaries if r.get("error")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_router(pipeline_cfg: dict[str, Any], config: dict[str, Any], root: Path, device: str) -> Any:
    backend = str(pipeline_cfg.get("router_backend", "yolo")).lower()
    if backend == "rfdetr":
        classes = {int(k): str(v) for k, v in config["classes"]["router"].items()}
        return RfdetrRouterInfer(
            RfdetrRouterConfig(
                checkpoint=resolve_path(pipeline_cfg["router_checkpoint"], root),
                class_names=classes,
                conf_threshold=float(pipeline_cfg.get("router_conf_threshold", 0.25)),
                low_conf_threshold=float(pipeline_cfg.get("router_low_conf_threshold", 0.10)),
                device=device,
                max_det=int(pipeline_cfg.get("router_max_det", 20)),
            )
        )
    return RouterInfer(
        RouterConfig(
            weights=resolve_path(pipeline_cfg["router_weights"], root),
            yolo_root=resolve_path(config.get("yolo_root", "../coarse_router_yolov9/yolov9"), root),
            data_yaml=resolve_path(pipeline_cfg.get("router_data_yaml", "../coarse_router_yolov9/datasets/coarse_router_3class_cleaned/data.yaml"), root),
            conf_threshold=float(pipeline_cfg.get("router_conf_threshold", 0.25)),
            low_conf_threshold=float(pipeline_cfg.get("router_low_conf_threshold", 0.10)),
            iou_threshold=float(pipeline_cfg.get("router_iou_threshold", 0.45)),
            device=device,
            imgsz=int(pipeline_cfg.get("imgsz", 640)),
            max_det=int(pipeline_cfg.get("router_max_det", 20)),
        )
    )


def run_one_safe(image_path: Path, router: RouterInfer, registry: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = run_one(image_path, router, registry, config)
    except Exception as exc:
        result = {
            "image": str(image_path),
            "error": f"pipeline_exception:{type(exc).__name__}",
            "error_detail": str(exc),
            "router": None,
            "raw_crack_detections": [],
            "crack_detections": [],
            "warnings": ["pipeline_exception"],
        }
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def apply_router_selection_policy(router_result: dict[str, Any], pipeline_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative router-region pruning before downstream detectors run."""
    original_detections = list(router_result.get("detections") or [])
    detections = list(original_detections)
    min_confidence = pipeline_cfg.get("router_min_region_confidence")
    if min_confidence is not None:
        threshold = float(min_confidence)
        detections = [
            det
            for det in detections
            if float(det.get("confidence") or 0.0) >= threshold
        ]
    max_regions = int(pipeline_cfg.get("router_max_regions") or 0)
    if max_regions > 0:
        detections = detections[:max_regions]

    rescue_top_k = int(pipeline_cfg.get("router_rescue_top_k_if_empty") or 0)
    rescued = False
    if not detections and rescue_top_k > 0 and original_detections:
        detections = original_detections[:rescue_top_k]
        rescued = True

    dominant_cfg = pipeline_cfg.get("dominant_router_class_policy", {}) or {}
    dominant_applied = False
    if detections and bool(dominant_cfg.get("enabled", False)):
        primary = detections[0]
        primary_class = str(primary.get("class_name", ""))
        allowed_classes = {str(v) for v in dominant_cfg.get("classes", [])}
        applies_to_class = not allowed_classes or primary_class in allowed_classes
        primary_conf = float(primary.get("confidence") or 0.0)
        primary_area = float(primary.get("area_ratio") or 0.0)
        min_conf = float(dominant_cfg.get("min_confidence", 0.90))
        min_area = float(dominant_cfg.get("min_area_ratio", 0.45))
        min_margin = float(dominant_cfg.get("min_confidence_margin", 0.0))
        next_other_conf = max(
            [float(det.get("confidence") or 0.0) for det in detections[1:] if str(det.get("class_name", "")) != primary_class],
            default=0.0,
        )
        if applies_to_class and primary_conf >= min_conf and primary_area >= min_area and primary_conf - next_other_conf >= min_margin:
            detections = [det for det in detections if str(det.get("class_name", "")) == primary_class]
            dominant_applied = True

    router_result = dict(router_result)
    router_result["detections"] = detections
    decision = dict(router_result.get("route_decision") or {})
    policy_parts = []
    if min_confidence is not None:
        policy_parts.append(f"min_confidence>={float(min_confidence):.2f}")
    if max_regions > 0:
        policy_parts.append(f"max_regions={max_regions}")
    if dominant_applied:
        policy_parts.append("dominant_class_only")
    if policy_parts:
        decision["strategy"] = "keep_all_router_boxes|" + "|".join(policy_parts)
        decision["primary_class"] = detections[0]["class_name"] if detections else None
        if not detections:
            decision["status"] = "unknown"
        elif rescued:
            decision["status"] = "low_confidence_rescue"
            decision["low_confidence_fallback_todo"] = True
            decision["router_selection_rescue"] = f"top{rescue_top_k}_if_empty"
    router_result["route_decision"] = decision
    return router_result


def run_one(image_path: Path, router: RouterInfer, registry: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"image": str(image_path), "error": "image_unreadable", "router": None, "raw_crack_detections": [], "crack_detections": [], "warnings": ["image_unreadable"]}
    router_result = router.predict(image)
    router_result = apply_router_selection_policy(router_result, config["pipeline"])
    all_cracks: list[Detection] = []
    raw_crack_records: list[dict[str, Any]] = []
    region_cfg = config["pipeline"]
    warnings: list[str] = []
    region_transport = str(region_cfg.get("region_transport", "ndarray_slice"))

    if region_transport == "full_image_filter":
        all_cracks, raw_crack_records, fallback_warnings = _run_full_image_filter_with_fallback(
            image=image,
            router_result=router_result,
            registry=registry,
            config=config,
        )
        warnings.extend(fallback_warnings)
    else:
        for router_region_index, router_det in enumerate(router_result["detections"]):
            router_class = router_det["class_name"]
            detectors = registry.get(router_class, [])
            region = make_region_view(
                image,
                router_det["bbox_xyxy"],
                padding_ratio=float(region_cfg.get("region_padding_ratio", 0.10)),
                make_contiguous=True,
            )
            for detector in detectors:
                try:
                    detector_outputs = detector.predict(region.image, router_class)
                except Exception as exc:
                    warnings.append(f"detector_exception:{getattr(detector, 'name', 'unknown')}:{type(exc).__name__}")
                    continue
                detector_outputs, detector_warnings = _apply_empty_detector_fallback(
                    detector=detector,
                    detector_outputs=detector_outputs,
                    image_bgr=region.image,
                    source_router_class=router_class,
                    config=config,
                )
                warnings.extend(detector_warnings)
                for det in detector_outputs:
                    mapped = map_region_xyxy_to_original(det.xyxy, region)
                    mapped_detection = Detection(
                        xyxy=mapped,
                        confidence=det.confidence,
                        grade=det.grade,
                        source_model=det.source_model,
                        source_router_class=det.source_router_class,
                    )
                    all_cracks.append(mapped_detection)
                    raw_record = detection_dict(mapped_detection)
                    raw_record.update(
                        {
                            "router_region_index": router_region_index,
                            "router_bbox_xyxy": [round(float(v), 3) for v in router_det["bbox_xyxy"]],
                            "router_confidence": router_det["confidence"],
                            "router_class_name": router_class,
                            "detector_input_shape": list(region.image.shape),
                            "region_transport": "ndarray_slice",
                        }
                    )
                    raw_crack_records.append(raw_record)
        rescue_cracks, rescue_records, rescue_warnings = _run_full_image_rescue(
            image=image,
            router_result=router_result,
            registry=registry,
            config=config,
            existing=all_cracks,
        )
        all_cracks.extend(rescue_cracks)
        raw_crack_records.extend(rescue_records)
        warnings.extend(rescue_warnings)

    merge_cfg = config.get("crack_merge", {})
    if str(merge_cfg.get("mode", "nms")) == "prod_like":
        merged = prod_like_merge_detections(
            all_cracks,
            same_model_ioa_threshold=float(merge_cfg.get("same_model_ioa_threshold", 0.70)),
            cross_model_iou_threshold=float(merge_cfg.get("cross_model_iou_threshold", 0.55)),
        )
    else:
        merged = nms_detections(
            all_cracks,
            same_grade_iou_threshold=float(merge_cfg.get("same_grade_iou_threshold", 0.50)),
            cross_grade_iou_threshold=float(merge_cfg.get("cross_grade_iou_threshold", 0.60)),
            prefer_higher_grade=bool(merge_cfg.get("prefer_higher_grade", True)),
        )
    ambiguity_cfg = config.get("ambiguity_display", {}) or {}
    ambiguity_groups, ambiguity_used_indices = build_ambiguity_candidate_groups(
        merged,
        iou_threshold=float(ambiguity_cfg.get("iou_threshold", 0.50)),
    )
    ambiguity_display = [det for group in ambiguity_groups for det in group["display_detections"]]
    wall_records_for_display = _wall_records_excluding_ambiguity(
        raw_crack_records, merged, ambiguity_used_indices
    )
    wall_display_cfg = config.get("wall_display", {})
    wall_candidate_display = build_wall_candidate_display(
        wall_records_for_display,
        iou_threshold=float(wall_display_cfg.get("pair_iou_threshold", merge_cfg.get("cross_model_iou_threshold", 0.55))),
        ioa_threshold=float(wall_display_cfg.get("pair_ioa_threshold", 0.70)),
        min_single_confidence=float(wall_display_cfg.get("min_single_confidence", 0.05)),
        min_single_confidence_by_model={
            str(key): float(value)
            for key, value in (wall_display_cfg.get("min_single_confidence_by_model", {}) or {}).items()
        },
        max_single_groups_per_model=int(wall_display_cfg.get("max_single_groups_per_model", 4)),
        use_union_bbox_for_pairs=bool(wall_display_cfg.get("use_union_bbox_for_pairs", True)),
    )
    wall_display_items = select_wall_display_items(
        wall_records=wall_records_for_display,
        wall_candidate_display=wall_candidate_display,
        wall_display_cfg=wall_display_cfg,
    )
    if router_result["route_decision"]["status"] == "low_confidence":
        warnings.append("router_low_confidence_multi_model_fallback_todo")
    if not router_result["detections"]:
        warnings.append("router_unknown")
    if ambiguity_groups:
        warnings.append(f"ambiguous_class_candidates:{len(ambiguity_groups)}")

    display_items = _compose_display_items(
        merged=merged,
        ambiguity_used_indices=ambiguity_used_indices,
        wall_display=wall_display_items,
        ambiguity_display=ambiguity_display,
    )
    display_items = _postprocess_final_display_items(
        display_items,
        config.get("final_display_postprocess", {}) or {},
        router_result=router_result,
    )
    display_merge_cfg = config.get("display_merge", {}) or {}
    if bool(display_merge_cfg.get("enabled", True)):
        display_items, suppressed_display_items = suppress_overlapping_display_detections(
            display_items,
            iou_threshold=float(display_merge_cfg.get("iou_threshold", 0.45)),
            ioa_threshold=float(display_merge_cfg.get("ioa_threshold", 0.70)),
            cross_class_ioa_threshold=float(display_merge_cfg.get("cross_class_ioa_threshold", 0.75)),
        )
    else:
        suppressed_display_items = []
    display_items, fallback_warnings = ensure_minimum_display_outputs(
        display_items=display_items,
        suppressed_display_items=suppressed_display_items,
        wall_records=wall_records_for_display,
        merged=merged,
        raw_records=raw_crack_records,
        wall_display_cfg=wall_display_cfg,
        fallback_cfg=config.get("final_output_fallback", {}) or {},
    )
    warnings.extend(fallback_warnings)

    return {
        "image": str(image_path),
        "image_shape": list(image.shape),
        "pipeline_version": "router3_crack_v2_class_safe",
        "router": router_result,
        "raw_crack_detections": raw_crack_records,
        "crack_detections": [detection_dict(d) for d in merged],
        "display_crack_detections": display_items,
        "suppressed_display_crack_detections": suppressed_display_items,
        "wall_candidate_display": wall_candidate_display,
        "ambiguity_candidate_groups": ambiguity_groups,
        "warnings": warnings,
    }


def detection_dict(det: Detection) -> dict[str, Any]:
    return {
        "bbox_xyxy": [round(v, 3) for v in det.xyxy],
        "confidence": det.confidence,
        "damage_grade": det.grade,
        "source_model": det.source_model,
        "source_router_class": det.source_router_class,
        "coordinate_space": "original_image",
    }


def display_crack_detections(merged: list[Detection], wall_display: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return product-display detections.

    Non-wall detections keep the existing merged output. Wall detections are
    replaced by the display groups: same level -> one item, different levels ->
    two subtype candidates under the same image record.
    """
    non_wall = [detection_dict(det) for det in merged if det.source_router_class != "壁类"]
    return non_wall + wall_display


def select_wall_display_items(
    wall_records: list[dict[str, Any]],
    wall_candidate_display: dict[str, Any],
    wall_display_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(wall_display_cfg.get("mode", "rule_merged"))
    if mode == "raw_all":
        return build_raw_wall_display_items(wall_records, wall_display_cfg)
    if mode == "merged_plus_raw":
        items = list(wall_candidate_display.get("display_detections", []))
        append_only_uncovered = bool(wall_display_cfg.get("raw_append_if_uncovered", True))
        raw_covered_ioa_threshold = float(wall_display_cfg.get("raw_covered_ioa_threshold", 0.80))
        raw_covered_iou_threshold = float(wall_display_cfg.get("raw_covered_iou_threshold", 0.50))
        for det in build_raw_wall_display_items(wall_records, wall_display_cfg):
            if append_only_uncovered and _wall_raw_item_is_represented(
                det,
                items,
                ioa_threshold=raw_covered_ioa_threshold,
                iou_threshold=raw_covered_iou_threshold,
            ):
                continue
            _append_unique_display_candidate(items, det)
        if bool(wall_display_cfg.get("merge_overlapping_display_items", False)):
            items = _merge_overlapping_wall_display_items(
                items,
                iou_threshold=float(wall_display_cfg.get("display_cluster_iou_threshold", 0.35)),
                ioa_threshold=float(wall_display_cfg.get("display_cluster_ioa_threshold", 0.70)),
            )
        return items
    return list(wall_candidate_display.get("display_detections", []))


def build_raw_wall_display_items(
    wall_records: list[dict[str, Any]],
    wall_display_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_models = {str(v) for v in (wall_display_cfg.get("raw_source_models", []) or [])}
    per_model_thresholds = {
        str(key): float(value)
        for key, value in (wall_display_cfg.get("raw_min_confidence_by_model", {}) or {}).items()
    }
    default_threshold = float(wall_display_cfg.get("raw_min_confidence", 0.0))
    items: list[dict[str, Any]] = []
    for record in wall_records:
        model = str(record.get("source_model") or "")
        if allowed_models and model not in allowed_models:
            continue
        threshold = per_model_thresholds.get(model, default_threshold)
        if float(record.get("confidence") or 0.0) < threshold:
            continue
        items.append(_display_item_from_raw_record(record))
    max_outputs = int(wall_display_cfg.get("raw_max_outputs", 0) or 0)
    items = sorted(items, key=_fallback_display_priority, reverse=True)
    return items[:max_outputs] if max_outputs > 0 else items


def ensure_minimum_display_outputs(
    display_items: list[dict[str, Any]],
    suppressed_display_items: list[dict[str, Any]],
    wall_records: list[dict[str, Any]],
    merged: list[Detection],
    raw_records: list[dict[str, Any]],
    wall_display_cfg: dict[str, Any],
    fallback_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    if display_items:
        return display_items, []
    if not bool(fallback_cfg.get("enabled", False)):
        return display_items, []

    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    if bool(fallback_cfg.get("restore_suppressed", True)):
        for det in suppressed_display_items:
            _append_unique_display_candidate(candidates, det)

    if bool(fallback_cfg.get("rebuild_wall_display_if_empty", True)) and wall_records:
        relaxed_wall_display = build_wall_candidate_display(
            wall_records,
            iou_threshold=float(
                fallback_cfg.get(
                    "relaxed_wall_pair_iou_threshold",
                    wall_display_cfg.get("pair_iou_threshold", 0.55),
                )
            ),
            ioa_threshold=float(
                fallback_cfg.get(
                    "relaxed_wall_pair_ioa_threshold",
                    wall_display_cfg.get("pair_ioa_threshold", 0.70),
                )
            ),
            min_single_confidence=float(
                fallback_cfg.get(
                    "relaxed_min_single_confidence",
                    wall_display_cfg.get("min_single_confidence", 0.05),
                )
            ),
            min_single_confidence_by_model={
                str(key): float(value)
                for key, value in (
                    fallback_cfg.get("relaxed_min_single_confidence_by_model")
                    or wall_display_cfg.get("min_single_confidence_by_model", {})
                    or {}
                ).items()
            },
            max_single_groups_per_model=int(
                fallback_cfg.get(
                    "relaxed_max_single_groups_per_model",
                    wall_display_cfg.get("max_single_groups_per_model", 4),
                )
            ),
            use_union_bbox_for_pairs=bool(
                fallback_cfg.get(
                    "relaxed_use_union_bbox_for_pairs",
                    wall_display_cfg.get("use_union_bbox_for_pairs", True),
                )
            ),
        )
        for det in relaxed_wall_display["display_detections"]:
            _append_unique_display_candidate(candidates, det)

    if bool(fallback_cfg.get("include_merged_candidates", True)):
        for det in merged:
            _append_unique_display_candidate(candidates, _display_item_from_detection(det))

    if bool(fallback_cfg.get("include_raw_candidates", True)):
        for record in raw_records:
            _append_unique_display_candidate(candidates, _display_item_from_raw_record(record))

    candidates = [
        det for det in sorted(candidates, key=_fallback_display_priority, reverse=True) if _valid_box(det)
    ]
    max_outputs = int(fallback_cfg.get("max_outputs", 8))
    if max_outputs > 0:
        candidates = candidates[:max_outputs]

    if candidates:
        warnings.append(f"final_output_fallback_used:{len(candidates)}")
    return candidates, warnings


def _compose_display_items(
    merged: list[Detection],
    ambiguity_used_indices: set[int],
    wall_display: list[dict[str, Any]],
    ambiguity_display: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the final display list with strict class-isolation.

    Order:
    1. Non-wall detections that are NOT part of an ambiguity group (天井 /
       RC柱 main outputs and independent RC柱 fallback rescues).
    2. Wall candidate display groups (inner_wall vs rc_wall pairing).
    3. Ambiguous class candidate groups (cross-class pairs preserved as
       parallel candidates so the auditor can pick the correct class).
    """
    non_wall = [
        detection_dict(det)
        for index, det in enumerate(merged)
        if det.source_router_class != "壁类" and index not in ambiguity_used_indices
    ]
    return non_wall + wall_display + ambiguity_display


def _apply_empty_detector_fallback(
    detector: Any,
    detector_outputs: list[Detection],
    image_bgr: Any,
    source_router_class: str,
    config: dict[str, Any],
) -> tuple[list[Detection], list[str]]:
    if detector_outputs:
        return detector_outputs, []
    cfg = config.get("downstream_empty_fallback", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return detector_outputs, []
    detector_name = str(getattr(detector, "name", "unknown"))
    fallback_outputs = _dynamic_empty_detector_outputs(
        detector=detector,
        image_bgr=image_bgr,
        source_router_class=source_router_class,
        cfg=cfg,
    )
    if not fallback_outputs:
        return detector_outputs, []
    max_outputs = int(cfg.get("max_outputs_per_region", 1) or 0)
    if max_outputs > 0:
        fallback_outputs = fallback_outputs[:max_outputs]
    return fallback_outputs, [f"downstream_empty_fallback:{detector_name}:{len(fallback_outputs)}"]


def _dynamic_empty_detector_outputs(
    detector: Any,
    image_bgr: Any,
    source_router_class: str,
    cfg: dict[str, Any],
) -> list[Detection]:
    detector_name = str(getattr(detector, "name", "unknown"))
    backend = getattr(detector, "backend", None)
    if backend is None or not hasattr(backend, "predict"):
        return []
    grade_names = getattr(detector, "grade_names", {0: "B", 1: "C", 2: "D"})
    for thresholds in _empty_fallback_threshold_schedule(detector, cfg):
        outputs: list[Detection] = []
        for raw_det in backend.predict(image_bgr, thresholds):
            grade = grade_names.get(raw_det.class_id, raw_det.class_name)
            outputs.append(
                Detection(
                    xyxy=raw_det.xyxy,
                    confidence=raw_det.confidence,
                    grade=str(grade),
                    source_model=detector_name,
                    source_router_class=source_router_class,
                )
            )
        if outputs:
            return outputs
    return []


def _empty_fallback_threshold_schedule(detector: Any, cfg: dict[str, Any]) -> list[list[float]]:
    detector_name = str(getattr(detector, "name", "unknown"))
    thresholds_by_model = cfg.get("thresholds_by_model", {}) or {}
    if not bool(cfg.get("dynamic", False)):
        thresholds = thresholds_by_model.get(detector_name) or cfg.get("thresholds")
        return [[float(value) for value in thresholds]] if thresholds else []

    base_thresholds = getattr(detector, "thresholds", None)
    if not base_thresholds:
        thresholds = thresholds_by_model.get(detector_name) or cfg.get("thresholds")
        base_thresholds = thresholds
    if not base_thresholds:
        return []

    current = [float(value) for value in base_thresholds]
    min_threshold = float(cfg.get("min_threshold", 0.05))
    step = max(1e-6, float(cfg.get("step", 0.05)))
    schedule: list[list[float]] = []
    while any(value > min_threshold for value in current):
        current = [max(min_threshold, value - step) for value in current]
        schedule.append(list(current))
    return schedule


def _dynamic_empty_detector_outputs_for_region(
    detector: Any,
    image_bgr: Any,
    source_router_class: str,
    cfg: dict[str, Any],
    filter_box: tuple[float, float, float, float],
    region_cfg: dict[str, Any],
) -> list[Detection]:
    detector_name = str(getattr(detector, "name", "unknown"))
    backend = getattr(detector, "backend", None)
    if backend is None or not hasattr(backend, "predict"):
        return []
    grade_names = getattr(detector, "grade_names", {0: "B", 1: "C", 2: "D"})
    for thresholds in _empty_fallback_threshold_schedule(detector, cfg):
        outputs: list[Detection] = []
        for raw_det in backend.predict(image_bgr, thresholds):
            det = Detection(
                xyxy=raw_det.xyxy,
                confidence=raw_det.confidence,
                grade=str(grade_names.get(raw_det.class_id, raw_det.class_name)),
                source_model=detector_name,
                source_router_class=source_router_class,
            )
            if detection_in_router_region(det.xyxy, filter_box, region_cfg):
                outputs.append(det)
        if outputs:
            return outputs
    return []


def _postprocess_final_display_items(
    display_items: list[dict[str, Any]],
    cfg: dict[str, Any],
    router_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not bool(cfg.get("enabled", False)):
        return display_items
    items = list(display_items)
    ambiguity_cfg = cfg.get("collapse_ambiguity", {}) or {}
    if bool(ambiguity_cfg.get("enabled", True)):
        items = _collapse_ambiguity_display_items(items, ambiguity_cfg)
    cluster_cfg = cfg.get("cluster_same_family", {}) or {}
    if bool(cluster_cfg.get("enabled", True)):
        items = _merge_overlapping_display_items_by_family(
            items,
            iou_threshold=float(cluster_cfg.get("iou_threshold", 0.35)),
            ioa_threshold=float(cluster_cfg.get("ioa_threshold", 0.70)),
        )
    dominant_cfg = cfg.get("dominant_router_filter", {}) or {}
    if bool(dominant_cfg.get("enabled", False)) and router_result is not None:
        items = _filter_display_by_dominant_router(items, router_result, dominant_cfg)
    return items


def _filter_display_by_dominant_router(
    items: list[dict[str, Any]],
    router_result: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    detections = list(router_result.get("detections") or [])
    if not detections:
        return items
    primary = detections[0]
    primary_class = str(primary.get("class_name") or "")
    allowed_classes = {str(value) for value in (cfg.get("classes", []) or [])}
    if allowed_classes and primary_class not in allowed_classes:
        return items
    primary_conf = float(primary.get("confidence") or 0.0)
    primary_area = float(primary.get("area_ratio") or 0.0)
    min_conf = float(cfg.get("min_confidence", 0.90))
    min_area = float(cfg.get("min_area_ratio", 0.0))
    min_margin = float(cfg.get("min_confidence_margin", 0.25))
    next_other_conf = max(
        [
            float(det.get("confidence") or 0.0)
            for det in detections[1:]
            if str(det.get("class_name") or "") != primary_class
        ],
        default=0.0,
    )
    if primary_conf < min_conf or primary_area < min_area or primary_conf - next_other_conf < min_margin:
        return items
    filtered = [item for item in items if _display_item_matches_router_class(item, primary_class)]
    return filtered or items


def _display_item_matches_router_class(item: dict[str, Any], router_class: str) -> bool:
    candidates = item.get("candidates") or []
    if candidates and any(_display_item_matches_router_class(candidate, router_class) for candidate in candidates):
        return True
    if router_class in {"壁类", "壁類"}:
        return _is_wall_display_item(item)
    if str(item.get("source_router_class") or "") == router_class:
        return True
    if router_class == "天井" and str(item.get("source_model") or "") == "ceiling":
        return True
    if router_class == "RC柱" and str(item.get("source_model") or "") == "rc_column":
        return True
    return False


def _collapse_ambiguity_display_items(
    items: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    prefer_wall = bool(cfg.get("prefer_wall_when_present", True))
    output: list[dict[str, Any]] = []
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if str(item.get("status") or "") != "ambiguous_class_candidate":
            output.append(item)
            continue
        groups.setdefault(int(item.get("group_index", -1)), []).append(item)

    for group_items in groups.values():
        candidates = _unique_candidate_records(
            candidate
            for item in group_items
            for candidate in (item.get("candidates") or [_display_merge_member(item)])
        )
        wall_candidates = [candidate for candidate in candidates if _is_wall_display_item(candidate)]
        if prefer_wall and wall_candidates:
            output.append(_display_item_from_candidate_group(
                wall_candidates,
                status="wall_ambiguity_resolved",
                structure_type="壁類",
                damage_prefix="壁-",
                reason="跨类别重叠候选中存在壁类模型输出，最终显示按壁大类归并。",
            ))
            continue
        output.append(_display_item_from_candidate_group(
            candidates,
            status="ambiguity_resolved_best",
            structure_type=None,
            damage_prefix="",
            reason="跨类别重叠候选已按最终显示优先级收敛为一个候选。",
        ))
    return output


def _display_item_from_candidate_group(
    candidates: list[dict[str, Any]],
    status: str,
    structure_type: str | None,
    damage_prefix: str,
    reason: str,
) -> dict[str, Any]:
    representative = max(candidates, key=_fallback_display_priority)
    boxes = [_box(candidate) for candidate in candidates if _valid_box(candidate)]
    grade = grade_level(str(representative.get("damage_grade") or representative.get("raw_damage_grade") or ""))
    return {
        "group_index": representative.get("group_index", -1),
        "status": status,
        "structure_type": structure_type or representative.get("structure_type"),
        "damage_grade": f"{damage_prefix}{grade}" if damage_prefix else grade,
        "raw_damage_grade": representative.get("raw_damage_grade") or representative.get("damage_grade"),
        "confidence": max(float(candidate.get("confidence") or 0.0) for candidate in candidates),
        "bbox_xyxy": [
            round(min(box[0] for box in boxes), 3),
            round(min(box[1] for box in boxes), 3),
            round(max(box[2] for box in boxes), 3),
            round(max(box[3] for box in boxes), 3),
        ],
        "source_model": representative.get("source_model"),
        "source_router_class": "壁类" if structure_type in {"壁類", "壁类"} else representative.get("source_router_class"),
        "reason": reason,
        "candidates": candidates,
        "display_bbox_source": "candidate_group_union",
        "display_suppressed_count": max(0, len(candidates) - 1),
    }


def _unique_candidate_records(candidates: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        _append_unique_display_candidate(output, dict(candidate))
    return output


def _merge_overlapping_display_items_by_family(
    items: list[dict[str, Any]],
    iou_threshold: float,
    ioa_threshold: float,
) -> list[dict[str, Any]]:
    if len(items) <= 1:
        return items
    groups = _overlapping_display_family_groups(items, iou_threshold, ioa_threshold)
    merged: list[dict[str, Any]] = []
    for group in groups:
        group_items = [items[index] for index in group]
        if len(group_items) == 1:
            merged.append(group_items[0])
        elif all(_is_wall_display_item(item) for item in group_items):
            merged.append(_merged_wall_display_group(group_items))
        else:
            merged.append(_merged_display_group(group_items))
    return sorted(merged, key=_fallback_display_priority, reverse=True)


def _overlapping_display_family_groups(
    items: list[dict[str, Any]],
    iou_threshold: float,
    ioa_threshold: float,
) -> list[list[int]]:
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            right = items[right_index]
            if _display_family(left) != _display_family(right):
                continue
            if not _valid_box(left) or not _valid_box(right):
                continue
            left_box = _box(left)
            right_box = _box(right)
            if iou_xyxy(left_box, right_box) >= iou_threshold or ioa_min_xyxy(left_box, right_box) >= ioa_threshold:
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(items)):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _merged_display_group(group_items: list[dict[str, Any]]) -> dict[str, Any]:
    representative = max(group_items, key=_fallback_display_priority)
    boxes = [_box(item) for item in group_items if _valid_box(item)]
    merged = dict(representative)
    merged["bbox_xyxy"] = [
        round(min(box[0] for box in boxes), 3),
        round(min(box[1] for box in boxes), 3),
        round(max(box[2] for box in boxes), 3),
        round(max(box[3] for box in boxes), 3),
    ]
    merged["confidence"] = max(float(item.get("confidence") or 0.0) for item in group_items)
    merged["status"] = "display_family_cluster_merged"
    merged["display_bbox_source"] = "display_family_cluster_union"
    merged["reason"] = "最终显示中同构件类别的重叠候选已聚类显示，避免重复碎框。"
    merged["candidates"] = _combined_wall_candidates(group_items)
    merged["display_merge_members"] = [_display_merge_member(item) for item in group_items]
    merged["display_suppressed_count"] = len(group_items) - 1
    return merged


def _display_family(item: dict[str, Any]) -> str:
    if _is_wall_display_item(item):
        return "wall"
    router_class = str(item.get("source_router_class") or "")
    if router_class:
        return router_class
    source_model = str(item.get("source_model") or "")
    if source_model == "ceiling":
        return "天井"
    if source_model == "rc_column":
        return "RC柱"
    return source_model or str(item.get("structure_type") or "")


def _display_item_from_detection(det: Detection) -> dict[str, Any]:
    base = detection_dict(det)
    if det.source_router_class != "壁类":
        base["status"] = "final_output_fallback"
        base["reason"] = "最终显示为空，回退到 merge 后最高优先级候选。"
        return base
    grade = grade_level(str(det.grade))
    return {
        "group_index": -1,
        "status": "final_output_fallback",
        "structure_type": "壁類",
        "damage_grade": f"壁-{grade}",
        "raw_damage_grade": det.grade,
        "confidence": det.confidence,
        "bbox_xyxy": [round(v, 3) for v in det.xyxy],
        "source_model": det.source_model,
        "source_router_class": det.source_router_class,
        "reason": "最终显示为空，回退到 merge 后壁类候选。",
        "candidates": [
            {
                "structure_type": "壁類",
                "source_model": det.source_model,
                "source_router_class": det.source_router_class,
                "damage_grade": grade,
                "raw_damage_grade": det.grade,
                "confidence": det.confidence,
                "bbox_xyxy": [round(v, 3) for v in det.xyxy],
            }
        ],
    }


def _display_item_from_raw_record(record: dict[str, Any]) -> dict[str, Any]:
    if str(record.get("source_router_class")) != "壁类":
        output = dict(record)
        output["status"] = "final_output_fallback"
        output["reason"] = "最终显示为空，回退到 raw 候选。"
        return output
    grade = grade_level(str(record.get("damage_grade", "")))
    return {
        "group_index": -1,
        "status": "final_output_fallback",
        "structure_type": "壁類",
        "damage_grade": f"壁-{grade}",
        "raw_damage_grade": record.get("damage_grade"),
        "confidence": record.get("confidence"),
        "bbox_xyxy": record.get("bbox_xyxy"),
        "source_model": record.get("source_model"),
        "source_router_class": record.get("source_router_class"),
        "reason": "最终显示为空，回退到 raw 壁类候选。",
        "candidates": [
            {
                "structure_type": "壁類",
                "source_model": record.get("source_model"),
                "source_router_class": record.get("source_router_class"),
                "damage_grade": grade,
                "raw_damage_grade": record.get("damage_grade"),
                "confidence": record.get("confidence"),
                "bbox_xyxy": record.get("bbox_xyxy"),
            }
        ],
    }


def _append_unique_display_candidate(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    same_iou_threshold: float = 0.90,
) -> None:
    if not _valid_box(candidate):
        return
    cand_box = _box(candidate)
    cand_label = str(candidate.get("damage_grade") or "")
    cand_source = str(candidate.get("source_model") or "")
    for existing in candidates:
        if not _valid_box(existing):
            continue
        if cand_label != str(existing.get("damage_grade") or ""):
            continue
        if cand_source != str(existing.get("source_model") or ""):
            continue
        if iou_xyxy(cand_box, _box(existing)) >= same_iou_threshold:
            return
    candidates.append(candidate)


def _wall_raw_item_is_represented(
    raw_item: dict[str, Any],
    display_items: list[dict[str, Any]],
    ioa_threshold: float,
    iou_threshold: float,
) -> bool:
    """Return whether a raw wall candidate is already represented in a group.

    `merged_plus_raw` uses raw candidates as a recall-first safety net. Raw
    boxes that already participate in a wall display group should not be drawn
    again, otherwise a correctly merged result looks visually unmerged.
    """
    if not _valid_box(raw_item):
        return False
    raw_box = _box(raw_item)
    raw_source = str(raw_item.get("source_model") or "")
    raw_grade = grade_level(str(raw_item.get("damage_grade") or ""))

    for display_item in display_items:
        for candidate in display_item.get("candidates") or []:
            if str(candidate.get("source_model") or "") != raw_source:
                continue
            candidate_grade = grade_level(str(candidate.get("damage_grade") or ""))
            if raw_grade and candidate_grade and raw_grade != candidate_grade:
                continue
            if _boxes_match_represented_raw(raw_box, candidate, ioa_threshold, iou_threshold):
                return True

        if str(display_item.get("source_model") or "") != raw_source:
            continue
        display_grade = grade_level(str(display_item.get("damage_grade") or ""))
        if raw_grade and display_grade and raw_grade != display_grade:
            continue
        if _boxes_match_represented_raw(raw_box, display_item, ioa_threshold, iou_threshold):
            return True

    return False


def _boxes_match_represented_raw(
    raw_box: tuple[float, float, float, float],
    candidate: dict[str, Any],
    ioa_threshold: float,
    iou_threshold: float,
) -> bool:
    if not _valid_box(candidate):
        return False
    candidate_box = _box(candidate)
    return (
        ioa_over_first_xyxy(raw_box, candidate_box) >= ioa_threshold
        or iou_xyxy(raw_box, candidate_box) >= iou_threshold
    )


def _merge_overlapping_wall_display_items(
    items: list[dict[str, Any]],
    iou_threshold: float,
    ioa_threshold: float,
) -> list[dict[str, Any]]:
    if len(items) <= 1:
        return items
    groups = _overlapping_wall_display_groups(items, iou_threshold, ioa_threshold)
    merged: list[dict[str, Any]] = []
    for group in groups:
        group_items = [items[index] for index in group]
        if len(group_items) == 1:
            merged.append(group_items[0])
            continue
        merged.append(_merged_wall_display_group(group_items))
    return sorted(merged, key=_fallback_display_priority, reverse=True)


def _overlapping_wall_display_groups(
    items: list[dict[str, Any]],
    iou_threshold: float,
    ioa_threshold: float,
) -> list[list[int]]:
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(items):
        if not _is_wall_display_item(left):
            continue
        for right_index in range(left_index + 1, len(items)):
            right = items[right_index]
            if not _is_wall_display_item(right):
                continue
            if not _valid_box(left) or not _valid_box(right):
                continue
            left_box = _box(left)
            right_box = _box(right)
            if iou_xyxy(left_box, right_box) >= iou_threshold or ioa_min_xyxy(left_box, right_box) >= ioa_threshold:
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(items)):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _merged_wall_display_group(group_items: list[dict[str, Any]]) -> dict[str, Any]:
    representative = max(group_items, key=_fallback_display_priority)
    boxes = [_box(item) for item in group_items if _valid_box(item)]
    merged = dict(representative)
    merged["bbox_xyxy"] = [
        round(min(box[0] for box in boxes), 3),
        round(min(box[1] for box in boxes), 3),
        round(max(box[2] for box in boxes), 3),
        round(max(box[3] for box in boxes), 3),
    ]
    merged["confidence"] = max(float(item.get("confidence") or 0.0) for item in group_items)
    merged["status"] = "wall_display_cluster_merged"
    merged["display_bbox_source"] = "wall_display_cluster_union"
    merged["reason"] = "重叠的壁类候选已按召回优先策略聚类显示；成员候选保留在 candidates/display_merge_members 中。"
    merged["candidates"] = _combined_wall_candidates(group_items)
    merged["display_merge_members"] = [_display_merge_member(item) for item in group_items]
    merged["display_suppressed_count"] = len(group_items) - 1
    return merged


def _combined_wall_candidates(group_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in group_items:
        item_candidates = item.get("candidates") or []
        if item_candidates:
            for candidate in item_candidates:
                _append_unique_display_candidate(candidates, dict(candidate))
        else:
            _append_unique_display_candidate(candidates, _display_merge_member(item))
    return candidates


def _display_merge_member(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "structure_type": item.get("structure_type"),
        "damage_grade": item.get("damage_grade"),
        "raw_damage_grade": item.get("raw_damage_grade"),
        "confidence": item.get("confidence"),
        "bbox_xyxy": item.get("bbox_xyxy"),
        "source_model": item.get("source_model"),
        "source_router_class": item.get("source_router_class"),
        "status": item.get("status"),
    }


def _is_wall_display_item(item: dict[str, Any]) -> bool:
    return (
        str(item.get("structure_type") or "") in {"壁類", "壁类"}
        or str(item.get("source_router_class") or "") in {"壁類", "壁类"}
        or str(item.get("source_model") or "") in {"inner_wall", "rc_wall", "wall_merged", "wall"}
    )


def _fallback_display_priority(det: dict[str, Any]) -> tuple[int, float, float]:
    status = str(det.get("status") or "")
    status_bonus = 1 if status in {"wall_rule_merged", "single_model", "ambiguous_class_candidate"} else 0
    grade = _grade_rank_from_display(det)
    area = (_box(det)[2] - _box(det)[0]) * (_box(det)[3] - _box(det)[1])
    return (grade + status_bonus, float(det.get("confidence") or 0.0), area)


def _grade_rank_from_display(det: dict[str, Any]) -> int:
    return {"B": 1, "C": 2, "D": 3}.get(grade_level(str(det.get("damage_grade", ""))), 0)


def _valid_box(det: dict[str, Any]) -> bool:
    values = det.get("bbox_xyxy") or []
    if len(values) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in values]
    return x2 > x1 and y2 > y1


def _box(det: dict[str, Any]) -> tuple[float, float, float, float]:
    values = det.get("bbox_xyxy") or [0, 0, 0, 0]
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def _wall_records_excluding_ambiguity(
    raw_records: list[dict[str, Any]],
    merged: list[Detection],
    ambiguity_used_indices: set[int],
) -> list[dict[str, Any]]:
    """Remove wall raw records whose merged representative was claimed by an
    ambiguity group, so they are not rendered twice (once via wall_display,
    once via ambiguity_display).
    """
    if not ambiguity_used_indices:
        return raw_records
    excluded_boxes: list[tuple[float, float, float, float]] = []
    for index in ambiguity_used_indices:
        det = merged[index]
        if det.source_router_class != "壁类":
            continue
        excluded_boxes.append(tuple(float(v) for v in det.xyxy))
    if not excluded_boxes:
        return raw_records
    filtered: list[dict[str, Any]] = []
    for record in raw_records:
        if str(record.get("source_router_class")) != "壁类":
            filtered.append(record)
            continue
        bbox = record.get("bbox_xyxy") or [0, 0, 0, 0]
        record_box = tuple(float(v) for v in bbox)
        if any(
            ioa_over_first_xyxy(record_box, excluded) >= 0.80
            for excluded in excluded_boxes
        ):
            continue
        filtered.append(record)
    return filtered


def detection_in_router_region(
    detection_xyxy: tuple[float, float, float, float],
    router_xyxy: tuple[float, float, float, float],
    config: dict[str, Any],
) -> bool:
    mode = str(config.get("region_filter_mode", "center_or_ioa"))
    ioa_threshold = float(config.get("region_filter_ioa_threshold", 0.50))
    if mode == "center":
        return center_in_xyxy(detection_xyxy, router_xyxy)
    if mode == "ioa":
        return ioa_over_first_xyxy(detection_xyxy, router_xyxy) >= ioa_threshold
    return center_in_xyxy(detection_xyxy, router_xyxy) or ioa_over_first_xyxy(detection_xyxy, router_xyxy) >= ioa_threshold


def _flatten_registry(registry: dict[str, list[Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for detectors in registry.values():
        for detector in detectors:
            name = getattr(detector, "name", None)
            if name and name not in flat:
                flat[name] = detector
    return flat


def _run_full_image_rescue(
    image: Any,
    router_result: dict[str, Any],
    registry: dict[str, list[Any]],
    config: dict[str, Any],
    existing: list[Detection],
) -> tuple[list[Detection], list[dict[str, Any]], list[str]]:
    cfg = config.get("full_image_rescue", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return [], [], []
    classes = {str(value) for value in cfg.get("router_classes", [])}
    padding = float(cfg.get("region_padding_ratio", config["pipeline"].get("region_padding_ratio", 0.10)))
    duplicate_iou = float(cfg.get("duplicate_iou_threshold", 0.70))
    min_confidence = float(cfg.get("min_confidence", 0.0))
    max_existing_confidence = cfg.get("max_existing_confidence")
    warnings: list[str] = []
    added: list[Detection] = []
    records: list[dict[str, Any]] = []
    detector_cache: dict[str, list[Detection]] = {}

    for router_region_index, router_det in enumerate(router_result.get("detections", [])):
        router_class = str(router_det.get("class_name", ""))
        if classes and router_class not in classes:
            continue
        filter_box = tuple(
            float(v) for v in padded_xyxy(router_det["bbox_xyxy"], image.shape, padding_ratio=padding)
        )
        if max_existing_confidence is not None:
            existing_conf = max(
                [
                    float(prev.confidence)
                    for prev in existing
                    if prev.source_router_class == router_class and detection_in_router_region(prev.xyxy, filter_box, config["pipeline"])
                ],
                default=0.0,
            )
            if existing_conf >= float(max_existing_confidence):
                continue
        for detector in registry.get(router_class, []):
            detector_name = str(getattr(detector, "name", "unknown"))
            if detector_name not in detector_cache:
                try:
                    detector_cache[detector_name] = detector.predict(image, router_class)
                except Exception as exc:
                    warnings.append(f"full_image_rescue_exception:{detector_name}:{type(exc).__name__}")
                    detector_cache[detector_name] = []
            for det in detector_cache[detector_name]:
                if float(det.confidence) < min_confidence:
                    continue
                if not detection_in_router_region(det.xyxy, filter_box, config["pipeline"]):
                    continue
                if any(iou_xyxy(det.xyxy, prev.xyxy) >= duplicate_iou for prev in existing + added):
                    continue
                rescued = Detection(
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    grade=det.grade,
                    source_model=det.source_model,
                    source_router_class=router_class,
                )
                added.append(rescued)
                raw_record = detection_dict(rescued)
                raw_record.update(
                    {
                        "router_region_index": router_region_index,
                        "router_bbox_xyxy": [round(float(v), 3) for v in router_det["bbox_xyxy"]],
                        "router_filter_bbox_xyxy": [round(float(v), 3) for v in filter_box],
                        "router_confidence": router_det["confidence"],
                        "router_class_name": router_class,
                        "detector_input_shape": list(image.shape),
                        "region_transport": "full_image_rescue",
                        "is_fallback": True,
                        "fallback_reasons": ["wall_full_image_rescue"],
                    }
                )
                records.append(raw_record)
    if added:
        warnings.append(f"full_image_rescue_detections:{len(added)}")
    return added, records, warnings


def _run_full_image_filter_with_fallback(
    image: Any,
    router_result: dict[str, Any],
    registry: dict[str, list[Any]],
    config: dict[str, Any],
) -> tuple[list[Detection], list[dict[str, Any]], list[str]]:
    """Two-phase task scheduler.

    Phase 1 runs main tasks plus the static-trigger fallback tasks
    (morphology / parallel-walls / low-confidence-router / empty-router).
    Phase 2 inspects the maximum main-detector confidence per router region
    and, if any region's main detectors fell silent, enqueues the
    cross-class sister detector(s) as a dynamic Trigger-B fallback.
    """
    region_cfg = config["pipeline"]
    fallback_cfg = config.get("fallback_policy", {}) or {}
    flat_registry = _flatten_registry(registry)
    available = list(flat_registry.keys())

    router_detections = router_result.get("detections", [])
    router_status = str(router_result.get("route_decision", {}).get("status", "unknown"))

    detector_outputs: dict[str, list[Detection]] = {}
    warnings: list[str] = []

    def _ensure_outputs(detector_name: str, source_router_class: str) -> list[Detection]:
        if detector_name in detector_outputs:
            return detector_outputs[detector_name]
        detector = flat_registry.get(detector_name)
        if detector is None:
            detector_outputs[detector_name] = []
            return []
        try:
            outputs = [
                Detection(
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    grade=det.grade,
                    source_model=det.source_model,
                    source_router_class=source_router_class,
                )
                for det in detector.predict(image, source_router_class)
            ]
        except Exception as exc:
            warnings.append(f"detector_exception:{detector_name}:{type(exc).__name__}")
            outputs = []
        detector_outputs[detector_name] = outputs
        return outputs

    aggregated: dict[tuple[str, int], dict[str, Any]] = {}

    def _apply_tasks(tasks: list[Task]) -> None:
        for task in tasks:
            outputs = _ensure_outputs(task.detector_name, task.source_router_class)
            accepted = False
            for det_index, det in enumerate(outputs):
                if det.confidence < task.min_confidence:
                    continue
                if not detection_in_router_region(det.xyxy, task.filter_box, region_cfg):
                    continue
                accepted = True
                key = (task.detector_name, det_index)
                entry = aggregated.get(key)
                if entry is None:
                    entry = {
                        "detection": det,
                        "router_region_indices": list(task.router_region_indices),
                        "filter_boxes": [list(task.filter_box)],
                        "task_kinds": ["fallback" if task.is_fallback else "main"],
                        "fallback_reasons": [task.fallback_reason] if task.is_fallback else [],
                        "source_router_class": task.source_router_class,
                        "is_fallback": task.is_fallback,
                        "fallback_reason": task.fallback_reason,
                        "min_confidence": task.min_confidence,
                    }
                    aggregated[key] = entry
                    continue
                for idx in task.router_region_indices:
                    if idx not in entry["router_region_indices"]:
                        entry["router_region_indices"].append(idx)
                entry["filter_boxes"].append(list(task.filter_box))
                entry["task_kinds"].append("fallback" if task.is_fallback else "main")
                if not task.is_fallback:
                    entry["is_fallback"] = False
                    entry["fallback_reason"] = ""
                    entry["source_router_class"] = task.source_router_class
                    entry["min_confidence"] = min(entry["min_confidence"], task.min_confidence)
                else:
                    if task.fallback_reason and task.fallback_reason not in entry["fallback_reasons"]:
                        entry["fallback_reasons"].append(task.fallback_reason)
            if accepted or task.is_fallback:
                continue
            dynamic_outputs = _dynamic_empty_detector_outputs_for_region(
                detector=flat_registry.get(task.detector_name),
                image_bgr=image,
                source_router_class=task.source_router_class,
                cfg=config.get("downstream_empty_fallback", {}) or {},
                filter_box=task.filter_box,
                region_cfg=region_cfg,
            )
            dynamic_count = 0
            max_dynamic = int((config.get("downstream_empty_fallback", {}) or {}).get("max_outputs_per_region", 1) or 0)
            for det_index, det in enumerate(dynamic_outputs):
                key = (task.detector_name, -100000 - len(aggregated) - det_index)
                aggregated[key] = {
                    "detection": det,
                    "router_region_indices": list(task.router_region_indices),
                    "filter_boxes": [list(task.filter_box)],
                    "task_kinds": ["dynamic_empty_threshold"],
                    "fallback_reasons": ["dynamic_empty_threshold"],
                    "source_router_class": task.source_router_class,
                    "is_fallback": True,
                    "fallback_reason": "dynamic_empty_threshold",
                    "min_confidence": float(det.confidence),
                }
                dynamic_count += 1
                if max_dynamic > 0 and dynamic_count >= max_dynamic:
                    break

    main_tasks = plan_main_tasks(
        router_detections=router_detections,
        image_shape=image.shape,
        region_cfg=region_cfg,
        fallback_cfg=fallback_cfg,
        available_detector_names=available,
    )
    static_fallback_tasks = plan_static_fallback_tasks(
        router_detections=router_detections,
        router_status=router_status,
        image_shape=image.shape,
        region_cfg=region_cfg,
        fallback_cfg=fallback_cfg,
        available_detector_names=available,
    )
    _apply_tasks(main_tasks + static_fallback_tasks)

    max_main_conf_by_region = _max_main_conf_per_region(aggregated, router_detections)
    dynamic_fallback_tasks = plan_dynamic_fallback_tasks(
        router_detections=router_detections,
        max_main_conf_by_region=max_main_conf_by_region,
        image_shape=image.shape,
        region_cfg=region_cfg,
        fallback_cfg=fallback_cfg,
        available_detector_names=available,
    )
    if dynamic_fallback_tasks:
        _apply_tasks(dynamic_fallback_tasks)

    all_cracks: list[Detection] = []
    raw_crack_records: list[dict[str, Any]] = []
    fallback_count = 0
    fallback_reasons_summary: dict[str, int] = {}
    for (detector_name, det_index), entry in aggregated.items():
        det: Detection = entry["detection"]
        promoted = Detection(
            xyxy=det.xyxy,
            confidence=det.confidence,
            grade=det.grade,
            source_model=det.source_model,
            source_router_class=entry["source_router_class"],
        )
        all_cracks.append(promoted)
        primary_router_index = next(iter(entry["router_region_indices"]), -1)
        primary_router_box = (
            router_detections[primary_router_index]["bbox_xyxy"]
            if 0 <= primary_router_index < len(router_detections)
            else []
        )
        primary_router_conf = (
            float(router_detections[primary_router_index]["confidence"])
            if 0 <= primary_router_index < len(router_detections)
            else 0.0
        )
        raw_record = detection_dict(promoted)
        raw_record.update(
            {
                "router_region_index": primary_router_index,
                "router_region_indices": list(entry["router_region_indices"]),
                "router_bbox_xyxy": [round(float(v), 3) for v in primary_router_box],
                "router_filter_bbox_xyxy": [round(float(v), 3) for v in entry["filter_boxes"][0]],
                "router_confidence": primary_router_conf,
                "router_class_name": entry["source_router_class"],
                "detector_input_shape": list(image.shape),
                "region_transport": "full_image_filter",
                "is_fallback": bool(entry["is_fallback"]),
                "fallback_reasons": list(entry["fallback_reasons"]),
                "task_kinds": list(entry["task_kinds"]),
                "min_confidence_floor": float(entry["min_confidence"]),
            }
        )
        raw_crack_records.append(raw_record)
        if entry["is_fallback"]:
            fallback_count += 1
            for reason in entry["fallback_reasons"]:
                family = reason.split(":", 1)[0] if reason else "unknown"
                fallback_reasons_summary[family] = fallback_reasons_summary.get(family, 0) + 1

    if fallback_count:
        warnings.append(f"fallback_detections:{fallback_count}")
        for family, count in sorted(fallback_reasons_summary.items()):
            warnings.append(f"fallback_trigger:{family}:{count}")

    return all_cracks, raw_crack_records, warnings


def _max_main_conf_per_region(
    aggregated: dict[tuple[str, int], dict[str, Any]],
    router_detections: list[dict[str, Any]],
) -> dict[int, float]:
    out: dict[int, float] = {i: 0.0 for i in range(len(router_detections))}
    for entry in aggregated.values():
        if entry.get("is_fallback"):
            continue
        det: Detection = entry["detection"]
        conf = float(det.confidence)
        for idx in entry["router_region_indices"]:
            if idx < 0 or idx >= len(router_detections):
                continue
            if conf > out.get(idx, 0.0):
                out[idx] = conf
    return out


def save_visualization(image_path: Path, result: dict[str, Any], out_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for det in (result.get("router") or {}).get("detections", []):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 180, 90), 3)
        labels.append((f"R:{display_label(det['class_name'])} {det['confidence']:.2f}", (x1, max(2, y1 - 28)), (30, 180, 90)))
    for det in result.get("display_crack_detections") or result.get("crack_detections", []):
        if "bbox_xyxy" not in det:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 70, 230), 2)
        labels.append((f"{display_label(det['damage_grade'])} {det['confidence']:.2f}", (x1, min(image.shape[0] - 24, y2 + 4)), (40, 70, 230)))
    image = draw_unicode_labels(image, labels)
    cv2.imwrite(str(out_path), image)


def display_label(value: Any) -> str:
    return {
        "壁类": "壁類",
        "天井": "天井",
        "RC柱": "RC柱",
        "内壁": "内壁",
        "RC壁": "RC壁",
        "壁-B": "壁-B",
        "壁-C": "壁-C",
        "壁-D": "壁-D",
    }.get(str(value), str(value))


def draw_unicode_labels(
    image_bgr: Any,
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
) -> Any:
    if not labels:
        return image_bgr
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = visualization_font(22)
    for text, (x, y), bgr in labels:
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x = max(0, min(int(x), pil_image.width - width - 6))
        y = max(0, min(int(y), pil_image.height - height - 6))
        draw.rectangle((x, y, x + width + 6, y + height + 6), fill=(255, 255, 255), outline=rgb, width=2)
        draw.text((x + 3, y + 2), text, font=font, fill=rgb)
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def visualization_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def count_many(values: Any) -> dict[str, int]:
    return count_values(values)


def router_status(result: dict[str, Any]) -> str:
    router = result.get("router")
    if not router:
        return "error"
    return str(router.get("route_decision", {}).get("status", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
