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
from .result_merge import Detection, center_in_xyxy, ioa_over_first_xyxy, nms_detections, prod_like_merge_detections
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


def run_one(image_path: Path, router: RouterInfer, registry: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"image": str(image_path), "error": "image_unreadable", "router": None, "raw_crack_detections": [], "crack_detections": [], "warnings": ["image_unreadable"]}
    router_result = router.predict(image)
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
        max_single_groups_per_model=int(wall_display_cfg.get("max_single_groups_per_model", 4)),
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
        wall_display=wall_candidate_display["display_detections"],
        ambiguity_display=ambiguity_display,
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
            for det_index, det in enumerate(outputs):
                if det.confidence < task.min_confidence:
                    continue
                if not detection_in_router_region(det.xyxy, task.filter_box, region_cfg):
                    continue
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
