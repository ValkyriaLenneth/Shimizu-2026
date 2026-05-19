"""End-to-end router + crack detection pipeline runner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from .crack_detector_registry import build_detector_registry
from .region_view import make_region_view, map_region_xyxy_to_original, padded_xyxy
from .result_merge import Detection, center_in_xyxy, ioa_over_first_xyxy, nms_detections, prod_like_merge_detections
from .router_infer import RouterConfig, RouterInfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="router_crack_pipeline/configs/pipeline.default.yaml")
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
    router = RouterInfer(
        RouterConfig(
            weights=resolve_path(pipeline_cfg["router_weights"], config_path.parent),
            yolo_root=resolve_path(config.get("yolo_root", "../coarse_router_yolov9/yolov9"), config_path.parent),
            data_yaml=resolve_path(pipeline_cfg.get("router_data_yaml", "../coarse_router_yolov9/datasets/coarse_router_3class_cleaned/data.yaml"), config_path.parent),
            conf_threshold=float(pipeline_cfg.get("router_conf_threshold", 0.25)),
            low_conf_threshold=float(pipeline_cfg.get("router_low_conf_threshold", 0.10)),
            iou_threshold=float(pipeline_cfg.get("router_iou_threshold", 0.45)),
            device=args.device,
            imgsz=int(pipeline_cfg.get("imgsz", 640)),
        )
    )

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
        full_image_outputs: dict[str, list[Detection]] = {}
        for router_region_index, router_det in enumerate(router_result["detections"]):
            router_class = router_det["class_name"]
            detectors = registry.get(router_class, [])
            filter_box = tuple(
                float(v)
                for v in padded_xyxy(
                    router_det["bbox_xyxy"],
                    image.shape,
                    padding_ratio=float(region_cfg.get("region_padding_ratio", 0.10)),
                )
            )
            for detector in detectors:
                detector_name = getattr(detector, "name", "unknown")
                if detector_name not in full_image_outputs:
                    try:
                        full_image_outputs[detector_name] = [
                            Detection(
                                xyxy=det.xyxy,
                                confidence=det.confidence,
                                grade=det.grade,
                                source_model=det.source_model,
                                source_router_class=router_class,
                            )
                            for det in detector.predict(image, router_class)
                        ]
                    except Exception as exc:
                        warnings.append(f"detector_exception:{detector_name}:{type(exc).__name__}")
                        full_image_outputs[detector_name] = []
                for det in full_image_outputs[detector_name]:
                    if not detection_in_router_region(det.xyxy, filter_box, region_cfg):
                        continue
                    mapped_detection = Detection(
                        xyxy=det.xyxy,
                        confidence=det.confidence,
                        grade=det.grade,
                        source_model=det.source_model,
                        source_router_class=router_class,
                    )
                    all_cracks.append(mapped_detection)
                    raw_record = detection_dict(mapped_detection)
                    raw_record.update(
                        {
                            "router_region_index": router_region_index,
                            "router_bbox_xyxy": [round(float(v), 3) for v in router_det["bbox_xyxy"]],
                            "router_filter_bbox_xyxy": [round(float(v), 3) for v in filter_box],
                            "router_confidence": router_det["confidence"],
                            "router_class_name": router_class,
                            "detector_input_shape": list(image.shape),
                            "region_transport": "full_image_filter",
                        }
                    )
                    raw_crack_records.append(raw_record)
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
    if router_result["route_decision"]["status"] == "low_confidence":
        warnings.append("router_low_confidence_multi_model_fallback_todo")
    if not router_result["detections"]:
        warnings.append("router_unknown")

    return {
        "image": str(image_path),
        "image_shape": list(image.shape),
        "pipeline_version": "router3_crack_v1",
        "router": router_result,
        "raw_crack_detections": raw_crack_records,
        "crack_detections": [detection_dict(d) for d in merged],
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


def save_visualization(image_path: Path, result: dict[str, Any], out_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for det in (result.get("router") or {}).get("detections", []):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 180, 90), 3)
        cv2.putText(image, f"R:{det['class_name']} {det['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 180, 90), 2)
    for det in result.get("crack_detections", []):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 70, 230), 2)
        cv2.putText(image, f"{det['damage_grade']} {det['confidence']:.2f}", (x1, min(image.shape[0] - 8, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 70, 230), 2)
    cv2.imwrite(str(out_path), image)


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
