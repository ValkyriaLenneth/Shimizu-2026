"""Crack detector registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .result_merge import Detection
from .yolov9_backend import YoloV9Backend


class CrackDetector(Protocol):
    name: str

    def predict(self, image_bgr: np.ndarray, source_router_class: str) -> list[Detection]:
        ...


@dataclass
class NoOpCrackDetector:
    name: str = "noop"

    def predict(self, image_bgr: np.ndarray, source_router_class: str) -> list[Detection]:
        return []


@dataclass
class MockCrackDetector:
    name: str = "mock"

    def predict(self, image_bgr: np.ndarray, source_router_class: str) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        if h < 4 or w < 4:
            return []
        return [
            Detection(
                xyxy=(w * 0.35, h * 0.45, w * 0.65, h * 0.55),
                confidence=0.50,
                grade="C",
                source_model=self.name,
                source_router_class=source_router_class,
            )
        ]


class YoloCrackDetector:
    def __init__(
        self,
        name: str,
        weights: str | Path,
        yolo_root: str | Path,
        data_yaml: str | Path | None,
        device: str,
        imgsz: int,
        conf_threshold: float,
        iou_threshold: float,
        grade_names: dict[int, str] | None = None,
    ) -> None:
        self.name = name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.grade_names = grade_names or {}
        self.backend = YoloV9Backend(weights, yolo_root, data_yaml, device=device, imgsz=imgsz)

    def predict(self, image_bgr: np.ndarray, source_router_class: str) -> list[Detection]:
        outputs = []
        for det in self.backend.predict(image_bgr, self.conf_threshold, self.iou_threshold):
            grade = self.grade_names.get(det.class_id, det.class_name)
            outputs.append(
                Detection(
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    grade=str(grade),
                    source_model=self.name,
                    source_router_class=source_router_class,
                )
            )
        return outputs


def build_detector_registry(config: dict[str, Any], root: Path, mock: bool = False) -> dict[str, list[CrackDetector]]:
    if mock:
        detector = MockCrackDetector()
        return {"天井": [detector], "壁类": [detector], "RC柱": [detector]}

    yolo_root = resolve_path(config.get("yolo_root", "../coarse_router_yolov9/yolov9"), root)
    device = str(config.get("device", "cpu"))
    imgsz = int(config.get("imgsz", 640))
    conf = float(config.get("conf_threshold", 0.25))
    iou = float(config.get("iou_threshold", 0.45))
    models = config.get("crack_models") or {}

    detectors: dict[str, list[CrackDetector]] = {
        "天井": detector_for("ceiling", models.get("ceiling"), yolo_root, device, imgsz, conf, iou, root),
        "RC柱": detector_for("rc_column", models.get("rc_column"), yolo_root, device, imgsz, conf, iou, root),
        "壁类": wall_detectors(models, yolo_root, device, imgsz, conf, iou, root),
    }
    return detectors


def detector_for(
    name: str,
    weight_value: str | None,
    yolo_root: Path,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    root: Path,
) -> list[CrackDetector]:
    if not weight_value:
        return [NoOpCrackDetector(name=f"noop_{name}")]
    return [YoloCrackDetector(name, resolve_path(weight_value, root), yolo_root, None, device, imgsz, conf, iou)]


def wall_detectors(
    models: dict[str, Any],
    yolo_root: Path,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    root: Path,
) -> list[CrackDetector]:
    if models.get("wall_merged"):
        return detector_for("wall_merged", models.get("wall_merged"), yolo_root, device, imgsz, conf, iou, root)
    detectors: list[CrackDetector] = []
    for name in ["inner_wall", "rc_wall"]:
        detectors.extend(detector_for(name, models.get(name), yolo_root, device, imgsz, conf, iou, root))
    return detectors or [NoOpCrackDetector(name="noop_wall")]


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
