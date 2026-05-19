"""Router inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .yolov9_backend import YoloDetection, YoloV9Backend, detection_to_dict


@dataclass(frozen=True)
class RouterConfig:
    weights: Path
    yolo_root: Path
    data_yaml: Path | None
    conf_threshold: float = 0.25
    low_conf_threshold: float = 0.10
    iou_threshold: float = 0.45
    device: str = "cpu"
    imgsz: int = 640
    max_det: int = 20


class RouterInfer:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self.backend = YoloV9Backend(
            weights=config.weights,
            yolo_root=config.yolo_root,
            data=config.data_yaml,
            device=config.device,
            imgsz=config.imgsz,
        )

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        high = self.backend.predict(
            image_bgr,
            conf_thres=self.config.conf_threshold,
            iou_thres=self.config.iou_threshold,
            max_det=self.config.max_det,
        )
        low = []
        if not high and self.config.low_conf_threshold < self.config.conf_threshold:
            low = self.backend.predict(
                image_bgr,
                conf_thres=self.config.low_conf_threshold,
                iou_thres=self.config.iou_threshold,
                max_det=self.config.max_det,
            )
        detections = high or low
        status = route_status(high, low)
        return {
            "router_model": str(self.config.weights),
            "classes": self.backend.names,
            "detections": [router_detection_to_dict(d, image_bgr.shape) for d in detections],
            "route_decision": {
                "status": status,
                "strategy": "keep_all_router_boxes",
                "primary_class": detections[0].class_name if detections else None,
                "low_confidence_fallback_todo": status == "low_confidence",
            },
        }


def route_status(high: list[YoloDetection], low: list[YoloDetection]) -> str:
    if high:
        return "ok"
    if low:
        return "low_confidence"
    return "unknown"


def router_detection_to_dict(det: YoloDetection, image_shape: tuple[int, ...]) -> dict[str, Any]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = det.xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    base = detection_to_dict(det)
    base["area_ratio"] = area / float(w * h) if w > 0 and h > 0 else 0.0
    return base
