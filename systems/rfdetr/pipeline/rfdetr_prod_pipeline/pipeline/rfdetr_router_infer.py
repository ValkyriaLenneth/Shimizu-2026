"""RF-DETR router inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rfdetr_backend import RfdetrBackend, RfdetrDetection


@dataclass(frozen=True)
class RfdetrRouterConfig:
    checkpoint: Path
    class_names: dict[int, str]
    conf_threshold: float = 0.25
    low_conf_threshold: float = 0.10
    device: str = "cpu"
    max_det: int = 20


class RfdetrRouterInfer:
    def __init__(self, config: RfdetrRouterConfig) -> None:
        self.config = config
        self.backend = RfdetrBackend(config.checkpoint, config.class_names, device=config.device)

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        high = self.backend.predict(
            image_bgr,
            thresholds=[self.config.conf_threshold] * len(self.config.class_names),
            max_det=self.config.max_det,
        )
        low = []
        if not high and self.config.low_conf_threshold < self.config.conf_threshold:
            low = self.backend.predict(
                image_bgr,
                thresholds=[self.config.low_conf_threshold] * len(self.config.class_names),
                max_det=self.config.max_det,
            )
        detections = high or low
        status = route_status(high, low)
        return {
            "router_model": str(self.config.checkpoint),
            "classes": self.config.class_names,
            "detections": [router_detection_to_dict(d, image_bgr.shape) for d in detections],
            "route_decision": {
                "status": status,
                "strategy": "keep_all_router_boxes",
                "primary_class": detections[0].class_name if detections else None,
                "low_confidence_fallback_todo": status == "low_confidence",
            },
        }


def route_status(high: list[RfdetrDetection], low: list[RfdetrDetection]) -> str:
    if high:
        return "ok"
    if low:
        return "low_confidence"
    return "unknown"


def router_detection_to_dict(det: RfdetrDetection, image_shape: tuple[int, ...]) -> dict[str, Any]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = det.xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return {
        "bbox_xyxy": list(det.xyxy),
        "confidence": det.confidence,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "area_ratio": area / float(w * h) if w > 0 and h > 0 else 0.0,
    }
