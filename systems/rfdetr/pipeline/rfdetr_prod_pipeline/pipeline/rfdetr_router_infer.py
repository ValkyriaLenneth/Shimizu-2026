"""RF-DETR router inference wrapper."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


@dataclass(frozen=True)
class PrecisionGateConfig:
    primary_threshold: float
    confirmation_model: str | None = None
    confirmation_threshold: float | None = None
    gate_iou: float | None = None
    bypass_primary: float | None = None


@dataclass(frozen=True)
class RfdetrPrecisionEnsembleConfig:
    primary_checkpoint: Path
    confirmation_checkpoints: dict[str, Path]
    class_names: dict[int, str]
    operating_points: dict[int, PrecisionGateConfig]
    devices: dict[str, str]
    max_det: int = 20
    candidate_max_det: int = 300
    parallel: bool = True
    lazy_confirmation_models: frozenset[str] = frozenset()


class RouterBackend(Protocol):
    def predict(
        self,
        image_bgr: np.ndarray,
        thresholds: list[float] | tuple[float, ...],
        max_det: int = 1000,
    ) -> list[RfdetrDetection]: ...


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


class RfdetrPrecisionEnsembleInfer:
    """Precision-first Router whose confirmation models may only reject boxes."""

    def __init__(
        self,
        config: RfdetrPrecisionEnsembleConfig,
        *,
        backends: dict[str, RouterBackend] | None = None,
    ) -> None:
        self.config = config
        self.backends = backends or self._build_backends()
        expected = {"primary", *config.confirmation_checkpoints}
        missing = expected - set(self.backends)
        if missing:
            raise ValueError(f"missing Router ensemble backends: {sorted(missing)}")
        if set(config.operating_points) != set(config.class_names):
            raise ValueError("operating points must cover every Router class")
        for class_id, point in config.operating_points.items():
            if point.confirmation_model is None:
                continue
            if point.confirmation_model not in config.confirmation_checkpoints:
                raise ValueError(f"class {class_id} uses unknown confirmation model")
            if point.confirmation_threshold is None or point.gate_iou is None or point.bypass_primary is None:
                raise ValueError(f"class {class_id} has an incomplete confirmation gate")

    def _build_backends(self) -> dict[str, RouterBackend]:
        backends: dict[str, RouterBackend] = {
            "primary": RfdetrBackend(
                self.config.primary_checkpoint,
                self.config.class_names,
                device=self.config.devices.get("primary", "cpu"),
            )
        }
        for name, checkpoint in self.config.confirmation_checkpoints.items():
            backends[name] = RfdetrBackend(
                checkpoint,
                self.config.class_names,
                device=self.config.devices.get(name, self.config.devices.get("primary", "cpu")),
            )
        return backends

    def _thresholds(self) -> dict[str, list[float]]:
        primary = [self.config.operating_points[class_id].primary_threshold for class_id in self.config.class_names]
        thresholds = {"primary": primary}
        for name in self.config.confirmation_checkpoints:
            active = [
                class_id
                for class_id, point in self.config.operating_points.items()
                if point.confirmation_model == name
            ]
            size = max(active, default=-1) + 1
            values = [1.01] * size
            for class_id in active:
                value = self.config.operating_points[class_id].confirmation_threshold
                assert value is not None
                values[class_id] = value
            thresholds[name] = values
        return thresholds

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        thresholds = self._thresholds()

        def infer(name: str) -> tuple[str, list[RfdetrDetection]]:
            return name, self.backends[name].predict(
                image_bgr,
                thresholds[name],
                max_det=self.config.candidate_max_det,
            )

        eager_names = [
            name
            for name in self.backends
            if name == "primary" or name not in self.config.lazy_confirmation_models
        ]
        if self.config.parallel and len(eager_names) > 1:
            with ThreadPoolExecutor(max_workers=len(eager_names)) as executor:
                predictions = dict(executor.map(infer, eager_names))
        else:
            predictions = dict(infer(name) for name in eager_names)

        needed_lazy = {
            point.confirmation_model
            for candidate in predictions["primary"]
            for point in [self.config.operating_points[candidate.class_id]]
            if point.confirmation_model in self.config.lazy_confirmation_models
            and point.bypass_primary is not None
            and candidate.confidence < point.bypass_primary
        }
        lazy_predictions = dict(infer(name) for name in sorted(needed_lazy) if name is not None)
        predictions.update(lazy_predictions)
        for name in self.config.lazy_confirmation_models:
            predictions.setdefault(name, [])

        selected = []
        for candidate in predictions["primary"]:
            point = self.config.operating_points[candidate.class_id]
            if point.confirmation_model is None:
                selected.append(candidate)
                continue
            assert point.bypass_primary is not None
            if candidate.confidence >= point.bypass_primary:
                selected.append(candidate)
                continue
            assert point.confirmation_threshold is not None and point.gate_iou is not None
            support = max(
                (
                    det.confidence
                    for det in predictions[point.confirmation_model]
                    if det.class_id == candidate.class_id
                    and box_iou(candidate.xyxy, det.xyxy) >= point.gate_iou
                ),
                default=0.0,
            )
            if support >= point.confirmation_threshold:
                selected.append(candidate)

        selected.sort(key=lambda item: item.confidence, reverse=True)
        selected = selected[: self.config.max_det]
        status = "ok" if selected else "unknown"
        model_paths = {
            "primary": str(self.config.primary_checkpoint),
            **{name: str(path) for name, path in self.config.confirmation_checkpoints.items()},
        }
        return {
            "router_model": str(self.config.primary_checkpoint),
            "router_models": model_paths,
            "classes": self.config.class_names,
            "detections": [router_detection_to_dict(d, image_bgr.shape) for d in selected],
            "route_decision": {
                "status": status,
                "strategy": "primary_boxes_with_specialized_confirmation",
                "primary_class": selected[0].class_name if selected else None,
                "low_confidence_fallback_todo": False,
            },
        }


def route_status(high: list[RfdetrDetection], low: list[RfdetrDetection]) -> str:
    if high:
        return "ok"
    if low:
        return "low_confidence"
    return "unknown"


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


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
