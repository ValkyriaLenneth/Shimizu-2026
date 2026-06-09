"""Small RF-DETR inference wrapper for ndarray inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RfdetrDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class RfdetrBackend:
    """Run an RF-DETR checkpoint on an in-memory BGR image."""

    def __init__(
        self,
        checkpoint: str | Path,
        class_names: dict[int, str] | None = None,
        device: str | None = None,
    ) -> None:
        import rfdetr

        self.checkpoint = Path(checkpoint).resolve()
        self.class_names = class_names or {}
        self.model = rfdetr.from_checkpoint(str(self.checkpoint))
        if device:
            model_ctx = getattr(self.model, "model", None)
            if model_ctx is not None and hasattr(model_ctx, "device"):
                import torch

                model_ctx.device = torch.device(device)

    def predict(
        self,
        image_bgr: np.ndarray,
        thresholds: list[float] | tuple[float, ...],
        max_det: int = 1000,
    ) -> list[RfdetrDetection]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got shape={image_bgr.shape}")
        min_threshold = min(float(value) for value in thresholds)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        raw = self.model.predict(image, threshold=min_threshold, include_source_image=False)
        xyxy = np.asarray(raw.xyxy)
        conf = np.asarray(raw.confidence)
        class_id = np.asarray(raw.class_id)

        detections: list[RfdetrDetection] = []
        for box, score, cls in zip(xyxy, conf, class_id, strict=False):
            cls_id = int(cls)
            if cls_id < 0 or cls_id >= len(thresholds):
                continue
            confidence = float(score)
            if confidence < float(thresholds[cls_id]):
                continue
            detections.append(
                RfdetrDetection(
                    xyxy=tuple(float(v) for v in box),  # type: ignore[arg-type]
                    confidence=confidence,
                    class_id=cls_id,
                    class_name=str(self.class_names.get(cls_id, cls_id)),
                )
            )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[:max_det]
