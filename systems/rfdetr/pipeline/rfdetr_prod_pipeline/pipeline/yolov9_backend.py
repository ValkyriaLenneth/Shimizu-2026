"""Small YOLOv9 inference wrapper for ndarray inputs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class YoloDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YoloV9Backend:
    """Run YOLOv9 DetectMultiBackend on an in-memory BGR image."""

    def __init__(
        self,
        weights: str | Path,
        yolo_root: str | Path,
        data: str | Path | None = None,
        device: str = "cpu",
        imgsz: int = 640,
        half: bool = False,
    ) -> None:
        self.weights = Path(weights).resolve()
        self.yolo_root = Path(yolo_root).resolve()
        self.data = None if data is None else str(Path(data).resolve())
        self.device_name = device
        self.imgsz = imgsz
        self.half = half
        self._load_yolov9_symbols()

        self.device = self.select_device(device)
        self.model = self.DetectMultiBackend(str(self.weights), device=self.device, data=self.data, fp16=half)
        self.stride = int(self.model.stride)
        self.names = dict(self.model.names)
        self.imgsz = int(self.check_img_size(imgsz, s=self.stride))
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

    def _load_yolov9_symbols(self) -> None:
        root = str(self.yolo_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from models.common import DetectMultiBackend  # type: ignore
        from utils.augmentations import letterbox  # type: ignore
        from utils.general import check_img_size, non_max_suppression, scale_boxes  # type: ignore
        from utils.torch_utils import select_device  # type: ignore

        self.DetectMultiBackend = DetectMultiBackend
        self.letterbox = letterbox
        self.check_img_size = check_img_size
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.select_device = select_device

    @torch.inference_mode()
    def predict(
        self,
        image_bgr: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        agnostic_nms: bool = False,
    ) -> list[YoloDetection]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got shape={image_bgr.shape}")

        im0 = image_bgr
        im = self.letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        tensor = torch.from_numpy(im).to(self.model.device)
        tensor = tensor.half() if self.model.fp16 else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor[None]

        pred = self.model(tensor)
        pred = self.non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)
        det = pred[0]
        if det is None or len(det) == 0:
            return []
        det[:, :4] = self.scale_boxes(tensor.shape[2:], det[:, :4], im0.shape).round()

        detections: list[YoloDetection] = []
        for *xyxy, conf, cls in det.detach().cpu().tolist():
            cls_id = int(cls)
            detections.append(
                YoloDetection(
                    xyxy=tuple(float(v) for v in xyxy),  # type: ignore[arg-type]
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=str(self.names.get(cls_id, cls_id)),
                )
            )
        return detections


def detection_to_dict(det: YoloDetection) -> dict[str, Any]:
    return {
        "bbox_xyxy": list(det.xyxy),
        "confidence": det.confidence,
        "class_id": det.class_id,
        "class_name": det.class_name,
    }
