#!/usr/bin/env python3
"""Generate report visuals for RF-DETR downstream expansion."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path("coarse_router_yolov9/yolov9").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))
from models.experimental import attempt_load  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_boxes  # noqa: E402

from evaluate_rfdetr_threshold_sweep import (  # noqa: E402
    IMAGE_EXTS,
    Prediction,
    Target,
    box_iou,
    read_targets,
)


@dataclass(frozen=True)
class Component:
    key: str
    label: str
    dataset_dir: Path
    yolo_weights: Path
    rfdetr_checkpoint: Path
    thresholds: tuple[float, float, float]
    yolo_precision: float
    yolo_recall: float
    rfdetr_precision: float
    rfdetr_recall: float


COMPONENTS = [
    Component(
        key="tenjo",
        label="天井",
        dataset_dir=Path("data/rfdetr_tenjo_all_non_legacy_test_v1"),
        yolo_weights=Path("downloads/previous_phase_gpl_model_unpacked/infer_models/TIANJING.pt"),
        rfdetr_checkpoint=Path("rfdetr_threshold_tuned_models_20260609/checkpoints/tenjo_standard_orig_checkpoint_epoch_009.pth"),
        thresholds=(0.25, 0.35, 0.35),
        yolo_precision=0.593,
        yolo_recall=0.845,
        rfdetr_precision=0.650,
        rfdetr_recall=0.812,
    ),
    Component(
        key="rc_wall",
        label="RC壁",
        dataset_dir=Path("data/rfdetr_rc_wall_all_non_legacy_test_v1"),
        yolo_weights=Path("downloads/previous_phase_gpl_model_unpacked/infer_models/RCBI.pt"),
        rfdetr_checkpoint=Path("rfdetr_threshold_tuned_models_20260609/checkpoints/rc_wall_checkpoint_epoch_009.pth"),
        thresholds=(0.28, 0.45, 0.25),
        yolo_precision=0.585,
        yolo_recall=0.720,
        rfdetr_precision=0.632,
        rfdetr_recall=0.750,
    ),
    Component(
        key="inner_wall",
        label="内壁",
        dataset_dir=Path("data/rfdetr_inner_wall_all_non_legacy_test_v1"),
        yolo_weights=Path("downloads/previous_phase_gpl_model_unpacked/infer_models/NEIBI.pt"),
        rfdetr_checkpoint=Path("rfdetr_threshold_tuned_models_20260609/checkpoints/inner_wall_checkpoint_epoch_026.pth"),
        thresholds=(0.40, 0.40, 0.40),
        yolo_precision=0.636,
        yolo_recall=0.750,
        rfdetr_precision=0.824,
        rfdetr_recall=0.848,
    ),
]


CLASS_NAMES = ["B", "C", "D"]
COLORS = {
    "gt": (245, 91, 91),
    "yolo": (238, 160, 42),
    "rfdetr": (49, 130, 206),
}


def font(size: int = 22) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def run_yolo(model, image: Image.Image, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640) -> list[Prediction]:
    arr = np.asarray(image)
    im = letterbox(arr, imgsz, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    tensor = torch.from_numpy(im).float() / 255.0
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)[0]
        det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=300)[0]
    if det is None or len(det) == 0:
        return []
    det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], arr.shape).round()
    preds = [
        Prediction(cls=int(cls), conf=float(score), xyxy=tuple(float(x) for x in xyxy))
        for *xyxy, score, cls in det.tolist()
        if 0 <= int(cls) < 3
    ]
    preds.sort(key=lambda item: item.conf, reverse=True)
    return preds


def run_rfdetr(model, image: Image.Image, thresholds: tuple[float, float, float]) -> list[Prediction]:
    import rfdetr  # noqa: F401

    detections = model.predict(image, threshold=min(thresholds), include_source_image=False)
    xyxy = np.asarray(detections.xyxy)
    conf = np.asarray(detections.confidence)
    class_id = np.asarray(detections.class_id)
    preds: list[Prediction] = []
    for box, score, cls in zip(xyxy, conf, class_id, strict=False):
        cls_i = int(cls)
        if cls_i < 0 or cls_i >= 3 or float(score) < thresholds[cls_i]:
            continue
        preds.append(Prediction(cls=cls_i, conf=float(score), xyxy=tuple(float(x) for x in box)))
    preds.sort(key=lambda item: item.conf, reverse=True)
    return preds


def match_targets(targets: list[Target], preds: list[Prediction], iou_threshold: float = 0.229) -> dict[int, tuple[Prediction, float]]:
    matched: dict[int, tuple[Prediction, float]] = {}
    used_preds: set[int] = set()
    for target_idx, target in enumerate(targets):
        candidates = []
        for pred_idx, pred in enumerate(preds):
            if pred_idx in used_preds or pred.cls != target.cls:
                continue
            iou = box_iou(target.xyxy, pred.xyxy)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, pred))
        if candidates:
            iou, pred_idx, pred = max(candidates, key=lambda item: item[0])
            used_preds.add(pred_idx)
            matched[target_idx] = (pred, iou)
    return matched


def draw_panel(image: Image.Image, title: str, targets: list[Target], preds: list[Prediction], pred_color: tuple[int, int, int]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    label_font = font(18)
    title_font = font(24)
    for target in targets:
        draw.rectangle(target.xyxy, outline=COLORS["gt"], width=4)
        draw.text((target.xyxy[0] + 4, target.xyxy[1] + 4), f"GT {CLASS_NAMES[target.cls]}", fill=COLORS["gt"], font=label_font)
    for pred in preds:
        draw.rectangle(pred.xyxy, outline=pred_color, width=3)
        draw.text((pred.xyxy[0] + 4, max(0, pred.xyxy[1] - 22)), f"{CLASS_NAMES[pred.cls]} {pred.conf:.2f}", fill=pred_color, font=label_font)
    header_h = 42
    out = Image.new("RGB", (canvas.width, canvas.height + header_h), "white")
    out.paste(canvas, (0, header_h))
    ImageDraw.Draw(out).text((12, 8), title, fill=(20, 30, 40), font=title_font)
    return out


def save_case_image(output_path: Path, image: Image.Image, targets: list[Target], yolo_preds: list[Prediction], rfdetr_preds: list[Prediction]) -> None:
    max_w = 760
    scale = min(1.0, max_w / image.width)
    if scale < 1:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size)
        def scale_box(box):
            return tuple(v * scale for v in box)
        targets = [Target(t.cls, scale_box(t.xyxy)) for t in targets]
        yolo_preds = [Prediction(p.cls, p.conf, scale_box(p.xyxy)) for p in yolo_preds]
        rfdetr_preds = [Prediction(p.cls, p.conf, scale_box(p.xyxy)) for p in rfdetr_preds]
    left = draw_panel(image, "YOLO9", targets, yolo_preds, COLORS["yolo"])
    right = draw_panel(image, "RF-DETR", targets, rfdetr_preds, COLORS["rfdetr"])
    gap = 16
    out = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), "white")
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, quality=92)


def make_metric_charts(output_dir: Path) -> None:
    labels = ["Ceiling", "RC wall", "Inner wall"]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, [item.yolo_precision for item in COMPONENTS], width, label="YOLO9 Precision", color="#dd8a26")
    ax.bar(x + width / 2, [item.rfdetr_precision for item in COMPONENTS], width, label="RF-DETR Precision", color="#2b6cb0")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Precision")
    ax.set_title("Precision comparison on official test split")
    ax.set_xticks(x, labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "downstream_precision_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, [item.yolo_recall for item in COMPONENTS], width, label="YOLO9 Recall", color="#dd8a26")
    ax.bar(x + width / 2, [item.rfdetr_recall for item in COMPONENTS], width, label="RF-DETR Recall", color="#2b6cb0")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title("Recall comparison on official test split")
    ax.set_xticks(x, labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "downstream_recall_comparison.png", dpi=180)
    plt.close(fig)


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    output_dir = Path("docs/report_assets_20260609_downstream")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_metric_charts(output_dir)

    import rfdetr

    case_rows = []
    for component in COMPONENTS:
        print(f"processing {component.key}")
        yolo = attempt_load(str(component.yolo_weights), device="cpu")
        rf_model = rfdetr.from_checkpoint(str(component.rfdetr_checkpoint))
        if getattr(rf_model, "model", None) is not None and hasattr(rf_model.model, "device"):
            rf_model.model.device = torch.device("cpu")
        image_dir = component.dataset_dir / "test/images"
        label_dir = component.dataset_dir / "test/labels"
        image_paths = sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTS)
        selected = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            targets = read_targets(label_dir / f"{image_path.stem}.txt", width, height)
            yolo_preds = run_yolo(yolo, image)
            rfdetr_preds = run_rfdetr(rf_model, image, component.thresholds)
            yolo_matches = match_targets(targets, yolo_preds)
            rfdetr_matches = match_targets(targets, rfdetr_preds)
            rescued = [idx for idx in rfdetr_matches if idx not in yolo_matches]
            if rescued:
                selected.append((len(rescued), image_path, targets, yolo_preds, rfdetr_preds, rescued, rfdetr_matches))
        selected.sort(key=lambda item: (-item[0], item[1].name))
        for case_idx, item in enumerate(selected[:2], start=1):
            _, image_path, targets, yolo_preds, rfdetr_preds, rescued, rfdetr_matches = item
            case_name = f"{component.key}_case_{case_idx}_{image_path.stem}.jpg"
            save_case_image(output_dir / case_name, Image.open(image_path).convert("RGB"), targets, yolo_preds, rfdetr_preds)
            grades = sorted({CLASS_NAMES[targets[idx].cls] for idx in rescued})
            confs = [rfdetr_matches[idx][0].conf for idx in rescued]
            case_rows.append({
                "component": component.key,
                "case": case_idx,
                "image": image_path.name,
                "rescued_grades": "/".join(grades),
                "rfdetr_max_conf": f"{max(confs):.3f}",
                "asset": case_name,
            })

    with (output_dir / "downstream_yolo_missed_rfdetr_detected_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "case", "image", "rescued_grades", "rfdetr_max_conf", "asset"])
        writer.writeheader()
        writer.writerows(case_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
