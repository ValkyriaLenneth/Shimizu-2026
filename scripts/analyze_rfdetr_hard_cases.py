#!/usr/bin/env python3
"""Export RF-DETR false-negative / false-positive hard cases for a YOLO split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


@dataclass
class BoxItem:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0
    matched: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--low-threshold", type=float, default=0.01)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def yolo_to_xyxy(line: str, width: int, height: int) -> BoxItem:
    parts = line.split()
    cls = int(parts[0])
    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return BoxItem(cls=cls, xyxy=(x1, y1, x2, y2))


def read_targets(label_path: Path, width: int, height: int) -> list[BoxItem]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [yolo_to_xyxy(line, width, height) for line in text.splitlines() if line.strip()]


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detections_to_items(detections, threshold: float, num_classes: int) -> list[BoxItem]:
    xyxy = np.asarray(detections.xyxy)
    conf = np.asarray(detections.confidence)
    class_id = np.asarray(detections.class_id)
    items: list[BoxItem] = []
    for box, score, cls in zip(xyxy, conf, class_id, strict=False):
        cls_i = int(cls)
        if float(score) < threshold or cls_i < 0 or cls_i >= num_classes:
            continue
        items.append(BoxItem(cls=cls_i, conf=float(score), xyxy=tuple(float(x) for x in box)))
    items.sort(key=lambda item: item.conf, reverse=True)
    return items


def best_overlap(target: BoxItem, preds: list[BoxItem], same_class: bool | None) -> tuple[BoxItem | None, float]:
    candidates = []
    for pred in preds:
        if same_class is True and pred.cls != target.cls:
            continue
        if same_class is False and pred.cls == target.cls:
            continue
        candidates.append((box_iou(target.xyxy, pred.xyxy), pred))
    if not candidates:
        return None, 0.0
    iou, pred = max(candidates, key=lambda item: item[0])
    return pred, iou


def match(targets: list[BoxItem], preds: list[BoxItem], iou_threshold: float) -> None:
    for pred in preds:
        candidates = [
            (box_iou(pred.xyxy, target.xyxy), idx)
            for idx, target in enumerate(targets)
            if not target.matched and target.cls == pred.cls
        ]
        if not candidates:
            continue
        best_iou, best_idx = max(candidates, key=lambda item: item[0])
        if best_iou >= iou_threshold:
            pred.matched = True
            targets[best_idx].matched = True


def fmt_box(box: tuple[float, float, float, float]) -> str:
    return ",".join(f"{x:.1f}" for x in box)


def main() -> int:
    args = parse_args()
    import torch
    import rfdetr

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    model = rfdetr.from_checkpoint(args.checkpoint)

    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / args.split / "images"
    label_dir = dataset_dir / args.split / "labels"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTS)

    rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        targets = read_targets(label_dir / f"{image_path.stem}.txt", width, height)
        detections = model.predict(image, threshold=args.low_threshold, include_source_image=False)
        low_preds = detections_to_items(detections, args.low_threshold, args.num_classes)
        preds = [BoxItem(p.cls, p.xyxy, p.conf) for p in low_preds if p.conf >= args.threshold]
        match(targets, preds, args.iou_threshold)

        for target in targets:
            if target.matched:
                continue
            low_same, low_same_iou = best_overlap(target, low_preds, same_class=True)
            wrong, wrong_iou = best_overlap(target, preds, same_class=False)
            rows.append(
                {
                    "image": image_path.name,
                    "case_type": "false_negative",
                    "gt_class": target.cls,
                    "pred_class": "" if wrong is None else wrong.cls,
                    "pred_conf": "" if wrong is None else f"{wrong.conf:.6f}",
                    "iou": f"{wrong_iou:.6f}",
                    "gt_xyxy": fmt_box(target.xyxy),
                    "pred_xyxy": "" if wrong is None else fmt_box(wrong.xyxy),
                    "low_same_class_conf": "" if low_same is None else f"{low_same.conf:.6f}",
                    "low_same_class_iou": f"{low_same_iou:.6f}",
                    "reason": (
                        "matched only below threshold"
                        if low_same is not None and low_same_iou >= args.iou_threshold
                        else "wrong-class overlap"
                        if wrong is not None and wrong_iou >= args.iou_threshold
                        else "no same-class IoU match"
                    ),
                }
            )

        for pred in preds:
            if pred.matched:
                continue
            best_gt, best_gt_iou = best_overlap(pred, targets, same_class=None)
            rows.append(
                {
                    "image": image_path.name,
                    "case_type": "false_positive",
                    "gt_class": "" if best_gt is None else best_gt.cls,
                    "pred_class": pred.cls,
                    "pred_conf": f"{pred.conf:.6f}",
                    "iou": f"{best_gt_iou:.6f}",
                    "gt_xyxy": "" if best_gt is None else fmt_box(best_gt.xyxy),
                    "pred_xyxy": fmt_box(pred.xyxy),
                    "low_same_class_conf": "",
                    "low_same_class_iou": "",
                    "reason": "background" if best_gt is None else "duplicate/wrong localization/class",
                }
            )
        if idx % 25 == 0 or idx == len(image_paths):
            print(f"analyzed {idx}/{len(image_paths)} images")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "case_type",
        "gt_class",
        "pred_class",
        "pred_conf",
        "iou",
        "gt_xyxy",
        "pred_xyxy",
        "low_same_class_conf",
        "low_same_class_iou",
        "reason",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
