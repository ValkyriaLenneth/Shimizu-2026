#!/usr/bin/env python3
"""Evaluate RF-DETR predictions routed to different checkpoints per class."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


@dataclass
class Target:
    cls: int
    xyxy: tuple[float, float, float, float]
    matched: bool = False


@dataclass
class Prediction:
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]
    matched: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path")
    parser.add_argument("--class-source", required=True, help="comma-separated class_id=name items")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="0.001,0.003,0.005,0.01,0.02,0.05,0.1,0.25,0.5")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_mapping(items: list[str], sep: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, value = item.split(sep, 1)
        out[key.strip()] = value.strip()
    return out


def yolo_to_xyxy(line: str, width: int, height: int) -> Target:
    parts = line.split()
    cls = int(parts[0])
    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return Target(cls=cls, xyxy=(x1, y1, x2, y2))


def read_targets(label_path: Path, width: int, height: int) -> list[Target]:
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


def detections_to_predictions(detections, threshold: float, num_classes: int) -> list[Prediction]:
    xyxy = np.asarray(detections.xyxy)
    conf = np.asarray(detections.confidence)
    class_id = np.asarray(detections.class_id)
    preds: list[Prediction] = []
    for box, score, cls in zip(xyxy, conf, class_id, strict=False):
        cls_i = int(cls)
        if float(score) < threshold or cls_i < 0 or cls_i >= num_classes:
            continue
        preds.append(Prediction(cls=cls_i, conf=float(score), xyxy=tuple(float(x) for x in box)))
    preds.sort(key=lambda p: p.conf, reverse=True)
    return preds


def match_counts(targets_in: list[Target], preds_in: list[Prediction], iou_threshold: float) -> dict[int, dict[str, int]]:
    targets = [Target(t.cls, t.xyxy) for t in targets_in]
    preds = [Prediction(p.cls, p.conf, p.xyxy) for p in preds_in]
    counts = {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(3)}
    for target in targets:
        counts[target.cls]["gt"] += 1
    for pred in preds:
        counts[pred.cls]["pred"] += 1
        candidates = [
            (box_iou(pred.xyxy, target.xyxy), idx)
            for idx, target in enumerate(targets)
            if not target.matched and target.cls == pred.cls
        ]
        if candidates:
            best_iou, best_idx = max(candidates, key=lambda item: item[0])
            if best_iou >= iou_threshold:
                targets[best_idx].matched = True
                pred.matched = True
                counts[pred.cls]["tp"] += 1
                continue
        counts[pred.cls]["fp"] += 1
    for target in targets:
        if not target.matched:
            counts[target.cls]["fn"] += 1
    return counts


def merge_counts(total: dict[int, dict[str, int]], item: dict[int, dict[str, int]]) -> None:
    for cls, values in item.items():
        for key, value in values.items():
            total[cls][key] += value


def metric(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> int:
    args = parse_args()
    import torch
    import rfdetr

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    checkpoint_paths = parse_mapping(args.checkpoint, "=")
    class_source = {int(k): v for k, v in parse_mapping(args.class_source.split(","), "=").items()}
    missing = [name for name in class_source.values() if name not in checkpoint_paths]
    if missing:
        raise ValueError(f"class-source references unknown checkpoints: {sorted(set(missing))}")

    models = {name: rfdetr.from_checkpoint(path) for name, path in checkpoint_paths.items()}
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    min_threshold = min(thresholds)

    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / args.split / "images"
    label_dir = dataset_dir / args.split / "labels"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTS)

    per_threshold = {
        threshold: {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(3)}
        for threshold in thresholds
    }

    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        targets = read_targets(label_dir / f"{image_path.stem}.txt", width, height)
        low_preds_by_source = {
            name: detections_to_predictions(model.predict(image, threshold=min_threshold, include_source_image=False), min_threshold, args.num_classes)
            for name, model in models.items()
        }
        for threshold in thresholds:
            routed = []
            for cls, source in class_source.items():
                routed.extend(p for p in low_preds_by_source[source] if p.cls == cls and p.conf >= threshold)
            routed.sort(key=lambda p: p.conf, reverse=True)
            merge_counts(per_threshold[threshold], match_counts(targets, routed, args.iou_threshold))
        if idx % 25 == 0 or idx == len(image_paths):
            print(f"evaluated {idx}/{len(image_paths)} images")

    rows = []
    for threshold, counts in per_threshold.items():
        overall = {"tp": 0, "fp": 0, "fn": 0}
        row = {"threshold": threshold, "images": len(image_paths)}
        for cls in range(3):
            values = counts[cls]
            precision, recall, f1 = metric(values["tp"], values["fp"], values["fn"])
            row[f"class_{cls}_tp"] = values["tp"]
            row[f"class_{cls}_fp"] = values["fp"]
            row[f"class_{cls}_fn"] = values["fn"]
            row[f"class_{cls}_precision"] = precision
            row[f"class_{cls}_recall"] = recall
            row[f"class_{cls}_f1"] = f1
            overall["tp"] += values["tp"]
            overall["fp"] += values["fp"]
            overall["fn"] += values["fn"]
        precision, recall, f1 = metric(overall["tp"], overall["fp"], overall["fn"])
        row["precision"] = precision
        row["recall"] = recall
        row["f1"] = f1
        rows.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
