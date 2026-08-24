#!/usr/bin/env python3
"""Evaluate a router checkpoint with fixed per-class operating thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import match_counts, merge_counts, metric, read_targets


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", required=True, help="comma-separated class thresholds")
    parser.add_argument("--class-names", default="天井,壁类,RC柱,ブレース,柱脚")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-dir", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def predictions(detections, thresholds: list[float]):
    from evaluate_rfdetr_threshold_sweep import Prediction

    rows = []
    for box, score, cls in zip(detections.xyxy, detections.confidence, detections.class_id, strict=False):
        cls = int(cls)
        if 0 <= cls < len(thresholds) and float(score) >= thresholds[cls]:
            rows.append(Prediction(cls=cls, conf=float(score), xyxy=tuple(float(v) for v in box)))
    rows.sort(key=lambda row: row.conf, reverse=True)
    return rows


def main() -> int:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    names = args.class_names.split(",")
    if len(thresholds) != len(names):
        raise ValueError("threshold and class-name counts differ")

    model = from_checkpoint_matched(args.checkpoint)
    image_dir = Path(args.dataset_dir) / args.split / "images"
    label_dir = Path(args.dataset_dir) / args.split / "labels"
    paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")
    )
    counts = {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(len(names))}
    for index, path in enumerate(paths, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        targets = read_targets(label_dir / f"{path.stem}.txt", *image.size)
        det = model.predict(image, threshold=min(thresholds), include_source_image=False)
        merge_counts(counts, match_counts(targets, predictions(det, thresholds), args.iou_threshold, len(names)))
        if index % 50 == 0:
            print(f"positive test: {index}/{len(paths)}", file=sys.stderr)

    overall = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {}
    for cls, name in enumerate(names):
        values = counts[cls]
        precision, recall, f1 = metric(values["tp"], values["fp"], values["fn"])
        per_class[name] = {**values, "threshold": thresholds[cls], "precision": precision, "recall": recall, "f1": f1}
        for key in overall:
            overall[key] += values[key]
    overall["precision"], overall["recall"], overall["f1"] = metric(
        overall["tp"], overall["fp"], overall["fn"]
    )

    negative_paths = sorted(
        path for directory in args.negative_dir for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")
    )
    fired = 0
    boxes = 0
    fired_by_class = {name: 0 for name in names}
    boxes_by_class = {name: 0 for name in names}
    for index, path in enumerate(negative_paths, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        det = model.predict(image, threshold=min(thresholds), include_source_image=False)
        kept = predictions(det, thresholds)
        if kept:
            fired += 1
        boxes += len(kept)
        for cls in {row.cls for row in kept}:
            fired_by_class[names[cls]] += 1
        for row in kept:
            boxes_by_class[names[row.cls]] += 1
        if index % 50 == 0:
            print(f"negative test: {index}/{len(negative_paths)}", file=sys.stderr)

    negative = {
        "images": len(negative_paths),
        "images_fired": fired,
        "image_false_alarm_rate": fired / len(negative_paths) if negative_paths else None,
        "boxes": boxes,
        "boxes_per_image": boxes / len(negative_paths) if negative_paths else None,
        "images_fired_by_class": fired_by_class,
        "boxes_by_class": boxes_by_class,
    }
    result = {
        "checkpoint": args.checkpoint,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "thresholds": dict(zip(names, thresholds, strict=True)),
        "positive_test_images": len(paths),
        "overall": overall,
        "per_class": per_class,
        "negative_test": negative,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
