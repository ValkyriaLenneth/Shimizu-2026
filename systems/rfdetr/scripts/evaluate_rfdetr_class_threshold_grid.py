#!/usr/bin/env python3
"""Evaluate RF-DETR checkpoints with class-specific confidence thresholds."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path

import torch
import rfdetr
from PIL import Image

from evaluate_rfdetr_threshold_sweep import (
    IMAGE_EXTS,
    detections_to_predictions,
    match_counts,
    merge_counts,
    metric,
    read_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold-grid", default="0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    thresholds = [float(item) for item in args.threshold_grid.split(",") if item.strip()]
    min_threshold = min(thresholds)

    model = rfdetr.from_checkpoint(args.checkpoint)
    model_ctx = getattr(model, "model", None)
    if model_ctx is not None and hasattr(model_ctx, "device"):
        model_ctx.device = torch.device(args.device)

    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / args.split / "images"
    label_dir = dataset_dir / args.split / "labels"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTS)

    cached = []
    for idx, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        targets = read_targets(label_dir / f"{image_path.stem}.txt", width, height)
        detections = model.predict(image, threshold=min_threshold, include_source_image=False)
        preds, ignored = detections_to_predictions(detections, min_threshold, args.num_classes)
        cached.append((targets, preds, ignored))
        if idx % 25 == 0 or idx == len(image_paths):
            print(f"cached predictions {idx}/{len(image_paths)} images")

    rows = []
    for class_thresholds in itertools.product(thresholds, repeat=args.num_classes):
        total = {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(args.num_classes)}
        ignored_predictions = 0
        for targets, preds, ignored in cached:
            selected = [pred for pred in preds if pred.conf >= class_thresholds[pred.cls]]
            ignored_predictions += ignored
            # match_counts in evaluate_rfdetr_threshold_sweep gained a required
            # num_classes argument; this call site was never updated, so the
            # per-class threshold grid raised TypeError before producing any row.
            merge_counts(total, match_counts(targets, selected, args.iou_threshold, args.num_classes))

        overall = {"tp": 0, "fp": 0, "fn": 0}
        row = {
            "thresholds": ",".join(f"{value:.6g}" for value in class_thresholds),
            "threshold_class_0": class_thresholds[0],
            "threshold_class_1": class_thresholds[1],
            "threshold_class_2": class_thresholds[2],
            "images": len(image_paths),
            "ignored_predictions": ignored_predictions,
        }
        for cls in range(args.num_classes):
            values = total[cls]
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
