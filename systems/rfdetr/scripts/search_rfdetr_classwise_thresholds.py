#!/usr/bin/env python3
"""Search class-specific confidence thresholds for an RF-DETR checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
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
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--class-thresholds", action="append", required=True, help="class_id=v1,v2,...")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_thresholds(items: list[str], num_classes: int) -> list[list[float]]:
    grids: list[list[float] | None] = [None] * num_classes
    for item in items:
        key, values = item.split("=", 1)
        cls = int(key)
        grids[cls] = [float(value) for value in values.split(",") if value.strip()]
    missing = [str(idx) for idx, values in enumerate(grids) if values is None]
    if missing:
        raise ValueError("missing class threshold grids: " + ", ".join(missing))
    return [values for values in grids if values is not None]


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    grids = parse_thresholds(args.class_thresholds, args.num_classes)
    min_threshold = min(min(values) for values in grids)

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
        targets = read_targets(label_dir / f"{image_path.stem}.txt", *image.size)
        detections = model.predict(image, threshold=min_threshold, include_source_image=False)
        preds, _ = detections_to_predictions(detections, min_threshold, args.num_classes)
        cached.append((targets, preds))
        if idx % 50 == 0 or idx == len(image_paths):
            print(f"cached predictions {idx}/{len(image_paths)} images", flush=True)

    rows = []
    for thresholds in product(*grids):
        total = {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(args.num_classes)}
        for targets, preds in cached:
            selected = [pred for pred in preds if pred.conf >= thresholds[pred.cls]]
            merge_counts(total, match_counts(targets, selected, args.iou_threshold, args.num_classes))

        overall = {"tp": 0, "fp": 0, "fn": 0}
        row = {f"threshold_class_{idx}": value for idx, value in enumerate(thresholds)}
        row["thresholds"] = ",".join(f"{value:.6g}" for value in thresholds)
        row["images"] = len(image_paths)
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
        row["min_class_recall"] = min(row[f"class_{cls}_recall"] for cls in range(args.num_classes))
        rows.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ok = [row for row in rows if row["precision"] >= args.target_precision]
    print(f"wrote {output} rows={len(rows)} target_rows={len(ok)}", flush=True)
    for label, key in [("best_recall", "recall"), ("best_min_recall", "min_class_recall")]:
        if not ok:
            continue
        best = max(ok, key=lambda row: (row[key], row["recall"], row["f1"]))
        print(
            label,
            best["thresholds"],
            "precision",
            best["precision"],
            "recall",
            best["recall"],
            "min_class_recall",
            best["min_class_recall"],
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
