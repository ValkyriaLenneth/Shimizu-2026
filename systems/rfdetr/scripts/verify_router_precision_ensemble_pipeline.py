#!/usr/bin/env python3
"""Verify the production Router ensemble against a labelled RF-DETR dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from evaluate_rfdetr_threshold_sweep import (  # noqa: E402
    IMAGE_EXTS,
    Prediction,
    match_counts,
    merge_counts,
    metric,
    read_targets,
)
from rfdetr_prod_pipeline.pipeline.run_full_pipeline import build_router  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.pipeline_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    classes = {int(key): str(value) for key, value in config["classes"]["router"].items()}
    router = build_router(config["pipeline"], config, config_path.parent, args.device)

    dataset_dir = Path(args.dataset_dir).resolve()
    image_dir = dataset_dir / args.split / "images"
    label_dir = dataset_dir / args.split / "labels"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")
    )
    if not image_paths:
        raise RuntimeError(f"no images found under {image_dir}")

    warmup = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if warmup is None:
        raise RuntimeError(f"unreadable warmup image: {image_paths[0]}")
    router.predict(warmup)

    total = {
        class_id: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for class_id in classes
    }
    latencies_ms = []
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable image: {image_path}")
        started = time.perf_counter()
        result = router.predict(image)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        predictions = [
            Prediction(
                cls=int(row["class_id"]),
                conf=float(row["confidence"]),
                xyxy=tuple(float(value) for value in row["bbox_xyxy"]),
            )
            for row in result["detections"]
        ]
        height, width = image.shape[:2]
        targets = read_targets(label_dir / f"{image_path.stem}.txt", width, height)
        merge_counts(total, match_counts(targets, predictions, args.match_iou, len(classes)))
        if index % 50 == 0 or index == len(image_paths):
            print(f"verified: {index}/{len(image_paths)}", flush=True)

    per_class = {}
    for class_id, name in classes.items():
        counts = total[class_id]
        precision, recall, f1 = metric(counts["tp"], counts["fp"], counts["fn"])
        per_class[name] = {**counts, "precision": precision, "recall": recall, "f1": f1}
    tp = sum(row["tp"] for row in per_class.values())
    fp = sum(row["fp"] for row in per_class.values())
    fn = sum(row["fn"] for row in per_class.values())
    precision, recall, f1 = metric(tp, fp, fn)
    summary = {
        "pipeline_config": str(config_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "images": len(image_paths),
        "match_iou": args.match_iou,
        "per_class": per_class,
        "overall": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "max": float(np.max(latencies_ms)),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
