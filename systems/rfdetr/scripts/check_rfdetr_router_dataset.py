#!/usr/bin/env python3
"""Preflight checks for the RF-DETR router dataset view."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/rfdetr_router_base_aug_v2")
    parser.add_argument("--write-summary", default="")
    return parser.parse_args()


def load_data_yaml(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    names = data.get("names")
    if not isinstance(names, (dict, list)):
        raise ValueError("data.yaml must contain YOLO class names as a dict or list")
    return data


def count_split(dataset_dir: Path, split: str) -> dict[str, Any]:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    class_counts: Counter[str] = Counter()
    malformed = []
    empty_labels = 0
    for label_path in labels:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue
        for index, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) < 5:
                malformed.append(f"{label_path}:{index}")
                continue
            class_counts[parts[0]] += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(image_stems - label_stems),
        "orphan_labels": len(label_stems - image_stems),
        "empty_labels": empty_labels,
        "malformed_lines": malformed[:20],
        "class_counts": dict(sorted(class_counts.items())),
    }


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    data_yaml = load_data_yaml(dataset_dir)
    summary = {
        "dataset_dir": str(dataset_dir),
        "data_yaml": data_yaml,
        "splits": {split: count_split(dataset_dir, split) for split in SPLITS},
    }

    failed = False
    for split, stats in summary["splits"].items():
        if stats["images"] == 0:
            failed = True
            print(f"[ERROR] {split}: no images")
        if stats["missing_labels"] or stats["orphan_labels"] or stats["malformed_lines"]:
            failed = True
            print(f"[ERROR] {split}: {stats}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write_summary:
        out = Path(args.write_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
