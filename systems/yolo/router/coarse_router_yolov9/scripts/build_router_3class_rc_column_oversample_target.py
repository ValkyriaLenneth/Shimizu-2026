#!/usr/bin/env python3
"""Build a router dataset with configurable RC-column train oversampling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
RC_COLUMN_CLASS = "2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "datasets" / "coarse_router_3class_cleaned"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-rc-boxes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.source).resolve()
    out = Path(args.output).resolve()
    if out.exists():
        shutil.rmtree(out)
    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {"source": str(src), "target": str(out), "target_rc_boxes": args.target_rc_boxes, "splits": {}}
    for split in ["train", "val", "test"]:
        labels = sorted((src / "labels" / split).glob("*.txt"))
        class_counts = Counter()
        for label in labels:
            image = find_image(src / "images" / split, label.stem)
            link_or_copy(image, out / "images" / split / image.name)
            link_or_copy(label, out / "labels" / split / label.name)
            class_counts.update(label_classes(label))

        added = 0
        if split == "train":
            rc_labels = [p for p in labels if RC_COLUMN_CLASS in label_classes(p)]
            current = class_counts.get(RC_COLUMN_CLASS, 0)
            index = 0
            while current < args.target_rc_boxes and rc_labels:
                label = rc_labels[index % len(rc_labels)]
                image = find_image(src / "images" / split, label.stem)
                suffix = f"__rc_os{index:04d}"
                link_or_copy(image, out / "images" / split / f"{image.stem}{suffix}{image.suffix.lower()}")
                link_or_copy(label, out / "labels" / split / f"{label.stem}{suffix}.txt")
                counts = label_classes(label)
                class_counts.update(counts)
                current += counts.get(RC_COLUMN_CLASS, 0)
                added += 1
                index += 1

        summary["splits"][split] = {
            "label_files": len(list((out / "labels" / split).glob("*.txt"))),
            "image_files": len([p for p in (out / "images" / split).iterdir() if p.is_file()]),
            "class_counts": dict(class_counts),
            "oversampled_files_added": added,
        }

    write_data_yaml(out)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def label_classes(path: Path) -> Counter[str]:
    counts = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[line.split()[0]] += 1
    return counts


def find_image(root: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        for candidate in [root / f"{stem}{suffix}", root / f"{stem}{suffix.upper()}"]:
            if candidate.exists():
                return candidate
    matches = [p for p in root.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES]
    if not matches:
        raise FileNotFoundError(f"missing image for {stem} under {root}")
    return matches[0]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_data_yaml(out: Path) -> None:
    text = f"""path: {out}
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: 天井
  1: 壁类
  2: RC柱
"""
    (out / "data.yaml").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
