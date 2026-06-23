#!/usr/bin/env python3
"""Build a router dataset with RC-column oversampling in the train split."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "coarse_router_3class_cleaned"
OUT = ROOT / "datasets" / "coarse_router_3class_cleaned_rc_column_oversample"
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
RC_COLUMN_CLASS = "2"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {"source": str(SRC), "target": str(OUT), "splits": {}}
    for split in ["train", "val", "test"]:
        labels = sorted((SRC / "labels" / split).glob("*.txt"))
        class_counts = Counter()
        for label in labels:
            image = find_image(SRC / "images" / split, label.stem)
            link_pair(image, OUT / "images" / split / image.name)
            link_pair(label, OUT / "labels" / split / label.name)
            class_counts.update(label_classes(label))

        added = 0
        if split == "train":
            target_count = max(class_counts.get("0", 0), class_counts.get("2", 0))
            rc_labels = [p for p in labels if RC_COLUMN_CLASS in label_classes(p)]
            current = class_counts.get(RC_COLUMN_CLASS, 0)
            index = 0
            while current < target_count and rc_labels:
                label = rc_labels[index % len(rc_labels)]
                image = find_image(SRC / "images" / split, label.stem)
                suffix = f"__rc_os{index:04d}"
                image_out = OUT / "images" / split / f"{image.stem}{suffix}{image.suffix.lower()}"
                label_out = OUT / "labels" / split / f"{label.stem}{suffix}.txt"
                link_pair(image, image_out)
                link_pair(label, label_out)
                counts = label_classes(label)
                class_counts.update(counts)
                current += counts.get(RC_COLUMN_CLASS, 0)
                added += 1
                index += 1

        summary["splits"][split] = {
            "label_files": len(list((OUT / "labels" / split).glob("*.txt"))),
            "image_files": len([p for p in (OUT / "images" / split).iterdir() if p.is_file()]),
            "class_counts": dict(class_counts),
            "oversampled_files_added": added,
        }

    write_data_yaml()
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def link_pair(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_data_yaml() -> None:
    text = f"""path: {OUT}
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: 天井
  1: 壁类
  2: RC柱
"""
    (OUT / "data.yaml").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
