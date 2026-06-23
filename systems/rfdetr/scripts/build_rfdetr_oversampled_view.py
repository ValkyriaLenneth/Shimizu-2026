#!/usr/bin/env python3
"""Build an RF-DETR YOLO dataset view with train-only class oversampling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-classes", required=True, help="comma-separated YOLO class ids")
    parser.add_argument("--repeat", type=int, default=2, help="total appearances for matched train samples")
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def read_classes(label_path: Path) -> set[int]:
    text = label_path.read_text(encoding="utf-8").strip()
    classes: set[int] = set()
    for line in text.splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))
    return classes


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTS)


def copy_data_yaml(source_dir: Path, output_dir: Path) -> None:
    data = yaml.safe_load((source_dir / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(output_dir.resolve())
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def copy_split(source_dir: Path, output_dir: Path, split: str, mode: str) -> dict[str, int | dict[str, int]]:
    copied = 0
    boxes: Counter[int] = Counter()
    for image_path in image_paths(source_dir / split / "images"):
        label_path = source_dir / split / "labels" / f"{image_path.stem}.txt"
        link_file(image_path, output_dir / split / "images" / image_path.name, mode)
        link_file(label_path, output_dir / split / "labels" / label_path.name, mode)
        copied += 1
        for cls in read_classes(label_path):
            boxes[cls] += 1
    return {"images": copied, "images_with_class": {str(cls): boxes.get(cls, 0) for cls in range(3)}}


def build_train(source_dir: Path, output_dir: Path, target_classes: set[int], repeat: int, mode: str) -> dict[str, object]:
    copied = 0
    repeated = 0
    class_image_counts: Counter[int] = Counter()
    class_box_counts: Counter[int] = Counter()

    for image_path in image_paths(source_dir / "train" / "images"):
        label_path = source_dir / "train" / "labels" / f"{image_path.stem}.txt"
        classes = read_classes(label_path)
        for cls in classes:
            class_image_counts[cls] += 1
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_box_counts[int(line.split()[0])] += 1

        appearances = repeat if classes & target_classes else 1
        for idx in range(appearances):
            suffix = "" if idx == 0 else f"__os{idx:02d}"
            out_stem = f"{image_path.stem}{suffix}"
            out_image = output_dir / "train" / "images" / f"{out_stem}{image_path.suffix}"
            out_label = output_dir / "train" / "labels" / f"{out_stem}.txt"
            link_file(image_path, out_image, mode)
            link_file(label_path, out_label, mode)
            copied += 1
            if idx > 0:
                repeated += 1

    return {
        "images": copied,
        "extra_repeated_images": repeated,
        "source_images_with_class": {str(cls): class_image_counts.get(cls, 0) for cls in range(3)},
        "source_boxes": {str(cls): class_box_counts.get(cls, 0) for cls in range(3)},
    }


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    target_classes = {int(item) for item in args.target_classes.split(",") if item.strip()}
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)

    summary = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "target_classes": sorted(target_classes),
        "repeat": args.repeat,
        "splits": {
            "train": build_train(source_dir, output_dir, target_classes, args.repeat, args.link_mode),
            "valid": copy_split(source_dir, output_dir, "valid", args.link_mode),
            "test": copy_split(source_dir, output_dir, "test", args.link_mode),
        },
    }
    copy_data_yaml(source_dir, output_dir)
    (output_dir / "oversample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
