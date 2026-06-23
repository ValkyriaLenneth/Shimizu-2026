#!/usr/bin/env python3
"""Build an RF-DETR YOLO dataset view by repeating train hard-case images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--hardcase-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-classes", required=True, help="comma-separated class ids")
    parser.add_argument("--include-fp", action="store_true")
    parser.add_argument("--include-fn", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-per-image", type=int, default=1)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def copy_split(source_dir: Path, output_dir: Path, split: str, mode: str) -> int:
    count = 0
    for sub in ["images", "labels"]:
        src_root = source_dir / split / sub
        dst_root = output_dir / split / sub
        dst_root.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_root.iterdir()):
            if src.is_file():
                link_or_copy(src, dst_root / src.name, mode)
                count += 1
    return count


def image_path_for(source_dir: Path, split: str, image_name: str) -> Path:
    image_dir = source_dir / split / "images"
    direct = image_dir / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    matches = [path for path in image_dir.iterdir() if path.stem == stem and path.suffix in IMAGE_EXTS]
    if not matches:
        raise FileNotFoundError(f"cannot find image for {image_name}")
    return matches[0]


def select_images(csv_path: Path, target_classes: set[str], include_fp: bool, include_fn: bool) -> Counter[str]:
    selected: Counter[str] = Counter()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if include_fp and row.get("case_type") == "false_positive" and row.get("pred_class") in target_classes:
                selected[row["image"]] += 1
            if include_fn and row.get("case_type") == "false_negative" and row.get("gt_class") in target_classes:
                selected[row["image"]] += 1
    return selected


def duplicate_train(source_dir: Path, output_dir: Path, selected: Counter[str], args: argparse.Namespace) -> dict[str, int]:
    copied = copy_split(source_dir, output_dir, "train", args.link_mode)
    out_images = output_dir / "train" / "images"
    out_labels = output_dir / "train" / "labels"
    duplicated = 0
    selected_images = 0
    for image_name, hits in selected.items():
        copies = min(hits, args.max_per_image) * args.repeat
        if copies <= 0:
            continue
        image_path = image_path_for(source_dir, "train", image_name)
        label_path = source_dir / "train" / "labels" / f"{image_path.stem}.txt"
        selected_images += 1
        for idx in range(copies):
            stem = f"{image_path.stem}__hard_{idx:02d}"
            link_or_copy(image_path, out_images / f"{stem}{image_path.suffix.lower()}", args.link_mode)
            dst_label = out_labels / f"{stem}.txt"
            if label_path.exists():
                link_or_copy(label_path, dst_label, args.link_mode)
            else:
                dst_label.write_text("", encoding="utf-8")
            duplicated += 1
    return {"copied_files": copied, "selected_images": selected_images, "duplicate_images": duplicated}


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_classes = {item.strip() for item in args.target_classes.split(",") if item.strip()}
    selected = select_images(Path(args.hardcase_csv), target_classes, args.include_fp, args.include_fn)
    summary = {
        "source_dir": str(source_dir),
        "hardcase_csv": args.hardcase_csv,
        "target_classes": sorted(target_classes),
        "include_fp": args.include_fp,
        "include_fn": args.include_fn,
        "repeat": args.repeat,
        "max_per_image": args.max_per_image,
        "train": duplicate_train(source_dir, output_dir, selected, args),
        "valid_files": copy_split(source_dir, output_dir, "valid", args.link_mode),
        "test_files": copy_split(source_dir, output_dir, "test", args.link_mode),
    }
    for name in ["data.yaml", "data_split.json"]:
        src = source_dir / name
        if src.exists():
            link_or_copy(src, output_dir / name, args.link_mode)
    (output_dir / "hardcase_view_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
