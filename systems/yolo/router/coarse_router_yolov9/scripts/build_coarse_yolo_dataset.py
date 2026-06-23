#!/usr/bin/env python3
"""Build a YOLO dataset from Gemini coarse building-element annotations."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASSES = ["天井", "内壁", "RC壁", "RC柱"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/gemini_wall_label_fixed_3_1_pro/results.jsonl")
    parser.add_argument("--output-dir", default="coarse_router_yolov9/datasets/coarse_cross_fixed")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-box-area", type=float, default=0.0005)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    return parser.parse_args()


def get_parsed(row: dict) -> dict:
    return (row.get("response") or {}).get("parsed") or {}


def source_group(row: dict) -> str:
    label = row.get("expected_label")
    if label in CLASS_TO_ID:
        return label
    rel = row.get("image_rel_path") or row.get("image_path") or ""
    for label in CLASSES:
        if label in rel:
            return label
    return "unknown"


def normalize_box(box: list[float]) -> tuple[float, float, float, float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = [float(x) for x in box]
    ymin, ymax = sorted((max(0.0, min(1000.0, ymin)), max(0.0, min(1000.0, ymax))))
    xmin, xmax = sorted((max(0.0, min(1000.0, xmin)), max(0.0, min(1000.0, xmax))))
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return None
    return ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0


def yolo_line(label: str, box: tuple[float, float, float, float], min_box_area: float) -> str | None:
    ymin, xmin, ymax, xmax = box
    width = xmax - xmin
    height = ymax - ymin
    if width * height < min_box_area:
        return None
    x_center = xmin + width / 2
    y_center = ymin + height / 2
    values = [CLASS_TO_ID[label], x_center, y_center, width, height]
    return f"{values[0]} " + " ".join(f"{v:.6f}" for v in values[1:])


def link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def split_rows(rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[source_group(row)].append(row)

    split = {"train": [], "val": [], "test": []}
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        split["train"].extend(group_rows[:n_train])
        split["val"].extend(group_rows[n_train:n_train + n_val])
        split["test"].extend(group_rows[n_train + n_val:])

    for rows_in_split in split.values():
        rng.shuffle(rows_in_split)
    return split


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    input_path = root / args.input
    output_dir = root / args.output_dir

    rows_by_key: dict[str, dict] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok", False):
                continue
            key = row.get("image_rel_path") or row.get("image_path")
            if not key or key in rows_by_key:
                continue
            rows_by_key[key] = row

    rows = list(rows_by_key.values())
    split = split_rows(rows, args.train_ratio, args.val_ratio, args.seed)

    stats = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "classes": CLASSES,
        "source_images": len(rows),
        "splits": {},
        "discarded_boxes": Counter(),
        "box_counts": Counter(),
        "image_source_counts": Counter(),
    }

    for subset, subset_rows in split.items():
        subset_image_dir = output_dir / "images" / subset
        subset_label_dir = output_dir / "labels" / subset
        subset_image_dir.mkdir(parents=True, exist_ok=True)
        subset_label_dir.mkdir(parents=True, exist_ok=True)
        subset_stats = {
            "images": 0,
            "images_with_labels": 0,
            "boxes": 0,
            "source_counts": Counter(),
            "label_counts": Counter(),
        }

        for idx, row in enumerate(subset_rows):
            rel = row.get("image_rel_path") or row.get("image_path")
            src = root / "data" / "unzip" / rel if not str(rel).startswith("data/") else root / rel
            if not src.exists():
                stats["discarded_boxes"]["missing_image"] += 1
                continue

            suffix = src.suffix.lower()
            safe_stem = f"{source_group(row)}_{src.stem}_{idx:05d}"
            dst_img = subset_image_dir / f"{safe_stem}{suffix}"
            dst_lbl = subset_label_dir / f"{safe_stem}.txt"

            lines: list[str] = []
            for element in get_parsed(row).get("elements") or []:
                label = element.get("label")
                if label not in CLASS_TO_ID:
                    stats["discarded_boxes"]["unknown_label"] += 1
                    continue
                box = normalize_box(element.get("bbox_2d"))
                if box is None:
                    stats["discarded_boxes"]["invalid_box"] += 1
                    continue
                line = yolo_line(label, box, args.min_box_area)
                if line is None:
                    stats["discarded_boxes"]["small_box"] += 1
                    continue
                lines.append(line)
                subset_stats["label_counts"][label] += 1
                stats["box_counts"][label] += 1

            link_image(src, dst_img, args.link_mode)
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            subset_stats["images"] += 1
            subset_stats["boxes"] += len(lines)
            subset_stats["images_with_labels"] += bool(lines)
            subset_stats["source_counts"][source_group(row)] += 1
            stats["image_source_counts"][source_group(row)] += 1

            # Verify image readability once while building the dataset.
            try:
                with Image.open(dst_img) as im:
                    im.verify()
            except Exception:
                stats["discarded_boxes"]["unreadable_image_linked"] += 1

        stats["splits"][subset] = {
            key: dict(value) if isinstance(value, Counter) else value
            for key, value in subset_stats.items()
        }

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(output_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)),
        encoding="utf-8",
    )

    summary_path = output_dir / "summary.json"
    serializable = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }
    summary_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
