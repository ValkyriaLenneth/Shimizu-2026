#!/usr/bin/env python3
"""Build an RF-DETR YOLO dataset view with train-only crop augmentation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-classes", required=True, help="comma-separated YOLO class ids")
    parser.add_argument("--crops-per-box", type=int, default=2)
    parser.add_argument("--context", type=float, default=3.0)
    parser.add_argument("--min-size", type=int, default=256)
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


def read_yolo(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append((int(parts[0]), *(float(x) for x in parts[1:5])))
    return rows


def yolo_to_xyxy(row: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, float, float, float, float]:
    cls, cx, cy, bw, bh = row
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return cls, x1, y1, x2, y2


def clip_box(box: tuple[int, float, float, float, float], crop: tuple[int, int, int, int]) -> tuple[int, float, float, float, float] | None:
    cls, x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop
    nx1 = max(x1, cx1) - cx1
    ny1 = max(y1, cy1) - cy1
    nx2 = min(x2, cx2) - cx1
    ny2 = min(y2, cy2) - cy1
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    if (nx2 - nx1) * (ny2 - ny1) < 16:
        return None
    return cls, nx1, ny1, nx2, ny2


def xyxy_to_yolo(box: tuple[int, float, float, float, float], width: int, height: int) -> str:
    cls, x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def crop_for_box(
    box: tuple[int, float, float, float, float],
    width: int,
    height: int,
    *,
    context: float,
    min_size: int,
    variant: int,
) -> tuple[int, int, int, int]:
    _, x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    size = max(min_size, int(max(bw, bh) * context))
    size = min(size, max(width, height))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    offsets = [(0.0, 0.0), (-0.18, -0.12), (0.18, 0.12), (-0.12, 0.18), (0.12, -0.18)]
    ox, oy = offsets[variant % len(offsets)]
    cx += ox * size
    cy += oy * size
    left = int(round(cx - size / 2))
    top = int(round(cy - size / 2))
    left = max(0, min(left, max(0, width - size)))
    top = max(0, min(top, max(0, height - size)))
    right = min(width, left + size)
    bottom = min(height, top + size)
    return left, top, right, bottom


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


def build_train(source_dir: Path, output_dir: Path, target_classes: set[int], args: argparse.Namespace) -> dict[str, int]:
    copied = copy_split(source_dir, output_dir, "train", args.link_mode)
    image_dir = source_dir / "train" / "images"
    label_dir = source_dir / "train" / "labels"
    out_image_dir = output_dir / "train" / "images"
    out_label_dir = output_dir / "train" / "labels"
    crops = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTS):
        label_path = label_dir / f"{image_path.stem}.txt"
        rows = read_yolo(label_path)
        if not rows:
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes = [yolo_to_xyxy(row, width, height) for row in rows]
        target_boxes = [box for box in boxes if box[0] in target_classes]
        for box_idx, box in enumerate(target_boxes):
            for variant in range(args.crops_per_box):
                crop = crop_for_box(box, width, height, context=args.context, min_size=args.min_size, variant=variant)
                clipped = [clip_box(other, crop) for other in boxes]
                clipped = [item for item in clipped if item is not None]
                if not clipped:
                    continue
                left, top, right, bottom = crop
                crop_image = image.crop(crop)
                stem = f"{image_path.stem}__crop_cls{box[0]}_{box_idx:02d}_{variant:02d}"
                crop_image.save(out_image_dir / f"{stem}.jpg", quality=95)
                crop_w = right - left
                crop_h = bottom - top
                lines = [xyxy_to_yolo(item, crop_w, crop_h) for item in clipped]
                (out_label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                crops += 1
    return {"copied_files": copied, "crop_images": crops}


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    target_classes = {int(item) for item in args.target_classes.split(",") if item.strip()}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dir": str(source_dir),
        "target_classes": sorted(target_classes),
        "train": build_train(source_dir, output_dir, target_classes, args),
        "valid_files": copy_split(source_dir, output_dir, "valid", args.link_mode),
        "test_files": copy_split(source_dir, output_dir, "test", args.link_mode),
    }
    for name in ["data.yaml", "data_split.json"]:
        src = source_dir / name
        if src.exists():
            link_or_copy(src, output_dir / name, args.link_mode)
    (output_dir / "crop_aug_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
