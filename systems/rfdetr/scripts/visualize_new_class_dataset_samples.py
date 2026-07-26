#!/usr/bin/env python3
"""Render a contact sheet of a new-class dataset split with its boxes drawn.

Visual confirmation that the 1:1 pairing survived deduplication and dataset
construction: every box drawn here comes from the label file that shipped
alongside the image, so a misaligned box means the pairing or the delivery is
wrong rather than the model.

```bash
python systems/rfdetr/scripts/visualize_new_class_dataset_samples.py \
  --dataset-dir data/rfdetr_brace_bcd_20260725_test_as_valid \
  --split train --limit 24 \
  --output outputs/rfdetr_new_classes/brace_train_samples.jpg
```
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADE_COLORS = {0: (46, 134, 222), 1: (255, 159, 26), 2: (232, 65, 24)}
GRADE_NAMES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--cell", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            boxes.append((int(fields[0]), *(float(value) for value in fields[1:])))
    return boxes


def render_cell(image_path: Path, label_path: Path, cell: int) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((cell, cell), Image.LANCZOS)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for class_id, cx, cy, box_w, box_h in load_boxes(label_path):
        x0 = (cx - box_w / 2) * width
        y0 = (cy - box_h / 2) * height
        x1 = (cx + box_w / 2) * width
        y1 = (cy + box_h / 2) * height
        color = GRADE_COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = GRADE_NAMES.get(class_id, str(class_id))
        draw.rectangle([x0, max(0, y0 - 16), x0 + 16, max(16, y0)], fill=color)
        draw.text((x0 + 4, max(1, y0 - 15)), tag, fill=(255, 255, 255))

    canvas = Image.new("RGB", (cell, cell), (24, 24, 24))
    canvas.paste(image, ((cell - width) // 2, (cell - height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, cell - 14), image_path.stem, fill=(200, 200, 200))
    return canvas


def main() -> None:
    args = parse_args()
    split_dir = Path(args.dataset_dir) / args.split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"no images in {images_dir}")

    rng = random.Random(args.seed)
    # Prefer a mix that shows the scarce grades rather than a uniform sample.
    by_rarest: dict[int, list[Path]] = {0: [], 1: [], 2: []}
    for image_path in images:
        boxes = load_boxes(labels_dir / f"{image_path.stem}.txt")
        present = {box[0] for box in boxes}
        rarest = 2 if 2 in present else (1 if 1 in present else 0)
        by_rarest[rarest].append(image_path)

    chosen: list[Path] = []
    for grade in (2, 1, 0):
        pool = by_rarest[grade][:]
        rng.shuffle(pool)
        chosen.extend(pool[: max(1, args.limit // 3)])
    chosen = chosen[: args.limit]

    columns = min(args.columns, len(chosen))
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * args.cell, rows * args.cell), (12, 12, 12))
    for index, image_path in enumerate(chosen):
        cell_image = render_cell(image_path, labels_dir / f"{image_path.stem}.txt", args.cell)
        sheet.paste(cell_image, ((index % columns) * args.cell, (index // columns) * args.cell))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    print(f"wrote {output} with {len(chosen)} samples from {split_dir}")


if __name__ == "__main__":
    main()
