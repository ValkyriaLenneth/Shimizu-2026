#!/usr/bin/env python3
"""Create quick visual QA sheets for a YOLO-format dataset."""

from __future__ import annotations

import argparse
import html
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = ["天井", "内壁", "RC壁", "RC柱"]
COLORS = {
    0: (44, 123, 229),
    1: (38, 166, 91),
    2: (230, 126, 34),
    3: (155, 89, 182),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coarse_router_yolov9/datasets/coarse_cross_fixed")
    parser.add_argument("--output", default="coarse_router_yolov9/qa/coarse_cross_fixed_labels")
    parser.add_argument("--per-split", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def draw_labels(image_path: Path, label_path: Path, output_path: Path) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    count = 0

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = [float(x) for x in parts[1:5]]
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            color = COLORS.get(cls, (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, width // 300))
            text = CLASSES[cls] if cls < len(CLASSES) else str(cls)
            text_bbox = draw.textbbox((0, 0), text)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, max(0, y1 - th - 6), x1 + tw + 8, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - th - 4)), text, fill=(255, 255, 255))
            count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((960, 960))
    image.save(output_path, quality=90)
    return count


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    dataset = root / args.dataset
    output = root / args.output
    samples_dir = output / "samples"
    rng = random.Random(args.seed)

    rows = []
    for split in ["train", "val", "test"]:
        image_paths = []
        for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG"]:
            image_paths.extend((dataset / "images" / split).glob(pattern))
        image_paths = sorted(image_paths)
        rng.shuffle(image_paths)
        for image_path in image_paths[: args.per_split]:
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            rel_out = Path("samples") / split / f"{image_path.stem}.jpg"
            count = draw_labels(image_path, label_path, output / rel_out)
            rows.append((split, image_path.name, str(rel_out), count))

    html_rows = "\n".join(
        "<div class='card'>"
        f"<img src='{html.escape(path)}' loading='lazy'>"
        f"<div><b>{html.escape(split)}</b> {html.escape(name)} | boxes: {count}</div>"
        "</div>"
        for split, name, path, count in rows
    )
    (output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Coarse YOLO Labels</title>"
        "<style>body{font-family:sans-serif;margin:20px;background:#f6f7f9}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}"
        ".card{background:#fff;border:1px solid #ddd;padding:8px;border-radius:6px}"
        "img{width:100%;height:auto;display:block}</style></head><body>"
        "<h1>Coarse YOLO Label Preview</h1><div class='grid'>"
        + html_rows
        + "</div></body></html>",
        encoding="utf-8",
    )
    print(f"wrote {output / 'index.html'} with {len(rows)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
