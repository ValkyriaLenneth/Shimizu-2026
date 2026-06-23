#!/usr/bin/env python3
"""Build GT-vs-prediction review pages for the coarse YOLO model."""

from __future__ import annotations

import argparse
import html
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = ["天井", "内壁", "RC壁", "RC柱"]
COLORS = {
    0: (44, 123, 229),
    1: (38, 166, 91),
    2: (230, 126, 34),
    3: (155, 89, 182),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coarse_router_yolov9/datasets/coarse_cross_fixed_copy")
    parser.add_argument("--pred-root", default="coarse_router_yolov9/runs/detect")
    parser.add_argument("--output", default="coarse_router_yolov9/qa/model_review_conf025")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def image_paths(dataset: Path, split: str) -> list[Path]:
    paths: list[Path] = []
    for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG"]:
        paths.extend((dataset / "images" / split).glob(pattern))
    return sorted(paths)


def draw_yolo(src: Path, label_path: Path, dst: Path, show_conf: bool) -> int:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    count = 0
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
            conf = float(parts[5]) if show_conf and len(parts) > 5 else None
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            x2 = (x + w / 2) * width
            y2 = (y + h / 2) * height
            color = COLORS.get(cls, (255, 0, 0))
            line_width = max(2, width // 350)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            label = CLASSES[cls] if cls < len(CLASSES) else str(cls)
            if conf is not None:
                label = f"{label} {conf:.2f}"
            bbox = draw.textbbox((0, 0), label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            y_text = max(0, y1 - th - 6)
            draw.rectangle([x1, y_text, x1 + tw + 8, y_text + th + 6], fill=color)
            draw.text((x1 + 4, y_text + 3), label, fill=(255, 255, 255))
            count += 1
    image.thumbnail((900, 900))
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, quality=90)
    return count


def pred_dir_name(split: str) -> str:
    return f"best_{split}_conf025"


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    dataset = root / args.dataset
    pred_root = root / args.pred_root
    output = root / args.output
    rng = random.Random(args.seed)

    split_links = []
    for split in args.splits:
        paths = image_paths(dataset, split)
        if args.max_images:
            rng.shuffle(paths)
            paths = sorted(paths[: args.max_images])

        cards = []
        for src in paths:
            gt_label = dataset / "labels" / split / f"{src.stem}.txt"
            pred_label = pred_root / pred_dir_name(split) / "labels" / f"{src.stem}.txt"
            gt_rel = Path("assets") / split / "gt" / f"{src.stem}.jpg"
            pred_rel = Path("assets") / split / "pred" / f"{src.stem}.jpg"
            gt_count = draw_yolo(src, gt_label, output / gt_rel, show_conf=False)
            pred_count = draw_yolo(src, pred_label, output / pred_rel, show_conf=True)
            cards.append(
                "<article class='card'>"
                f"<h2>{html.escape(src.name)}</h2>"
                f"<div class='meta'>GT boxes: {gt_count} | Pred boxes: {pred_count}</div>"
                "<div class='pair'>"
                f"<figure><figcaption>GT</figcaption><img src='{html.escape(str(gt_rel))}'></figure>"
                f"<figure><figcaption>Pred conf>=0.25</figcaption><img src='{html.escape(str(pred_rel))}'></figure>"
                "</div></article>"
            )

        page_name = f"{split}.html"
        split_links.append(f"<a href='{page_name}'>{split}</a>")
        (output / page_name).write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Coarse Router Review - {html.escape(split)}</title>"
            "<style>body{font-family:Arial,sans-serif;margin:20px;background:#f5f6f8;color:#1f2933}"
            "nav{margin-bottom:16px}nav a{margin-right:12px}.card{background:white;border:1px solid #d9dee7;border-radius:6px;padding:12px;margin-bottom:14px}"
            "h1{font-size:22px}h2{font-size:15px;margin:0 0 4px}.meta{font-size:13px;color:#52606d;margin-bottom:8px}"
            ".pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}figure{margin:0}figcaption{font-weight:700;font-size:13px;margin-bottom:4px}"
            "img{width:100%;height:auto;display:block;border:1px solid #e1e5ea;background:#fff}"
            "@media(max-width:900px){.pair{grid-template-columns:1fr}}</style></head><body>"
            f"<nav>{' '.join(split_links)}</nav><h1>{html.escape(split)} review</h1>"
            + "\n".join(cards)
            + "</body></html>",
            encoding="utf-8",
        )

    index = output / "index.html"
    index.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Coarse Router Review</title></head><body>"
        "<h1>Coarse Router Review</h1><ul>"
        + "".join(f"<li>{link}</li>" for link in split_links)
        + "</ul></body></html>",
        encoding="utf-8",
    )
    print(f"wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
