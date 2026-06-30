#!/usr/bin/env python3
"""Build high-DPI router/raw/final stage visualizations for pipeline review."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rfdetr_prod_pipeline.pipeline.run_full_pipeline import (
    apply_router_selection_policy,
    build_router,
    load_config,
    resolve_path,
)


GRADE_BY_ID = {0: "B", 1: "C", 2: "D"}
EXPECTED_ROUTER_CLASS = {
    "tenjo": "天井",
    "inner_wall": "壁类",
    "rc_wall": "壁类",
    "rc_column": "RC柱",
}
MANUAL_PRIORITY = [
    "inner_wall__labels_20251107__b-30151.jpg",
    "rc_wall__data_add100__c-40537.jpg",
    "tenjo__data_add100__1-B-10086.jpg",
    "tenjo__data_add100__a-40251.jpg",
    "tenjo__labels_20251107__a-30042.jpg",
    "tenjo__data_add100__1-C-00014.jpg",
    "tenjo__data_add100__1-D-00016.jpg",
    "tenjo__data_add100__1-D-10006.jpg",
    "tenjo__data_add100__1-D-10061.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--analysis", required=True, help="per_image_analysis.csv from evaluate_pipeline_outputs.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--panel-max-side", type=int, default=1500)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--final-source", choices=["display", "crack"], default="display")
    parser.add_argument("--focus-gt", action="store_true", help="Keep the boxes most relevant to GT first.")
    parser.add_argument("--max-boxes-per-panel", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    montage_dir = out_dir / "montages"
    image_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_path(args.config, repo)
    config = load_config(config_path)
    router = build_router(config["pipeline"], config, config_path.parent, args.device)

    results = load_results(Path(args.results))
    samples = load_samples(Path(args.split))
    samples_by_name = {Path(row["eval_image"]).name: row for row in samples}
    analysis_rows = list(csv.DictReader(Path(args.analysis).open(encoding="utf-8")))
    selected = select_rows(analysis_rows, args.limit)

    summary_rows: list[dict[str, Any]] = []
    thumbs: list[Image.Image] = []
    for index, row in enumerate(selected):
        name = Path(row["image"]).name
        sample = samples_by_name.get(name)
        result = results.get(name)
        if not sample or not result:
            continue
        image_path = Path(sample["eval_image"])
        label_path = Path(sample["eval_label"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        raw_router = router.predict(image_bgr)
        post_router = apply_router_selection_policy(raw_router, config["pipeline"])
        gt = load_gt(label_path, image_bgr.shape[1], image_bgr.shape[0])

        canvas = build_stage_canvas(
            image_bgr=image_bgr,
            gt=gt,
            raw_router=raw_router,
            post_router=post_router,
            result=result,
            component=str(row.get("component", "")),
            metrics=row,
            max_side=args.panel_max_side,
            final_source=args.final_source,
            focus_gt=args.focus_gt,
            max_boxes_per_panel=args.max_boxes_per_panel,
        )
        out_name = f"{index:02d}_{row.get('component','unknown')}__{name}"
        out_path = image_dir / out_name
        canvas.save(out_path, quality=args.jpeg_quality, subsampling=0, dpi=(args.dpi, args.dpi))
        thumbs.append(make_thumb(canvas, width=900))
        summary_rows.append(
            {
                "index": index,
                "image": name,
                "component": row.get("component", ""),
                "strict_tp": row.get("strict_tp", ""),
                "strict_fp": row.get("strict_fp", ""),
                "strict_fn": row.get("strict_fn", ""),
                "loc_tp": row.get("loc_tp", ""),
                "loc_fp": row.get("loc_fp", ""),
                "loc_fn": row.get("loc_fn", ""),
                "router_raw": router_classes(raw_router),
                "router_post": router_classes(post_router),
                "warnings": result.get("warnings", []),
                "visualization": str(out_path.relative_to(out_dir)),
            }
        )

    write_csv(out_dir / "summary.csv", summary_rows)
    if thumbs:
        build_montage(thumbs[:24]).save(montage_dir / "top24_montage.jpg", quality=94, subsampling=0, dpi=(args.dpi, args.dpi))
    build_index(out_dir, summary_rows)
    print(json.dumps({"output_dir": str(out_dir), "images": len(summary_rows)}, ensure_ascii=False, indent=2))
    return 0


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {Path(row["image"]).name: row for row in rows}


def load_samples(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))["samples"]
    return list(csv.DictReader(path.open(encoding="utf-8")))


def select_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_name = {Path(row["image"]).name: row for row in rows}
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in MANUAL_PRIORITY:
        row = by_name.get(name)
        if row:
            selected.append(row)
            seen.add(name)
    ranked = sorted(
        rows,
        key=lambda row: (
            -(int(row.get("strict_fn") or 0) * 4 + int(row.get("strict_fp") or 0) + int(row.get("loc_fn") or 0) * 2),
            row.get("component", ""),
            Path(row.get("image", "")).name,
        ),
    )
    for row in ranked:
        name = Path(row["image"]).name
        if name in seen:
            continue
        selected.append(row)
        seen.add(name)
        if len(selected) >= limit:
            break
    return selected[:limit]


def load_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = [float(v) for v in parts]
        out.append(
            {
                "grade": GRADE_BY_ID.get(int(cls), str(int(cls))),
                "bbox_xyxy": [
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ],
            }
        )
    return out


def build_stage_canvas(
    image_bgr: np.ndarray,
    gt: list[dict[str, Any]],
    raw_router: dict[str, Any],
    post_router: dict[str, Any],
    result: dict[str, Any],
    component: str,
    metrics: dict[str, str],
    max_side: int,
    final_source: str,
    focus_gt: bool,
    max_boxes_per_panel: int,
) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    raw_boxes = detection_boxes(result.get("raw_crack_detections", []), raw=True)
    final_items = result.get("display_crack_detections") or result.get("crack_detections") or []
    if final_source == "crack":
        final_items = result.get("crack_detections") or []
    final_boxes = detection_boxes(final_items, raw=False)
    if focus_gt:
        raw_boxes = focus_boxes(raw_boxes, gt, max_boxes_per_panel)
        final_boxes = focus_boxes(final_boxes, gt, max_boxes_per_panel)
    panels = [
        draw_panel(
            image,
            title="自動識別結果",
            subtitle=f"expected={EXPECTED_ROUTER_CLASS.get(component, component)}",
            gt=gt,
            boxes=router_boxes(raw_router),
            max_side=max_side,
        ),
        draw_panel(
            image,
            title="自動識別後処理結果",
            subtitle=f"classes={router_classes(post_router)}",
            gt=gt,
            boxes=router_boxes(post_router),
            max_side=max_side,
        ),
        draw_panel(
            image,
            title="損傷判別結果",
            subtitle=f"raw={len(result.get('raw_crack_detections', []))} shown={len(raw_boxes)}",
            gt=gt,
            boxes=raw_boxes,
            max_side=max_side,
        ),
        draw_panel(
            image,
            title="後処理後の最終表示",
            subtitle=f"{final_source} shown={len(final_boxes)} TP/FP/FN={metrics.get('strict_tp')}/{metrics.get('strict_fp')}/{metrics.get('strict_fn')}",
            gt=gt,
            boxes=final_boxes,
            max_side=max_side,
        ),
    ]
    gap = 20
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def draw_panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    gt: list[dict[str, Any]],
    boxes: list[dict[str, Any]],
    max_side: int,
) -> Image.Image:
    scale = min(max_side / max(image.width, image.height), 1.0)
    panel_image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    header_h = 104
    panel = Image.new("RGB", (panel_image.width, panel_image.height + header_h), (250, 250, 250))
    panel.paste(panel_image, (0, header_h))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width - 1, panel.height - 1), outline=(80, 80, 80), width=2)
    font_title = font(34)
    font_sub = font(22)
    draw.text((18, 12), title, font=font_title, fill=(15, 15, 15))
    draw.text((18, 58), subtitle, font=font_sub, fill=(70, 70, 70))
    for item in boxes:
        draw_box(draw, item["bbox_xyxy"], scale, header_h, item["color"], item["label"], width=item.get("width", 4))
    for item in gt:
        draw_box(draw, item["bbox_xyxy"], scale, header_h, (230, 40, 40), f"GT-{item['grade']}", width=5)
    return panel


def router_boxes(router_result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for det in router_result.get("detections", []):
        label = f"R-{det.get('class_name')} {float(det.get('confidence', 0.0)):.2f}"
        out.append({"bbox_xyxy": det["bbox_xyxy"], "label": label, "color": (20, 165, 80), "width": 5})
    return out


def detection_boxes(items: list[dict[str, Any]], raw: bool) -> list[dict[str, Any]]:
    out = []
    for det in items:
        bbox = det.get("bbox_xyxy")
        if not bbox:
            continue
        model = str(det.get("source_model", ""))
        grade = str(det.get("damage_grade", ""))
        conf = float(det.get("confidence", 0.0))
        transport = str(det.get("region_transport", ""))
        prefix = "raw" if raw else "final"
        suffix = " rescue" if "rescue" in transport else ""
        color = (35, 95, 220) if raw else (170, 45, 185)
        if "rescue" in transport:
            color = (235, 140, 30)
        out.append(
            {
                "bbox_xyxy": bbox,
                "label": f"{prefix}-{grade} {model} {conf:.2f}{suffix}",
                "color": color,
                "width": 4,
                "confidence": conf,
            }
        )
    return out


def focus_boxes(boxes: list[dict[str, Any]], gt: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(boxes) <= limit:
        return boxes
    gt_boxes = [item["bbox_xyxy"] for item in gt]
    ranked = sorted(
        boxes,
        key=lambda item: (
            max((iou_xyxy(item["bbox_xyxy"], gt_box) for gt_box in gt_boxes), default=0.0),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    scale: float,
    header_h: int,
    color: tuple[int, int, int],
    label: str,
    width: int,
) -> None:
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
    y1 += header_h
    y2 += header_h
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    label_font = font(20)
    bbox = draw.textbbox((0, 0), label, font=label_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    lx = max(0, min(x1, draw.im.size[0] - tw - 10))
    ly = max(header_h, y1 - th - 10)
    draw.rectangle((lx, ly, lx + tw + 8, ly + th + 6), fill=(255, 255, 255), outline=color, width=2)
    draw.text((lx + 4, ly + 2), label, font=label_font, fill=color)


def font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def router_classes(router_result: dict[str, Any]) -> str:
    return "|".join(str(det.get("class_name", "")) for det in router_result.get("detections", []))


def make_thumb(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    return image.resize((width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)


def build_montage(images: list[Image.Image], columns: int = 2, gap: int = 18) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    canvas = Image.new("RGB", (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap), "white")
    for index, img in enumerate(images):
        row, col = divmod(index, columns)
        canvas.paste(img, (col * (cell_w + gap), row * (cell_h + gap)))
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_index(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:'Noto Sans CJK JP','Noto Sans CJK SC',sans-serif;margin:24px}"
        "img{max-width:1200px;border:1px solid #ccc}td{vertical-align:top;padding:8px;border-top:1px solid #ddd}"
        "code{background:#eee;padding:2px 4px}</style></head><body>",
        "<h1>Router Stage Visual Review</h1>",
        "<p>Panels: 自動識別結果 / 自動識別後処理結果 / 損傷判別結果 / 後処理後の最終表示. Red boxes are GT.</p>",
    ]
    montage = out_dir / "montages" / "top24_montage.jpg"
    if montage.exists():
        body.append("<h2>Top Montage</h2>")
        body.append("<p><a href='montages/top24_montage.jpg'><img src='montages/top24_montage.jpg'></a></p>")
    body.append("<table>")
    for row in rows:
        path = html.escape(str(row["visualization"]))
        body.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td><code>{html.escape(str(row['image']))}</code><br>{html.escape(str(row['component']))}<br>"
            f"raw: {html.escape(str(row['router_raw']))}<br>post: {html.escape(str(row['router_post']))}</td>"
            f"<td><a href='{path}'><img src='{path}'></a></td>"
            "</tr>"
        )
    body.append("</table></body></html>")
    (out_dir / "index.html").write_text("\n".join(body), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
