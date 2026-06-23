#!/usr/bin/env python3
"""Visualize E2E debug samples with GT, router boxes, and crack detections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]

COLORS = {
    "gt": (255, 255, 255),
    "router": (0, 215, 255),
    "main": (0, 255, 0),
    "secondary": (255, 128, 0),
    "miss": (0, 0, 255),
    "text_bg": (20, 20, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--contact-thumb-width", type=int, default=520)
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    e2e_dir = resolve(args.e2e_dir)
    out_dir = resolve(args.output_dir) if args.output_dir else e2e_dir / "issue_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = list(csv.DictReader((e2e_dir / "eval_by_image.csv").open(encoding="utf-8")))
    results = [json.loads(line) for line in (e2e_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_image = {str(Path(r["image"]).resolve()): r for r in results}

    issue_rows = [r for r in eval_rows if has_issue(r)]
    issue_rows.sort(key=issue_sort_key)
    if args.limit > 0:
        issue_rows = issue_rows[: args.limit]

    summary_rows: list[dict[str, Any]] = []
    contact_items: list[tuple[np.ndarray, str]] = []
    for index, row in enumerate(issue_rows, start=1):
        image_path = (REPO / row["image"]).resolve()
        result = by_image.get(str(image_path))
        if not result:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        label_path = resolve(result["eval_meta"]["label"])
        gt = load_gt(label_path, image.shape[1], image.shape[0])
        issue_tags = classify_issue(row)
        vis = draw_sample(image, row, result, gt, issue_tags, args.max_side)
        stem = safe_stem(f"{index:02d}_{row['class_key']}_{Path(row['image']).stem}_{'_'.join(issue_tags)}")
        rel_out = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(rel_out), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        summary_rows.append(
            {
                "visualization": str(rel_out.relative_to(REPO)),
                "image": row["image"],
                "class_key": row["class_key"],
                "issue_tags": "|".join(issue_tags),
                "router_classes": row["router_classes"],
                "gt_boxes": row["gt_boxes"],
                "main_predictions": row["main_predictions"],
                "secondary_predictions": row["secondary_predictions"],
                "main_matches_iou50": row["main_matches_iou50"],
                "main_false_negative": row["main_false_negative"],
                "main_false_positive": row["main_false_positive"],
                "matched_grade_mismatch": row["matched_grade_mismatch"],
            }
        )
        contact_items.append((make_thumb(vis, args.contact_thumb_width), f"{index:02d} {row['class_key']} {'/'.join(issue_tags)}"))

    write_csv(out_dir / "issue_visualization_index.csv", summary_rows)
    if contact_items:
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), make_contact_sheet(contact_items), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(json.dumps({"issues": len(summary_rows), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def has_issue(row: dict[str, str]) -> bool:
    return any(
        int(row[key]) > 0
        for key in [
            "fn_router_miss",
            "fn_no_main_output",
            "fn_iou_miss",
            "main_false_positive",
            "matched_grade_mismatch",
        ]
    )


def issue_sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    class_rank = {"rc_column": 0, "rc_wall": 1, "inner_wall": 2, "tenjo": 3}.get(row["class_key"], 9)
    severity = (
        10 * int(row["fn_router_miss"])
        + 7 * int(row["fn_no_main_output"])
        + 5 * int(row["fn_iou_miss"])
        + 2 * int(row["matched_grade_mismatch"])
        + int(row["main_false_positive"])
    )
    return (class_rank, -severity, -int(row["main_false_negative"]), -int(row["main_false_positive"]), row["image"])


def classify_issue(row: dict[str, str]) -> list[str]:
    tags = []
    if int(row["fn_router_miss"]) > 0:
        tags.append("router_miss")
    if int(row["fn_no_main_output"]) > 0:
        tags.append("no_main")
    if int(row["fn_iou_miss"]) > 0:
        tags.append("iou_miss")
    if int(row["main_false_positive"]) > 0:
        tags.append("fp")
    if int(row["matched_grade_mismatch"]) > 0:
        tags.append("grade_mismatch")
    return tags or ["ok"]


def load_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    gt = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = [float(v) for v in parts]
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height
        gt.append({"bbox_xyxy": [x1, y1, x2, y2], "grade": {0: "B", 1: "C", 2: "D"}.get(int(cls), str(int(cls)))})
    return gt


def draw_sample(image: np.ndarray, row: dict[str, str], result: dict[str, Any], gt: list[dict[str, Any]], issue_tags: list[str], max_side: int) -> np.ndarray:
    canvas = image.copy()
    scale = min(1.0, max_side / max(canvas.shape[:2]))
    if scale < 1.0:
        canvas = cv2.resize(canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    def box(values: list[float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = values
        return round(x1 * scale), round(y1 * scale), round(x2 * scale), round(y2 * scale)

    for i, det in enumerate((result.get("router") or {}).get("detections", [])):
        draw_box(canvas, box(det["bbox_xyxy"]), COLORS["router"], f"R{i}:{det.get('class_name')} {float(det.get('confidence', 0)):.2f}", 2)

    for i, item in enumerate(gt):
        draw_box(canvas, box(item["bbox_xyxy"]), COLORS["gt"], f"GT{i}:{item['grade']}", 3)

    main_model = row["main_model"]
    for i, det in enumerate(result.get("raw_crack_detections", [])):
        is_main = det.get("source_model") == main_model
        color = COLORS["main"] if is_main else COLORS["secondary"]
        label = f"{'M' if is_main else 'S'}{i}:{det.get('source_model')} {grade_from_text(det.get('damage_grade', ''))} {float(det.get('confidence', 0)):.2f}"
        draw_box(canvas, box(det["bbox_xyxy"]), color, label, 2)

    header = [
        f"{row['class_key']}  issues={','.join(issue_tags)}",
        f"router expected={row['expected_router_class']} got={row['router_classes']}",
        f"gt={row['gt_boxes']} main={row['main_predictions']} sec={row['secondary_predictions']} match={row['main_matches_iou50']} fn={row['main_false_negative']} fp={row['main_false_positive']} grade_mis={row['matched_grade_mismatch']}",
        str(Path(row["image"])),
    ]
    draw_header(canvas, header)
    draw_legend(canvas)
    return canvas


def draw_box(img: np.ndarray, xyxy: tuple[int, int, int, int], color: tuple[int, int, int], label: str, thickness: int) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    put_label(img, label, (x1, max(18, y1 - 4)), color)


def put_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    x = max(0, min(x, img.shape[1] - w - 4))
    y = max(h + 4, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 3), COLORS["text_bg"], -1)
    cv2.putText(img, text, (x + 2, y), font, scale, color, thick, cv2.LINE_AA)


def draw_header(img: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 1
    line_h = 24
    h = line_h * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line[:180], (8, 24 + i * line_h), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_legend(img: np.ndarray) -> None:
    labels = [("GT", COLORS["gt"]), ("Router", COLORS["router"]), ("Main crack", COLORS["main"]), ("Secondary crack", COLORS["secondary"])]
    x = 8
    y = img.shape[0] - 12
    for text, color in labels:
        cv2.rectangle(img, (x, y - 14), (x + 18, y), color, -1)
        cv2.putText(img, text, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += 132


def grade_from_text(text: str) -> str:
    match = re.search(r"程度([BCD])", text)
    return match.group(1) if match else "?"


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def make_thumb(img: np.ndarray, width: int) -> np.ndarray:
    scale = width / img.shape[1]
    return cv2.resize(img, (width, max(1, round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def make_contact_sheet(items: list[tuple[np.ndarray, str]]) -> np.ndarray:
    cols = 3
    pad = 12
    label_h = 30
    widths = [img.shape[1] for img, _ in items]
    thumb_w = max(widths)
    thumb_h = max(img.shape[0] for img, _ in items)
    rows = math.ceil(len(items) / cols)
    sheet = np.full((rows * (thumb_h + label_h + pad) + pad, cols * (thumb_w + pad) + pad, 3), 245, dtype=np.uint8)
    for idx, (img, label) in enumerate(items):
        r, c = divmod(idx, cols)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + label_h + pad)
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
        cv2.putText(sheet, label[:70], (x, y + thumb_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return sheet


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
