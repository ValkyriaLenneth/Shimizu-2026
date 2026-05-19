#!/usr/bin/env python3
"""Audit Gemini router annotations and build 3-class YOLO datasets.

This script turns Gemini coarse building-element annotations into:
- quantitative QA metrics
- suspicious-sample review lists
- full and cleaned 3-class YOLO datasets
- visual examples and process documentation
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

SOURCE_CLASSES = ["天井", "内壁", "RC壁", "RC柱"]
MERGED_CLASSES = ["天井", "壁类", "RC柱"]
MERGE_LABEL = {"天井": "天井", "内壁": "壁类", "RC壁": "壁类", "RC柱": "RC柱"}
MERGED_TO_ID = {name: i for i, name in enumerate(MERGED_CLASSES)}
CRITICAL_ISSUES = {
    "not_ok",
    "image_missing",
    "image_unreadable",
    "empty_elements",
    "no_valid_boxes",
    "invalid_label",
    "invalid_bbox_shape",
    "invalid_bbox_value",
    "invalid_bbox_geometry",
    "image_unrecoverable",
}
COLORS = {"天井": (0, 160, 220), "壁类": (40, 170, 95), "RC柱": (220, 115, 40), "invalid": (220, 40, 40)}


@dataclass
class BoxRecord:
    source_label: str
    merged_label: str | None
    confidence: float | None
    reason: str
    bbox_1000: list[float]
    bbox_norm: tuple[float, float, float, float] | None
    yolo: tuple[int, float, float, float, float] | None
    area_ratio: float | None
    aspect_ratio: float | None
    issues: list[str] = field(default_factory=list)


@dataclass
class AuditRow:
    row_index: int
    image_path: str
    image_rel_path: str
    expected_label: str
    expected_merged_label: str | None
    ok: bool
    error: str | None
    score: int
    issues: list[str]
    image_width: int | None
    image_height: int | None
    elements_count: int
    valid_boxes_count: int
    kept_boxes_count: int
    labels_raw: list[str]
    labels_merged: list[str]
    max_confidence: float | None
    min_confidence: float | None
    max_area_ratio: float | None
    boxes: list[BoxRecord]
    source_row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19/results.jsonl")
    parser.add_argument("--image-root", default="data/unzip")
    parser.add_argument("--qa-dir", default="outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19/qa")
    parser.add_argument("--full-dataset", default="coarse_router_yolov9/datasets/coarse_router_3class_full")
    parser.add_argument("--cleaned-dataset", default="coarse_router_yolov9/datasets/coarse_router_3class_cleaned")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--clean-score-threshold", type=int, default=80)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.98)
    parser.add_argument("--min-side-ratio", type=float, default=0.04)
    parser.add_argument("--max-aspect-ratio", type=float, default=20.0)
    parser.add_argument("--too-many-boxes", type=int, default=6)
    parser.add_argument("--viz-per-group", type=int, default=80)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parsed_payload(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("response") or {}).get("parsed") or {}


def image_path_for(row: dict[str, Any], image_root: Path) -> Path:
    p = Path(row.get("image_path") or "")
    if p.exists():
        return p
    rel = row.get("image_rel_path") or ""
    return image_root / rel


def normalize_bbox(raw: Any) -> tuple[tuple[float, float, float, float] | None, list[str]]:
    issues: list[str] = []
    if not isinstance(raw, list) or len(raw) != 4:
        return None, ["invalid_bbox_shape"]
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in raw]
    except Exception:
        return None, ["invalid_bbox_value"]
    if any(math.isnan(v) or math.isinf(v) for v in [ymin, xmin, ymax, xmax]):
        return None, ["invalid_bbox_value"]
    if min(ymin, xmin, ymax, xmax) < 0 or max(ymin, xmin, ymax, xmax) > 1000:
        issues.append("bbox_out_of_bounds_clipped")
    ymin = max(0.0, min(1000.0, ymin))
    xmin = max(0.0, min(1000.0, xmin))
    ymax = max(0.0, min(1000.0, ymax))
    xmax = max(0.0, min(1000.0, xmax))
    if ymax <= ymin or xmax <= xmin:
        return None, issues + ["invalid_bbox_geometry"]
    return (ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0), issues


def box_to_yolo(merged_label: str, bbox: tuple[float, float, float, float]) -> tuple[int, float, float, float, float]:
    ymin, xmin, ymax, xmax = bbox
    bw = xmax - xmin
    bh = ymax - ymin
    return (MERGED_TO_ID[merged_label], xmin + bw / 2, ymin + bh / 2, bw, bh)


def score_row(issues: list[str], valid_boxes: int) -> int:
    penalties = {
        "not_ok": 100,
        "image_missing": 100,
        "image_unreadable": 100,
        "image_unrecoverable": 100,
        "image_truncated_recovered": 8,
        "image_reencoded": 0,
        "empty_elements": 70,
        "no_valid_boxes": 70,
        "invalid_label": 45,
        "invalid_bbox_shape": 45,
        "invalid_bbox_value": 45,
        "invalid_bbox_geometry": 45,
        "bbox_out_of_bounds_clipped": 20,
        "low_confidence": 18,
        "label_mismatch": 18,
        "multi_class_conflict": 15,
        "too_many_boxes": 15,
        "too_small_box": 14,
        "too_large_box": 12,
        "thin_box": 10,
        "almost_full_image_box": 8,
    }
    total = 0
    for issue in set(issues):
        total += penalties.get(issue, 5)
    if valid_boxes == 0:
        total += 30
    return max(0, 100 - total)


def audit_one(row: dict[str, Any], index: int, image_root: Path, args: argparse.Namespace) -> AuditRow:
    issues: list[str] = []
    boxes: list[BoxRecord] = []
    image_path = image_path_for(row, image_root)
    width = height = None
    if not image_path.exists():
        issues.append("image_missing")
    else:
        width, height, image_issues = inspect_image(image_path)
        issues.extend(image_issues)

    expected = row.get("expected_label") or ""
    expected_merged = MERGE_LABEL.get(expected)
    if not row.get("ok"):
        issues.append("not_ok")
    payload = parsed_payload(row)
    elements = payload.get("elements") or []
    if not elements:
        issues.append("empty_elements")
    if len(elements) > args.too_many_boxes:
        issues.append("too_many_boxes")

    raw_labels: list[str] = []
    merged_labels: list[str] = []
    confidences: list[float] = []
    areas: list[float] = []

    for element in elements:
        label = str(element.get("label", ""))
        raw_labels.append(label)
        merged = MERGE_LABEL.get(label)
        box_issues: list[str] = []
        if merged is None:
            box_issues.append("invalid_label")
            issues.append("invalid_label")
        else:
            merged_labels.append(merged)
        conf = element.get("confidence")
        conf_float = None
        if isinstance(conf, (int, float)):
            conf_float = float(conf)
            confidences.append(conf_float)
            if conf_float < args.min_confidence:
                box_issues.append("low_confidence")
                issues.append("low_confidence")
        bbox_norm, norm_issues = normalize_bbox(element.get("bbox_2d"))
        box_issues.extend(norm_issues)
        issues.extend(norm_issues)
        yolo = None
        area = None
        aspect = None
        if bbox_norm is not None and merged is not None:
            ymin, xmin, ymax, xmax = bbox_norm
            bw = xmax - xmin
            bh = ymax - ymin
            area = bw * bh
            areas.append(area)
            aspect = max(bw / bh, bh / bw) if bw > 0 and bh > 0 else None
            if area < args.min_area_ratio:
                box_issues.append("too_small_box")
                issues.append("too_small_box")
            if area > args.max_area_ratio:
                box_issues.append("too_large_box")
                issues.append("too_large_box")
            if bw < args.min_side_ratio or bh < args.min_side_ratio:
                box_issues.append("thin_box")
                issues.append("thin_box")
            if aspect is not None and aspect > args.max_aspect_ratio:
                box_issues.append("thin_box")
                issues.append("thin_box")
            if area > 0.90:
                box_issues.append("almost_full_image_box")
                issues.append("almost_full_image_box")
            yolo = box_to_yolo(merged, bbox_norm)
        boxes.append(BoxRecord(label, merged, conf_float, str(element.get("reason", "")), list(element.get("bbox_2d") or []), bbox_norm, yolo, area, aspect, box_issues))

    valid_boxes = [b for b in boxes if b.yolo is not None]
    if not valid_boxes:
        issues.append("no_valid_boxes")
    unique_merged = sorted(set(b.merged_label for b in valid_boxes if b.merged_label))
    if expected_merged and unique_merged and expected_merged not in unique_merged:
        issues.append("label_mismatch")
    if len(unique_merged) > 1:
        issues.append("multi_class_conflict")

    score = score_row(issues, len(valid_boxes))
    return AuditRow(
        row_index=index,
        image_path=str(image_path),
        image_rel_path=row.get("image_rel_path") or str(image_path),
        expected_label=expected,
        expected_merged_label=expected_merged,
        ok=bool(row.get("ok")),
        error=row.get("error"),
        score=score,
        issues=sorted(set(issues)),
        image_width=width,
        image_height=height,
        elements_count=len(elements),
        valid_boxes_count=len(valid_boxes),
        kept_boxes_count=len(valid_boxes),
        labels_raw=raw_labels,
        labels_merged=unique_merged,
        max_confidence=max(confidences) if confidences else None,
        min_confidence=min(confidences) if confidences else None,
        max_area_ratio=max(areas) if areas else None,
        boxes=boxes,
        source_row=row,
    )


def safe_stem(audit: AuditRow, seq: int) -> str:
    stem = Path(audit.image_rel_path).stem
    prefix = audit.expected_label or "unknown"
    return f"{prefix}_{stem}_{seq:05d}"


def link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
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


def inspect_image(path: Path) -> tuple[int | None, int | None, list[str]]:
    """Fully decode the image so truncated JPEGs are caught before YOLO scans them."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None, ["image_unreadable", "image_unrecoverable"]
    height, width = image.shape[:2]
    issues = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            with path.open("rb") as f:
                f.seek(-2, os.SEEK_END)
                if f.read() != b"\xff\xd9":
                    issues.append("image_truncated_recovered")
        except OSError:
            issues.append("image_truncated_recovered")
    return width, height, issues


def write_clean_image(src: Path, dst: Path) -> None:
    """Decode and re-encode as a baseline JPEG to remove truncated/corrupt markers."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {src}")
    ok = cv2.imwrite(str(dst), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"failed to write image: {dst}")


def split_audits(audits: list[AuditRow], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[AuditRow]]:
    rng = random.Random(seed)
    by_group: dict[str, list[AuditRow]] = defaultdict(list)
    for audit in audits:
        by_group[audit.expected_label or "unknown"].append(audit)
    splits = {"train": [], "val": [], "test": []}
    for rows in by_group.values():
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train+n_val])
        splits["test"].extend(rows[n_train+n_val:])
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def write_dataset(dataset_dir: Path, audits: list[AuditRow], args: argparse.Namespace, cleaned: bool) -> dict[str, Any]:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    valid = []
    for audit in audits:
        if cleaned and audit.score < args.clean_score_threshold:
            continue
        if cleaned and (set(audit.issues) & CRITICAL_ISSUES):
            continue
        if not cleaned and audit.valid_boxes_count == 0:
            continue
        if cleaned and audit.valid_boxes_count == 0:
            continue
        if "image_missing" in audit.issues or "image_unreadable" in audit.issues:
            continue
        valid.append(audit)
    splits = split_audits(valid, args.seed, args.train_ratio, args.val_ratio)
    manifest_rows = []
    stats = {"dataset_dir": str(dataset_dir), "cleaned": cleaned, "images": 0, "boxes": 0, "splits": {}, "class_counts": Counter(), "issue_counts": Counter()}
    seq = 0
    for split, rows in splits.items():
        split_stats = {"images": 0, "boxes": 0, "class_counts": Counter()}
        for audit in rows:
            seq += 1
            src = Path(audit.image_path)
            stem = safe_stem(audit, seq)
            dst_img = dataset_dir / "images" / split / f"{stem}.jpg"
            dst_lbl = dataset_dir / "labels" / split / f"{stem}.txt"
            write_clean_image(src, dst_img)
            lines = []
            for box in audit.boxes:
                if box.yolo is None:
                    continue
                cls, xc, yc, bw, bh = box.yolo
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                name = MERGED_CLASSES[cls]
                split_stats["class_counts"][name] += 1
                stats["class_counts"][name] += 1
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_stats["images"] += 1
            split_stats["boxes"] += len(lines)
            stats["images"] += 1
            stats["boxes"] += len(lines)
            for issue in audit.issues:
                stats["issue_counts"][issue] += 1
            manifest_rows.append({
                "split": split,
                "dataset_image": str(dst_img.relative_to(dataset_dir)),
                "dataset_label": str(dst_lbl.relative_to(dataset_dir)),
                "source_image": audit.image_path,
                "image_rel_path": audit.image_rel_path,
                "expected_label": audit.expected_label,
                "expected_merged_label": audit.expected_merged_label or "",
                "score": audit.score,
                "issues": ";".join(audit.issues),
                "boxes": len(lines),
            })
        stats["splits"][split] = {"images": split_stats["images"], "boxes": split_stats["boxes"], "class_counts": dict(split_stats["class_counts"])}
    (dataset_dir / "data.yaml").write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(MERGED_CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(MERGED_CLASSES)),
        encoding="utf-8",
    )
    write_csv(dataset_dir / "manifest.csv", manifest_rows)
    serializable = {**stats, "class_counts": dict(stats["class_counts"]), "issue_counts": dict(stats["issue_counts"])}
    (dataset_dir / "summary.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return serializable


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_to_csv_row(a: AuditRow) -> dict[str, Any]:
    return {
        "row_index": a.row_index,
        "image_rel_path": a.image_rel_path,
        "image_path": a.image_path,
        "expected_label": a.expected_label,
        "expected_merged_label": a.expected_merged_label or "",
        "score": a.score,
        "suspicious": bool(a.issues),
        "issues": ";".join(a.issues),
        "elements_count": a.elements_count,
        "valid_boxes_count": a.valid_boxes_count,
        "labels_raw": ";".join(a.labels_raw),
        "labels_merged": ";".join(a.labels_merged),
        "max_confidence": "" if a.max_confidence is None else f"{a.max_confidence:.4f}",
        "min_confidence": "" if a.min_confidence is None else f"{a.min_confidence:.4f}",
        "max_area_ratio": "" if a.max_area_ratio is None else f"{a.max_area_ratio:.6f}",
        "image_width": a.image_width or "",
        "image_height": a.image_height or "",
    }


def draw_audit(audit: AuditRow, out_path: Path, max_side: int = 1100) -> None:
    src = Path(audit.image_path)
    with Image.open(src) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        scale = min(1.0, max_side / max(ow, oh))
        if scale < 1:
            im = im.resize((round(ow * scale), round(oh * scale)), Image.Resampling.LANCZOS)
        w, h = im.size
        draw = ImageDraw.Draw(im)
        font = load_font(max(16, round(max(w, h) / 55)))
        line_w = max(3, round(max(w, h) / 220))
        for box in audit.boxes:
            if box.bbox_norm is None:
                continue
            ymin, xmin, ymax, xmax = box.bbox_norm
            xy = [xmin*w, ymin*h, xmax*w, ymax*h]
            color = COLORS.get(box.merged_label or "invalid", COLORS["invalid"])
            draw.rectangle(xy, outline=color, width=line_w)
            label = box.merged_label or box.source_label or "invalid"
            if box.confidence is not None:
                label += f" {box.confidence:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            th = bbox[3] - bbox[1]
            tw = bbox[2] - bbox[0]
            x, y = xy[0], max(0, xy[1] - th - 8)
            draw.rectangle([x, y, x + tw + 8, y + th + 8], fill=color)
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, quality=90)


def write_review_html(qa_dir: Path, cards: list[dict[str, Any]], title: str) -> None:
    qa_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cards:
        chips = "".join(f"<span>{html.escape(i)}</span>" for i in c["issues"])
        rows.append(
            "<div class='card'>"
            f"<img src='{html.escape(c['rel_viz'])}' loading='lazy'>"
            f"<div class='meta'><b>{html.escape(c['image_rel_path'])}</b><br>expected={html.escape(c['expected_label'])} -> {html.escape(c['expected_merged_label'])}<br>score={c['score']} boxes={c['boxes']}</div>"
            f"<div class='chips'>{chips}</div>"
            "</div>"
        )
    html_text = "".join([
        "<!doctype html><html><head><meta charset='utf-8'><title>", html.escape(title), "</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:20px;background:#f5f7fa;color:#18212f}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}.card{background:#fff;border:1px solid #d9dee8;border-radius:8px;padding:10px}.card img{width:100%;height:auto;border:1px solid #edf0f4}.meta{font-size:13px;line-height:1.45;margin-top:8px}.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}.chips span{font-size:12px;background:#eef2ff;border-radius:999px;padding:2px 7px}</style></head><body>",
        f"<h1>{html.escape(title)}</h1><div class='grid'>", "\n".join(rows), "</div></body></html>"
    ])
    (qa_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_visual_review(qa_dir: Path, audits: list[AuditRow], args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    suspicious = [a for a in audits if a.issues]
    clean = [a for a in audits if not a.issues and a.valid_boxes_count > 0]
    mismatch = [a for a in audits if "label_mismatch" in a.issues or "multi_class_conflict" in a.issues]
    groups = {"suspicious": suspicious, "clean_examples": clean, "label_conflicts": mismatch}
    summary = {}
    all_cards = []
    for name, rows in groups.items():
        rows = list(rows)
        rng.shuffle(rows)
        chosen = rows[:args.viz_per_group]
        cards = []
        for i, audit in enumerate(chosen, start=1):
            rel = Path("visualizations") / name / f"{i:03d}_{Path(audit.image_rel_path).stem}.jpg"
            draw_audit(audit, qa_dir / rel)
            card = {"rel_viz": str(rel), "image_rel_path": audit.image_rel_path, "expected_label": audit.expected_label, "expected_merged_label": audit.expected_merged_label or "", "score": audit.score, "issues": audit.issues or ["clean"], "boxes": audit.valid_boxes_count}
            cards.append(card)
            all_cards.append({**card, "group": name})
        write_review_html(qa_dir / name, [{**c, "rel_viz": "../" + c["rel_viz"]} for c in cards], f"{name} review")
        summary[name] = {"available": len(rows), "rendered": len(chosen), "html": str((qa_dir / name / "index.html").relative_to(qa_dir))}
    write_review_html(qa_dir, all_cards, "Gemini router annotation QA overview")
    return summary


def write_docs(qa_dir: Path, audit_summary: dict[str, Any], full_summary: dict[str, Any], cleaned_summary: dict[str, Any], args: argparse.Namespace) -> None:
    md = f"""# Gemini 三类路由标注检查与清洗报告

## 输入

- Gemini 结果：`{args.results}`
- 原始图片根目录：`{args.image_root}`
- 模型：`gemini-3.1-pro-preview`

## 目标类别

原始四类被合并为三类：

```text
天井 -> 天井
内壁 -> 壁类
RC壁 -> 壁类
RC柱 -> RC柱
```

## 量化评分规则

每张图初始分为 100 分。发现问题后按问题类型扣分，存在 issue 的样本会进入人工确认清单。`full` 数据集保留所有可转换合法框；`cleaned` 数据集排除关键错误样本，并要求分数达到 `{args.clean_score_threshold}`。非关键 issue（例如多类别共存、接近整图框、轻微标签不一致）仍会保留在 cleaned 中，但会在 manifest 和 QA 报告中标记，供人工复查。

主要扣分项包括：

- 解析失败、图片缺失、图片不可读。
- 空 elements 或没有合法 bbox。
- 非法标签、非法 bbox、越界 bbox。
- 低置信度，阈值 `{args.min_confidence}`。
- bbox 过小，面积阈值 `{args.min_area_ratio}`。
- bbox 过大，面积阈值 `{args.max_area_ratio}`。
- 细长框、多类别冲突、与原目录标签合并后不一致。

## 审计摘要

```json
{json.dumps(audit_summary, ensure_ascii=False, indent=2)}
```

## 数据集摘要

### Full

```json
{json.dumps(full_summary, ensure_ascii=False, indent=2)}
```

### Cleaned

```json
{json.dumps(cleaned_summary, ensure_ascii=False, indent=2)}
```

## 产物

- 审计表：`annotation_audit.csv`
- 可疑样本：`suspicious_samples.jsonl`
- bbox 统计：`bbox_stats.csv`
- 混淆矩阵：`label_confusion_summary.csv`
- QA 主页：`index.html`
- Full 数据集：`{args.full_dataset}`
- Cleaned 数据集：`{args.cleaned_dataset}`

## 人工确认建议

优先查看：

1. `suspicious/index.html`
2. `label_conflicts/index.html`
3. `annotation_audit.csv` 中 `score` 最低的样本

第一轮训练建议先使用 `full` 数据集建立 baseline，再使用 `cleaned` 数据集对比混淆矩阵和 recall。如果 `cleaned` 过于保守，应放宽只因 `almost_full_image_box` 或 `label_mismatch` 触发的过滤规则。
"""
    (qa_dir / "cleaning_report.md").write_text(md, encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_path = Path(args.results)
    image_root = Path(args.image_root)
    qa_dir = Path(args.qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with results_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                rows.append(audit_one(json.loads(line), i, image_root, args))

    audit_rows = [audit_to_csv_row(a) for a in rows]
    write_csv(qa_dir / "annotation_audit.csv", audit_rows)
    suspicious = [a for a in rows if a.issues]
    with (qa_dir / "suspicious_samples.jsonl").open("w", encoding="utf-8") as f:
        for a in suspicious:
            f.write(json.dumps({**audit_to_csv_row(a), "boxes": [b.__dict__ for b in a.boxes]}, ensure_ascii=False) + "\n")

    bbox_rows = []
    confusion = Counter()
    issue_counts = Counter()
    raw_label_counts = Counter()
    merged_label_counts = Counter()
    for a in rows:
        confusion[(a.expected_merged_label or "unknown", "+".join(a.labels_merged) or "none")] += 1
        for issue in a.issues:
            issue_counts[issue] += 1
        for label in a.labels_raw:
            raw_label_counts[label] += 1
        for label in a.labels_merged:
            merged_label_counts[label] += 1
        for b in a.boxes:
            if b.area_ratio is not None:
                bbox_rows.append({"image_rel_path": a.image_rel_path, "source_label": b.source_label, "merged_label": b.merged_label or "", "confidence": b.confidence if b.confidence is not None else "", "area_ratio": f"{b.area_ratio:.6f}", "aspect_ratio": "" if b.aspect_ratio is None else f"{b.aspect_ratio:.4f}", "issues": ";".join(b.issues)})
    write_csv(qa_dir / "bbox_stats.csv", bbox_rows)
    write_csv(qa_dir / "label_confusion_summary.csv", [{"expected_merged_label": k[0], "predicted_merged_labels": k[1], "count": v} for k, v in sorted(confusion.items())])

    full_summary = write_dataset(Path(args.full_dataset), rows, args, cleaned=False)
    cleaned_summary = write_dataset(Path(args.cleaned_dataset), rows, args, cleaned=True)
    viz_summary = build_visual_review(qa_dir, rows, args)

    audit_summary = {
        "results": str(results_path),
        "total_images": len(rows),
        "ok_images": sum(1 for a in rows if a.ok),
        "suspicious_images": len(suspicious),
        "clean_images": sum(1 for a in rows if a.valid_boxes_count > 0 and a.score >= args.clean_score_threshold and not (set(a.issues) & CRITICAL_ISSUES)),
        "issue_counts": dict(issue_counts),
        "raw_label_counts": dict(raw_label_counts),
        "merged_label_counts": dict(merged_label_counts),
        "score_threshold": args.clean_score_threshold,
        "visual_review": viz_summary,
    }
    (qa_dir / "annotation_audit.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_docs(qa_dir, audit_summary, full_summary, cleaned_summary, args)
    print(json.dumps({"audit": audit_summary, "full": full_summary, "cleaned": cleaned_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
