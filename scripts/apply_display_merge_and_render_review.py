#!/usr/bin/env python3
"""Apply final display suppression to saved pipeline results and render review images."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from rfdetr_prod_pipeline.pipeline.display_merge import suppress_overlapping_display_detections
from rfdetr_prod_pipeline.pipeline.result_merge import grade_level, iou_xyxy
from rfdetr_prod_pipeline.pipeline.wall_candidate_display import build_wall_candidate_display


CLASS_ID_TO_GRADE = {"0": "B", "1": "C", "2": "D"}
ROUTER_LABELS = {"天井": "TENJO", "壁类": "WALL", "壁類": "WALL", "RC柱": "RCCOL"}
SOURCE_LABELS = {
    "ceiling": "CEIL",
    "inner_wall": "WALL",
    "rc_wall": "WALL",
    "rc_column": "RCCOL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline")
    parser.add_argument("--split-json", default="data/pipeline_eval_official_plus_20260615/split.json")
    parser.add_argument("--output-dir", default="outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1")
    parser.add_argument("--priority-count", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "human_review_static_large"
    cases_dir = review_dir / "cases"
    priority_dir = review_dir / "priority"
    if review_dir.exists():
        shutil.rmtree(review_dir)
    cases_dir.mkdir(parents=True)
    priority_dir.mkdir(parents=True)

    sample_by_image = load_samples(Path(args.split_json))
    original_rows = load_jsonl(baseline_dir / "results.jsonl")
    old_review_order = load_review_order(baseline_dir / "human_review" / "review_cases.csv")

    updated_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for row in original_rows:
        wall_candidate_display = build_wall_candidate_display(row.get("raw_crack_detections") or [])
        before = rebuild_display_items(row, wall_candidate_display["display_detections"])
        after, suppressed = suppress_overlapping_display_detections(before)
        updated = dict(row)
        updated["wall_candidate_display"] = wall_candidate_display
        updated["display_crack_detections_before_display_merge"] = before
        updated["display_crack_detections"] = after
        updated["suppressed_display_crack_detections"] = suppressed
        updated["display_merge_summary"] = {
            "before": len(before),
            "after": len(after),
            "suppressed": len(suppressed),
        }
        updated_rows.append(updated)

        sample = sample_by_image.get(Path(row["image"]).name, {})
        gt = load_gt(sample.get("eval_label"), row.get("image_shape") or [])
        metrics = evaluate_display(gt, after)
        review_rows.append(build_review_row(updated, sample, metrics))

    write_jsonl(output_dir / "results.jsonl", updated_rows)
    write_csv(output_dir / "review_cases.csv", sorted_review_rows(review_rows, old_review_order))
    summary = summarize(original_rows, updated_rows, review_rows)
    (output_dir / "display_merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered_rows = sorted_review_rows(review_rows, old_review_order)
    row_by_image = {row["image"]: row for row in updated_rows}
    html_sections = []
    for index, review in enumerate(ordered_rows, start=1):
        result = row_by_image[review["image"]]
        sample = sample_by_image.get(Path(review["image"]).name, {})
        gt = load_gt(sample.get("eval_label"), result.get("image_shape") or [])
        out_name = safe_case_name(index, review)
        out_path = cases_dir / out_name
        render_case(result, gt, review, out_path, index)
        if index <= args.priority_count:
            shutil.copy2(out_path, priority_dir / out_name)
        rel = f"cases/{out_name}"
        html_sections.append(
            "<section>"
            f"<h2>Case {index:03d}: {html.escape(review['component'])} {html.escape(review['status'])} "
            f"TP/FP/FN={review['display_tp']}/{review['display_fp']}/{review['display_fn']}</h2>"
            f"<p>{html.escape(Path(review['image']).name)} | before={review['display_before']} "
            f"after={review['display_after']} suppressed={review['display_suppressed']} | "
            f"router={html.escape(review['router_primary_ascii'])} expected={html.escape(review['router_expected_ascii'])} | "
            f"{review['elapsed_ms']} ms</p>"
            f"<a href='{html.escape(rel)}' target='_blank'><img src='{html.escape(rel)}'></a>"
            "</section>"
        )
        if index % 40 == 0:
            print(f"rendered {index}/{len(ordered_rows)}")

    write_index(review_dir / "large_index.html", html_sections, summary)
    (review_dir / "large_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(output_dir / "review_cases.csv", review_dir / "large_review_cases.csv")
    print(json.dumps({"output_dir": str(output_dir), "review_index": str(review_dir / "large_index.html"), **summary}, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {Path(sample["eval_image"]).name: sample for sample in data["samples"]}


def load_review_order(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {Path(row["image"]).name: index for index, row in enumerate(csv.DictReader(f))}


def rebuild_display_items(row: dict[str, Any], wall_display: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_wall = [
        det
        for det in row.get("display_crack_detections", [])
        if str(det.get("source_router_class") or "") not in {"壁类", "壁類"}
    ]
    return non_wall + wall_display


def load_gt(label_path: str | None, image_shape: list[int]) -> list[dict[str, Any]]:
    if not label_path or not image_shape:
        return []
    path = Path(label_path)
    if not path.exists():
        return []
    height, width = int(image_shape[0]), int(image_shape[1])
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[:5]
        cx_f, cy_f, bw_f, bh_f = map(float, (cx, cy, bw, bh))
        x1 = (cx_f - bw_f / 2.0) * width
        y1 = (cy_f - bh_f / 2.0) * height
        x2 = (cx_f + bw_f / 2.0) * width
        y2 = (cy_f + bh_f / 2.0) * height
        out.append({"grade": CLASS_ID_TO_GRADE.get(cls, cls), "bbox_xyxy": [x1, y1, x2, y2]})
    return out


def evaluate_display(gt: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float = 0.50) -> dict[str, int]:
    used_preds: set[int] = set()
    tp = 0
    for item in gt:
        best_index = None
        best_iou = 0.0
        for index, pred in enumerate(preds):
            if index in used_preds:
                continue
            if grade_level(str(pred.get("damage_grade", ""))) != item["grade"]:
                continue
            iou = iou_xyxy(tuple(item["bbox_xyxy"]), tuple(float(v) for v in pred.get("bbox_xyxy", [0, 0, 0, 0])))
            if iou >= iou_threshold and iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index is not None:
            used_preds.add(best_index)
            tp += 1
    fp = max(0, len(preds) - len(used_preds))
    fn = max(0, len(gt) - tp)
    return {"tp": tp, "fp": fp, "fn": fn, "gt": len(gt), "pred": len(preds)}


def build_review_row(result: dict[str, Any], sample: dict[str, Any], metrics: dict[str, int]) -> dict[str, Any]:
    router = result.get("router") or {}
    route = router.get("route_decision") or {}
    primary = str(route.get("primary_class") or "")
    expected = expected_router(sample.get("component", ""))
    suppressed = len(result.get("suppressed_display_crack_detections", []))
    status = "GOOD"
    if metrics["fn"]:
        status = "FN"
    elif metrics["fp"] >= 4:
        status = "FP_many"
    elif primary and expected and primary != expected:
        status = "Router_primary_miss"
    elif suppressed:
        status = "DEDUPED"
    score = metrics["fn"] * 20 + metrics["fp"] * 3 + (10 if primary and expected and primary != expected else 0)
    return {
        "image": result["image"],
        "component": sample.get("component", ""),
        "subset": sample.get("subset", ""),
        "status": status,
        "display_tp": metrics["tp"],
        "display_fp": metrics["fp"],
        "display_fn": metrics["fn"],
        "display_gt": metrics["gt"],
        "display_before": result["display_merge_summary"]["before"],
        "display_after": result["display_merge_summary"]["after"],
        "display_suppressed": suppressed,
        "raw_pred": len(result.get("crack_detections", [])),
        "router_primary": primary,
        "router_expected": expected,
        "router_primary_ascii": ascii_router(primary),
        "router_expected_ascii": ascii_router(expected),
        "router_detections": len(router.get("detections") or []),
        "elapsed_ms": round(float(result.get("elapsed_ms") or 0.0), 3),
        "warnings": ";".join(result.get("warnings") or []),
        "ux_priority_score": score,
    }


def expected_router(component: str) -> str:
    return {"tenjo": "天井", "inner_wall": "壁类", "rc_wall": "壁类", "rc_column": "RC柱"}.get(component, "")


def sorted_review_rows(rows: list[dict[str, Any]], old_order: dict[str, int]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (old_order.get(Path(row["image"]).name, 10**9), -float(row["ux_priority_score"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(original_rows: list[dict[str, Any]], updated_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = sum(len(row.get("display_crack_detections", [])) for row in original_rows)
    after = sum(len(row.get("display_crack_detections", [])) for row in updated_rows)
    suppressed = sum(len(row.get("suppressed_display_crack_detections", [])) for row in updated_rows)
    return {
        "cases": len(updated_rows),
        "display_before": before,
        "display_after": after,
        "display_suppressed": suppressed,
        "images_with_suppression": sum(1 for row in updated_rows if row.get("suppressed_display_crack_detections")),
        "status_counts": dict(Counter(row["status"] for row in review_rows)),
    }


def render_case(result: dict[str, Any], gt: list[dict[str, Any]], review: dict[str, Any], out_path: Path, case_index: int) -> None:
    image = Image.open(result["image"]).convert("RGB")
    panels = [
        ("1. GT LABELS", gt, "gt"),
        ("2. ROUTER REGIONS", result.get("router", {}).get("detections") or [], "router"),
        ("3. PC DISPLAY FINAL (DEDUPED)", result.get("display_crack_detections") or [], "display"),
        ("4. INTERNAL PRE-DISPLAY", result.get("crack_detections") or [], "internal"),
    ]
    panel_w, panel_h = 1160, 780
    title_h, top_h = 74, 58
    canvas = Image.new("RGB", (panel_w * 2, top_h + (title_h + panel_h) * 2), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    font_big = load_font(30, bold=True)
    font = load_font(22)
    font_small = load_font(18)
    header = (
        f"CASE {case_index:03d} | {Path(result['image']).name} | {review['component']} {review['subset']} | "
        f"status={review['status']} | TP/FP/FN={review['display_tp']}/{review['display_fp']}/{review['display_fn']} | "
        f"display {review['display_before']}->{review['display_after']} (-{review['display_suppressed']}) | "
        f"router={review['router_primary_ascii']} exp={review['router_expected_ascii']} | {review['elapsed_ms']}ms"
    )
    draw.rectangle([0, 0, canvas.width, top_h], fill=(31, 36, 48))
    draw.text((14, 14), header, fill=(255, 255, 255), font=font)
    for idx, (title, dets, mode) in enumerate(panels):
        col = idx % 2
        row = idx // 2
        x = col * panel_w
        y = top_h + row * (title_h + panel_h)
        draw.rectangle([x, y, x + panel_w, y + title_h], fill=(31, 36, 48))
        draw.text((x + 14, y + 13), title, fill=(255, 255, 255), font=font_big)
        draw.text((x + 14, y + 47), header[:140], fill=(235, 238, 245), font=font_small)
        panel_img = render_panel(image, dets, mode, panel_w, panel_h)
        canvas.paste(panel_img, (x, y + title_h))
    canvas.save(out_path, quality=92)


def render_panel(image: Image.Image, dets: list[dict[str, Any]], mode: str, panel_w: int, panel_h: int) -> Image.Image:
    scale = min(panel_w / image.width, panel_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_w, panel_h), (238, 241, 245))
    ox = (panel_w - new_size[0]) // 2
    oy = (panel_h - new_size[1]) // 2
    panel.paste(resized, (ox, oy))
    draw = ImageDraw.Draw(panel)
    font = load_font(18)
    for det in dets:
        box = det.get("bbox_xyxy") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        sx1, sy1 = ox + x1 * scale, oy + y1 * scale
        sx2, sy2 = ox + x2 * scale, oy + y2 * scale
        color = color_for(mode)
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=4)
        label = label_for(det, mode)
        tw = draw.textbbox((0, 0), label, font=font)[2]
        ly = max(0, sy1 - 22)
        draw.rectangle([sx1, ly, sx1 + tw + 8, ly + 22], fill=(255, 255, 255), outline=color, width=1)
        draw.text((sx1 + 4, ly + 1), label, fill=color, font=font)
    return panel


def label_for(det: dict[str, Any], mode: str) -> str:
    if mode == "gt":
        return f"GT-{det.get('grade', '')}"
    if mode == "router":
        return f"R-{ascii_router(det.get('class_name'))} {float(det.get('confidence') or 0):.2f}"
    grade = grade_level(str(det.get("damage_grade", "")))
    if mode == "display":
        family = display_family(det)
        prefix = {"wall": "WALL", "ceiling": "CEIL", "rc_column": "RCCOL"}.get(family, family.upper()[:8])
        suppressed = det.get("display_suppressed_count")
        suffix = f" m{suppressed}" if suppressed else ""
        return f"{prefix}-{grade} {float(det.get('confidence') or 0):.2f}{suffix}"
    source = SOURCE_LABELS.get(str(det.get("source_model") or ""), str(det.get("source_model") or "DET").upper()[:8])
    return f"{source}-{grade} {float(det.get('confidence') or 0):.2f}"


def display_family(det: dict[str, Any]) -> str:
    structure = str(det.get("structure_type") or "")
    if structure in {"壁類", "壁类"}:
        return "wall"
    source = str(det.get("source_model") or "")
    if source in {"inner_wall", "rc_wall", "wall_merged"}:
        return "wall"
    return source


def color_for(mode: str) -> tuple[int, int, int]:
    return {
        "gt": (0, 160, 80),
        "router": (245, 166, 35),
        "display": (20, 105, 210),
        "internal": (210, 55, 50),
    }[mode]


def ascii_router(value: Any) -> str:
    return ROUTER_LABELS.get(str(value or ""), str(value or "NONE"))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def safe_case_name(index: int, review: dict[str, Any]) -> str:
    stem = Path(review["image"]).stem.replace("/", "_")
    status = str(review["status"]).replace("/", "_")
    return f"case_{index:03d}_{review['component']}_{status}_{stem}.jpg"


def write_index(path: Path, sections: list[str], summary: dict[str, Any]) -> None:
    content = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Display Merge Review</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;background:#f5f6f8}"
        "section{background:white;border:1px solid #ddd;margin:0 0 28px;padding:14px}"
        "img{width:100%;max-width:2320px;height:auto;display:block}h1,h2{margin:0 0 8px}"
        "p{margin:4px 0 12px;color:#444}</style></head><body>"
        "<h1>Display Merge Review</h1>"
        f"<p>Cases={summary['cases']} | display boxes {summary['display_before']} -> {summary['display_after']} "
        f"| suppressed={summary['display_suppressed']} | images_with_suppression={summary['images_with_suppression']}</p>"
        + "".join(sections)
        + "</body></html>"
    )
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
