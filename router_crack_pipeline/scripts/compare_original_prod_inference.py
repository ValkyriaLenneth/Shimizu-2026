#!/usr/bin/env python3
"""Run original prod full-image inference on E2E samples and compare visually."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]
PROD_ROOT = Path("/workspace/Shimizu-VLM-Crack-Detection-Prod")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))

from api.inference import InferenceEngine  # noqa: E402
from router_crack_pipeline.pipeline.result_merge import iou_xyxy  # noqa: E402


CLASS_TO_PROD_TYPE = {
    "tenjo": "天井",
    "inner_wall": "内壁",
    "rc_wall": "耐震壁",
    "rc_column": "RC柱",
}

MODEL_PATHS = {
    "天井": REPO / "downloads/previous_phase_gpl_model_unpacked/infer_models/TIANJING.pt",
    "内壁": REPO / "downloads/previous_phase_gpl_model_unpacked/infer_models/NEIBI.pt",
    "耐震壁": REPO / "downloads/previous_phase_gpl_model_unpacked/infer_models/RCBI.pt",
    "RC柱": REPO / "downloads/previous_phase_gpl_model_unpacked/infer_models/RCZHU.pt",
}

COLORS = {
    "gt": (255, 255, 255),
    "router": (0, 215, 255),
    "current_main": (0, 255, 0),
    "current_secondary": (255, 128, 0),
    "original": (40, 80, 255),
    "text_bg": (20, 20, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--sample-filter", choices=["all", "current_issues"], default="all")
    parser.add_argument("--compare-filter", choices=["all", "any_issue"], default="any_issue")
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--thumb-width", type=int, default=520)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    e2e_dir = resolve(args.e2e_dir)
    out_dir = resolve(args.output_dir) if args.output_dir else e2e_dir / "original_prod_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = list(csv.DictReader((e2e_dir / "eval_by_image.csv").open(encoding="utf-8")))
    current_results = load_jsonl(e2e_dir / "results.jsonl")
    current_by_image = {str(Path(r["image"]).resolve()): r for r in current_results}
    if args.sample_filter == "current_issues":
        eval_rows = [r for r in eval_rows if has_current_issue(r)]

    engine = InferenceEngine(
        {k: str(v) for k, v in MODEL_PATHS.items()},
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        line_thickness=3,
    )

    original_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    contact_items: list[tuple[np.ndarray, str]] = []

    for idx, row in enumerate(eval_rows, start=1):
        image_path = (REPO / row["image"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        prod_type = CLASS_TO_PROD_TYPE[row["class_key"]]
        image_b64 = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
        _, detections = engine.run_inference([prod_type], image_b64, postprocess=True)
        current_result = current_by_image[str(image_path)]
        label_path = resolve(current_result["eval_meta"]["label"])
        gt = load_gt(label_path, image.shape[1], image.shape[0])
        original_eval = evaluate_original(row, gt, detections, args.iou_threshold)
        eval_records.append(original_eval)
        original_records.append(
            {
                "image": str(image_path),
                "eval_meta": current_result["eval_meta"],
                "prod_requested_types": [prod_type],
                "detections": detections,
                "original_eval": original_eval,
            }
        )

        any_issue = has_current_issue(row) or has_original_issue(original_eval)
        if args.compare_filter == "any_issue" and not any_issue:
            continue
        current_panel = draw_current_panel(image, row, current_result, gt, args.max_side)
        original_panel = draw_original_panel(image, row, detections, gt, original_eval, args.max_side)
        comparison = hconcat_with_titles(current_panel, original_panel, "Current router E2E", "Original prod full-image")
        stem = safe_stem(f"{idx:02d}_{row['class_key']}_{image_path.stem}")
        out_path = vis_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), comparison)
        compare_rows.append(
            {
                "visualization": str(out_path.relative_to(REPO)),
                "image": row["image"],
                "class_key": row["class_key"],
                "prod_type": prod_type,
                "current_router_classes": row["router_classes"],
                "current_match": row["main_matches_iou50"],
                "current_fn": row["main_false_negative"],
                "current_fp": row["main_false_positive"],
                "original_match": original_eval["original_matches_iou50"],
                "original_fn": original_eval["original_false_negative"],
                "original_fp": original_eval["original_false_positive"],
                "original_grade_ok": original_eval["original_grade_ok"],
                "original_grade_mismatch": original_eval["original_grade_mismatch"],
            }
        )
        contact_items.append((make_thumb(comparison, args.thumb_width), f"{idx:02d} {row['class_key']} cur {row['main_matches_iou50']}/{row['gt_boxes']} orig {original_eval['original_matches_iou50']}/{original_eval['gt_boxes']}"))

    write_jsonl(out_dir / "original_results.jsonl", original_records)
    write_csv(out_dir / "original_eval_by_image.csv", eval_records)
    write_csv(out_dir / "comparison_index.csv", compare_rows)
    summary = summarize(eval_records, eval_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if contact_items:
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), make_contact_sheet(contact_items))
    print(json.dumps({**summary, "output_dir": str(out_dir), "visualizations": len(compare_rows)}, ensure_ascii=False, indent=2))
    return 0


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_current_issue(row: dict[str, str]) -> bool:
    return any(int(row[k]) > 0 for k in ["fn_router_miss", "fn_no_main_output", "fn_iou_miss", "main_false_positive", "matched_grade_mismatch"])


def has_original_issue(row: dict[str, Any]) -> bool:
    return int(row["original_false_negative"]) > 0 or int(row["original_false_positive"]) > 0 or int(row["original_grade_mismatch"]) > 0


def load_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    gt = []
    for line in path.read_text(encoding="utf-8").splitlines():
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


def evaluate_original(row: dict[str, str], gt: list[dict[str, Any]], detections: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    matches = match_predictions(gt, detections, iou_threshold)
    grade_ok = sum(1 for m in matches if m["gt_grade"] == m["pred_grade"])
    return {
        "image": row["image"],
        "class_key": row["class_key"],
        "prod_type": CLASS_TO_PROD_TYPE[row["class_key"]],
        "gt_boxes": len(gt),
        "original_predictions": len(detections),
        "original_matches_iou50": len(matches),
        "original_false_negative": max(0, len(gt) - len(matches)),
        "original_false_positive": max(0, len(detections) - len(matches)),
        "original_grade_ok": grade_ok,
        "original_grade_mismatch": max(0, len(matches) - grade_ok),
    }


def match_predictions(gt: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    candidates = []
    for gi, item in enumerate(gt):
        for pi, pred in enumerate(preds):
            score = iou_xyxy(tuple(item["bbox_xyxy"]), tuple(float(v) for v in pred["bbox"]))
            if score >= iou_threshold:
                candidates.append((score, gi, pi, item, pred))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt = set()
    used_pred = set()
    matches = []
    for score, gi, pi, item, pred in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append({"iou": score, "gt_grade": item["grade"], "pred_grade": str(pred.get("level", ""))})
    return matches


def draw_current_panel(image: np.ndarray, row: dict[str, str], result: dict[str, Any], gt: list[dict[str, Any]], max_side: int) -> np.ndarray:
    canvas, scale = resized(image, max_side)
    draw_header(canvas, ["CURRENT router E2E", f"{row['class_key']} router={row['router_classes']}", f"gt={row['gt_boxes']} main={row['main_predictions']} sec={row['secondary_predictions']} match={row['main_matches_iou50']} fn={row['main_false_negative']} fp={row['main_false_positive']}"])
    for i, det in enumerate((result.get("router") or {}).get("detections", [])):
        draw_box(canvas, scale_box(det["bbox_xyxy"], scale), COLORS["router"], f"R{i} {ascii_type(det.get('class_name'))} {float(det.get('confidence', 0)):.2f}", 2)
    for i, item in enumerate(gt):
        draw_box(canvas, scale_box(item["bbox_xyxy"], scale), COLORS["gt"], f"GT{i}:{item['grade']}", 3)
    main_model = row["main_model"]
    for i, det in enumerate(result.get("raw_crack_detections", [])):
        is_main = det.get("source_model") == main_model
        color = COLORS["current_main"] if is_main else COLORS["current_secondary"]
        draw_box(canvas, scale_box(det["bbox_xyxy"], scale), color, f"{'M' if is_main else 'S'}{i}:{det.get('source_model')} {grade_from_text(det.get('damage_grade', ''))} {float(det.get('confidence', 0)):.2f}", 2)
    return canvas


def draw_original_panel(image: np.ndarray, row: dict[str, str], detections: list[dict[str, Any]], gt: list[dict[str, Any]], eval_row: dict[str, Any], max_side: int) -> np.ndarray:
    canvas, scale = resized(image, max_side)
    draw_header(canvas, ["ORIGINAL prod full-image", f"requested={ascii_type(CLASS_TO_PROD_TYPE[row['class_key']])}", f"gt={eval_row['gt_boxes']} pred={eval_row['original_predictions']} match={eval_row['original_matches_iou50']} fn={eval_row['original_false_negative']} fp={eval_row['original_false_positive']} grade_mis={eval_row['original_grade_mismatch']}"])
    for i, item in enumerate(gt):
        draw_box(canvas, scale_box(item["bbox_xyxy"], scale), COLORS["gt"], f"GT{i}:{item['grade']}", 3)
    for i, det in enumerate(detections):
        draw_box(canvas, scale_box(det["bbox"], scale), COLORS["original"], f"O{i}:{ascii_type(det.get('type'))} {det.get('level')} {float(det.get('confidence', 0)):.2f}", 2)
    return canvas


def resized(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale >= 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA), scale


def scale_box(values: list[float], scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = values
    return round(x1 * scale), round(y1 * scale), round(x2 * scale), round(y2 * scale)


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
    line_h = 24
    h = line_h * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line[:180], (8, 24 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def hconcat_with_titles(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    left = pad_to_height(left, h)
    right = pad_to_height(right, h)
    divider = np.full((h, 8, 3), 30, dtype=np.uint8)
    combined = cv2.hconcat([left, divider, right])
    bar = np.full((34, combined.shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(bar, left_title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, right_title, (left.shape[1] + 24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([bar, combined])


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    return cv2.copyMakeBorder(img, 0, height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))


def grade_from_text(text: str) -> str:
    match = re.search(r"程度([BCD])", text)
    return match.group(1) if match else "?"


def ascii_type(value: Any) -> str:
    return {"天井": "ceiling", "内壁": "inner_wall", "耐震壁": "rc_wall", "RC柱": "rc_column", "壁类": "wall"}.get(str(value), str(value))


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def make_thumb(img: np.ndarray, width: int) -> np.ndarray:
    scale = width / img.shape[1]
    return cv2.resize(img, (width, max(1, round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def make_contact_sheet(items: list[tuple[np.ndarray, str]]) -> np.ndarray:
    cols = 2
    pad = 12
    label_h = 30
    thumb_w = max(img.shape[1] for img, _ in items)
    thumb_h = max(img.shape[0] for img, _ in items)
    rows = math.ceil(len(items) / cols)
    sheet = np.full((rows * (thumb_h + label_h + pad) + pad, cols * (thumb_w + pad) + pad, 3), 245, dtype=np.uint8)
    for idx, (img, label) in enumerate(items):
        r, c = divmod(idx, cols)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + label_h + pad)
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
        cv2.putText(sheet, label[:90], (x, y + thumb_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return sheet


def summarize(original_rows: list[dict[str, Any]], current_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_class = {}
    for class_key in ["tenjo", "inner_wall", "rc_wall", "rc_column"]:
        originals = [r for r in original_rows if r["class_key"] == class_key]
        currents = [r for r in current_rows if r["class_key"] == class_key]
        by_class[class_key] = {
            "images": len(originals),
            "gt_boxes": sum(int(r["gt_boxes"]) for r in originals),
            "current_matches_iou50": sum(int(r["main_matches_iou50"]) for r in currents),
            "current_fn": sum(int(r["main_false_negative"]) for r in currents),
            "current_fp": sum(int(r["main_false_positive"]) for r in currents),
            "original_predictions": sum(int(r["original_predictions"]) for r in originals),
            "original_matches_iou50": sum(int(r["original_matches_iou50"]) for r in originals),
            "original_fn": sum(int(r["original_false_negative"]) for r in originals),
            "original_fp": sum(int(r["original_false_positive"]) for r in originals),
            "original_grade_ok": sum(int(r["original_grade_ok"]) for r in originals),
            "original_grade_mismatch": sum(int(r["original_grade_mismatch"]) for r in originals),
        }
    return {
        "images": len(original_rows),
        "mode": "original_prod_user_selected_type_full_image",
        "prod_params": {"imgsz": 960, "conf": 0.01, "iou": 0.45, "postprocess": True},
        "by_class": by_class,
        "all": {
            "gt_boxes": sum(v["gt_boxes"] for v in by_class.values()),
            "current_matches_iou50": sum(v["current_matches_iou50"] for v in by_class.values()),
            "current_fn": sum(v["current_fn"] for v in by_class.values()),
            "current_fp": sum(v["current_fp"] for v in by_class.values()),
            "original_predictions": sum(v["original_predictions"] for v in by_class.values()),
            "original_matches_iou50": sum(v["original_matches_iou50"] for v in by_class.values()),
            "original_fn": sum(v["original_fn"] for v in by_class.values()),
            "original_fp": sum(v["original_fp"] for v in by_class.values()),
            "original_grade_ok": sum(v["original_grade_ok"] for v in by_class.values()),
            "original_grade_mismatch": sum(v["original_grade_mismatch"] for v in by_class.values()),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
