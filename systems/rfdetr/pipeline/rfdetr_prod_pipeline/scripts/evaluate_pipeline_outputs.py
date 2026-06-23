#!/usr/bin/env python3
"""Evaluate pipeline JSONL outputs against a pipeline split manifest."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2


GRADE_BY_ID = {0: "B", 1: "C", 2: "D"}
COMPONENTS = ["tenjo", "inner_wall", "rc_wall", "rc_column"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--split", required=True, help="split.json or manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument(
        "--exclude-list",
        default="",
        help="Optional CSV/line file of eval image names to exclude from metric aggregation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(Path(args.results))
    samples = load_samples(Path(args.split))
    excluded_names = load_excluded_names(Path(args.exclude_list)) if args.exclude_list else set()
    if excluded_names:
        samples = [sample for sample in samples if Path(sample["eval_image"]).name not in excluded_names]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "review"
    review_dir.mkdir(exist_ok=True)

    image_rows = []
    strict_counts = init_counts()
    loc_counts = init_counts()
    grade_confusion: Counter[tuple[str, str]] = Counter()
    latency = []
    for sample in samples:
        result = results.get(Path(sample["eval_image"]).name)
        row = evaluate_sample(sample, result, args.iou_threshold)
        image_rows.append(row)
        accumulate(strict_counts, row, "strict")
        accumulate(loc_counts, row, "loc")
        for pair in row["grade_pairs"]:
            grade_confusion[(pair["gt"], pair["pred"])] += 1
        if result and result.get("elapsed_ms") is not None:
            latency.append(float(result["elapsed_ms"]))

    write_csv(out_dir / "per_image_analysis.csv", flatten_image_rows(image_rows))
    write_csv(out_dir / "grade_confusion.csv", [
        {"gt": gt, "pred": pred, "count": count}
        for (gt, pred), count in sorted(grade_confusion.items())
    ])
    summary = {
        "images": len(samples),
        "excluded_images": sorted(excluded_names),
        "matched_results": sum(1 for row in image_rows if row["has_result"]),
        "iou_threshold": args.iou_threshold,
        "strict_grade_metrics": metrics_block(strict_counts),
        "localization_any_grade_metrics": metrics_block(loc_counts),
        "latency_ms": latency_block(latency),
        "router": router_block(samples, results),
        "warnings": dict(warning_counts(results.values())),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_review(image_rows, results, review_dir, args.review_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {Path(row["image"]).name: row for row in rows}


def load_samples(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))["samples"]
    return list(csv.DictReader(path.open(encoding="utf-8")))


def load_excluded_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        out = set()
        for row in rows:
            decision = str(row.get("decision", ""))
            if "exclude" not in decision:
                continue
            image = row.get("image") or row.get("eval_image_name") or row.get("eval_image")
            if image:
                out.add(Path(image).name)
        return out
    return {
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def evaluate_sample(sample: dict[str, str], result: dict[str, Any] | None, iou_threshold: float) -> dict[str, Any]:
    image = cv2.imread(sample["eval_image"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unreadable image: {sample['eval_image']}")
    gt = load_gt(Path(sample["eval_label"]), image.shape[1], image.shape[0])
    preds = []
    if result:
        preds = normalized_predictions(result.get("display_crack_detections") or result.get("crack_detections") or [])
    strict_matches = match(gt, preds, iou_threshold, require_grade=True)
    loc_matches = match(gt, preds, iou_threshold, require_grade=False)
    strict_gt = {m["gt_index"] for m in strict_matches}
    strict_pred = {m["pred_index"] for m in strict_matches}
    loc_gt = {m["gt_index"] for m in loc_matches}
    loc_pred = {m["pred_index"] for m in loc_matches}
    grade_pairs = [
        {"gt": gt[m["gt_index"]]["grade"], "pred": preds[m["pred_index"]]["grade"]}
        for m in loc_matches
    ]
    return {
        "sample": sample,
        "has_result": result is not None,
        "error": result.get("error", "") if result else "missing_result",
        "gt": gt,
        "preds": preds,
        "strict_tp": len(strict_matches),
        "strict_fp": max(0, len(preds) - len(strict_pred)),
        "strict_fn": max(0, len(gt) - len(strict_gt)),
        "loc_tp": len(loc_matches),
        "loc_fp": max(0, len(preds) - len(loc_pred)),
        "loc_fn": max(0, len(gt) - len(loc_gt)),
        "grade_pairs": grade_pairs,
        "router_classes": "|".join(
            d.get("class_name", "")
            for d in ((result or {}).get("router") or {}).get("detections", [])
        ),
        "warnings": "|".join((result or {}).get("warnings", [])),
        "elapsed_ms": (result or {}).get("elapsed_ms", ""),
    }


def load_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = [float(v) for v in parts]
        out.append({
            "grade": GRADE_BY_ID.get(int(cls), str(int(cls))),
            "bbox": [
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ],
        })
    return out


def normalized_predictions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        bbox = item.get("bbox_xyxy")
        if not bbox:
            continue
        out.append({
            "grade": normalize_grade(item.get("damage_grade", "")),
            "bbox": [float(v) for v in bbox],
            "confidence": float(item.get("confidence", 0.0)),
            "source_model": item.get("source_model", ""),
            "source_router_class": item.get("source_router_class", ""),
        })
    return out


def normalize_grade(value: Any) -> str:
    text = str(value)
    for grade in ["B", "C", "D"]:
        if grade in text:
            return grade
    return text


def match(gt: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float, require_grade: bool) -> list[dict[str, Any]]:
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            if require_grade and g["grade"] != p["grade"]:
                continue
            score = iou(g["bbox"], p["bbox"])
            if score >= iou_threshold:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append({"iou": score, "gt_index": gi, "pred_index": pi})
    return matches


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def init_counts() -> dict[str, Counter[str]]:
    return {"overall": Counter(), **{component: Counter() for component in COMPONENTS}}


def accumulate(counts: dict[str, Counter[str]], row: dict[str, Any], prefix: str) -> None:
    component = row["sample"]["class_key"]
    for key in ["tp", "fp", "fn"]:
        value = int(row[f"{prefix}_{key}"])
        counts["overall"][key] += value
        counts[component][key] += value


def metrics_block(counts: dict[str, Counter[str]]) -> dict[str, Any]:
    return {key: metrics(value) for key, value in counts.items()}


def metrics(c: Counter[str]) -> dict[str, Any]:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def latency_block(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": round(mean(values), 3),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": round(max(values), 3),
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return round(ordered[index], 3)


def router_block(samples: list[dict[str, str]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {"tenjo": "天井", "inner_wall": "壁类", "rc_wall": "壁类", "rc_column": "RC柱"}
    out = {component: Counter() for component in COMPONENTS}
    for sample in samples:
        component = sample["class_key"]
        result = results.get(Path(sample["eval_image"]).name)
        classes = [d.get("class_name") for d in ((result or {}).get("router") or {}).get("detections", [])]
        out[component]["images"] += 1
        if expected[component] in classes:
            out[component]["any_candidate_hit"] += 1
        if classes and classes[0] == expected[component]:
            out[component]["top1_hit"] += 1
    return {
        component: {
            "images": c["images"],
            "top1_hit_rate": round(c["top1_hit"] / c["images"], 4) if c["images"] else 0.0,
            "any_candidate_hit_rate": round(c["any_candidate_hit"] / c["images"], 4) if c["images"] else 0.0,
        }
        for component, c in out.items()
    }


def warning_counts(rows: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("warnings", []))
    return counts


def flatten_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        sample = row["sample"]
        out.append({
            "image": sample["eval_image"],
            "component": sample["class_key"],
            "eval_group": sample.get("eval_group", ""),
            "gt_boxes": len(row["gt"]),
            "pred_boxes": len(row["preds"]),
            "strict_tp": row["strict_tp"],
            "strict_fp": row["strict_fp"],
            "strict_fn": row["strict_fn"],
            "loc_tp": row["loc_tp"],
            "loc_fp": row["loc_fp"],
            "loc_fn": row["loc_fn"],
            "router_classes": row["router_classes"],
            "warnings": row["warnings"],
            "error": row["error"],
            "elapsed_ms": row["elapsed_ms"],
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_review(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], out_dir: Path, limit: int) -> None:
    ranked = sorted(
        rows,
        key=lambda r: (
            -(r["strict_fn"] + r["strict_fp"]),
            -r["loc_tp"],
            r["sample"]["class_key"],
            Path(r["sample"]["eval_image"]).name,
        ),
    )[:limit]
    html_rows = []
    for index, row in enumerate(ranked):
        image_path = Path(row["sample"]["eval_image"])
        canvas = draw_review_image(image_path, row["gt"], row["preds"])
        out_name = f"{index:03d}_{image_path.name}"
        cv2.imwrite(str(out_dir / out_name), canvas)
        html_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(row['sample']['class_key'])}</td>"
            f"<td>TP/FP/FN {row['strict_tp']}/{row['strict_fp']}/{row['strict_fn']}</td>"
            f"<td>loc {row['loc_tp']}/{row['loc_fp']}/{row['loc_fn']}</td>"
            f"<td>{html.escape(row['router_classes'])}</td>"
            f"<td><img src='{html.escape(out_name)}'></td>"
            "</tr>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif} img{max-width:720px} td{vertical-align:top;padding:6px;border-top:1px solid #ddd}</style>"
        "</head><body><h1>Pipeline Review</h1><table>"
        + "\n".join(html_rows)
        + "</table></body></html>"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def draw_review_image(path: Path, gt: list[dict[str, Any]], preds: list[dict[str, Any]]) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unreadable image: {path}")
    for item in gt:
        draw_box(image, item["bbox"], (40, 180, 40), f"GT-{item['grade']}", 3)
    for item in preds:
        label = f"P-{item['grade']} {item['confidence']:.2f}"
        draw_box(image, item["bbox"], (40, 70, 230), label, 2)
    return image


def draw_box(image: Any, box: list[float], color: tuple[int, int, int], label: str, width: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width)
    y = max(16, y1 - 5)
    cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
