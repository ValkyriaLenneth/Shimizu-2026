#!/usr/bin/env python3
"""Run a tiny E2E evaluation sample and expose branch-statistics issues."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from router_crack_pipeline.pipeline.crack_detector_registry import build_detector_registry
from router_crack_pipeline.pipeline.result_merge import iou_xyxy
from router_crack_pipeline.pipeline.router_infer import RouterConfig, RouterInfer
from router_crack_pipeline.pipeline.run_full_pipeline import load_config, resolve_path, run_one_safe


CLASS_TO_ROUTER = {
    "tenjo": "天井",
    "inner_wall": "壁类",
    "rc_wall": "壁类",
    "rc_column": "RC柱",
}

CLASS_TO_MAIN_MODEL = {
    "tenjo": "ceiling",
    "inner_wall": "inner_wall",
    "rc_wall": "rc_wall",
    "rc_column": "rc_column",
}

GRADE_RANK = {"B": 1, "C": 2, "D": 3}


@dataclass
class GroundTruth:
    xyxy: tuple[float, float, float, float]
    grade: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/final_crack_yolo_20260519/manifest.csv")
    parser.add_argument("--config", default="router_crack_pipeline/configs/pipeline.previous_phase_gpl.local.yaml")
    parser.add_argument("--output-dir", default="outputs/e2e_debug_sample_20260519")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--ioa-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.output_dir, REPO)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sample_manifest(resolve_path(args.manifest, REPO), args.per_class, args.seed)
    write_csv(out_dir / "sampled_manifest.csv", rows)

    config_path = resolve_path(args.config, REPO)
    config = load_config(config_path)
    router = build_router(config, config_path, args.device)
    registry = build_detector_registry({**config, "device": args.device}, config_path.parent, mock=False)

    result_rows = []
    image_rows = []
    wall_pairs = []
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            image_path = resolve_path(row["image"], REPO)
            result = run_one_safe(image_path, router, registry, config)
            result["eval_meta"] = {
                "class_key": row["class_key"],
                "source": row["source"],
                "label": row["label"],
                "expected_router_class": CLASS_TO_ROUTER[row["class_key"]],
                "main_model": CLASS_TO_MAIN_MODEL[row["class_key"]],
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            result_rows.append(result)
            image_rows.append(evaluate_one(row, result, args.iou_threshold))
            wall_pairs.extend(find_wall_grade_shift_pairs(row, result, args.iou_threshold, args.ioa_threshold))

    write_csv(out_dir / "eval_by_image.csv", image_rows)
    write_csv(out_dir / "wall_grade_shift_pairs.csv", wall_pairs)
    summary = summarize(image_rows, wall_pairs, args.per_class)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_router(config: dict[str, Any], config_path: Path, device: str) -> RouterInfer:
    pipeline_cfg = config["pipeline"]
    return RouterInfer(
        RouterConfig(
            weights=resolve_path(pipeline_cfg["router_weights"], config_path.parent),
            yolo_root=resolve_path(config.get("yolo_root", "../coarse_router_yolov9/yolov9"), config_path.parent),
            data_yaml=resolve_path(pipeline_cfg.get("router_data_yaml", "../coarse_router_yolov9/datasets/coarse_router_3class_cleaned/data.yaml"), config_path.parent),
            conf_threshold=float(pipeline_cfg.get("router_conf_threshold", 0.25)),
            low_conf_threshold=float(pipeline_cfg.get("router_low_conf_threshold", 0.10)),
            iou_threshold=float(pipeline_cfg.get("router_iou_threshold", 0.45)),
            device=device,
            imgsz=int(pipeline_cfg.get("imgsz", 640)),
        )
    )


def sample_manifest(path: Path, per_class: int, seed: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_key"]].append(row)
    rng = random.Random(seed)
    sampled = []
    for class_key in ["tenjo", "inner_wall", "rc_wall", "rc_column"]:
        candidates = sorted(grouped[class_key], key=lambda r: (r["source"], r["original_stem"]))
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_class])
    return sampled


def evaluate_one(row: dict[str, str], result: dict[str, Any], iou_threshold: float) -> dict[str, Any]:
    image = cv2.imread(str(resolve_path(row["image"], REPO)), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unreadable image: {row['image']}")
    gt = load_gt(resolve_path(row["label"], REPO), image.shape[1], image.shape[0])
    expected_router = CLASS_TO_ROUTER[row["class_key"]]
    main_model = CLASS_TO_MAIN_MODEL[row["class_key"]]
    router_classes = [d.get("class_name") for d in (result.get("router") or {}).get("detections", [])]
    raw = result.get("raw_crack_detections", [])
    main_preds = [d for d in raw if d.get("source_model") == main_model]
    secondary_preds = [d for d in raw if d.get("source_model") != main_model]
    matches = match_predictions(gt, main_preds, iou_threshold)
    fn_count = max(0, len(gt) - len(matches))
    router_hit = int(expected_router in router_classes)
    fn_router_miss = fn_count if not router_hit else 0
    fn_no_main_output = fn_count if router_hit and not main_preds else 0
    fn_iou_miss = fn_count if router_hit and main_preds else 0
    matched_grade_ok = sum(1 for m in matches if m["gt_grade"] == normalize_grade(m["pred_grade"]))
    return {
        "image": row["image"],
        "class_key": row["class_key"],
        "source": row["source"],
        "expected_router_class": expected_router,
        "router_classes": "|".join(router_classes),
        "router_hit": router_hit,
        "main_model": main_model,
        "gt_boxes": len(gt),
        "raw_predictions": len(raw),
        "main_predictions": len(main_preds),
        "secondary_predictions": len(secondary_preds),
        "main_matches_iou50": len(matches),
        "main_false_negative": fn_count,
        "fn_router_miss": fn_router_miss,
        "fn_no_main_output": fn_no_main_output,
        "fn_iou_miss": fn_iou_miss,
        "main_false_positive": max(0, len(main_preds) - len(matches)),
        "matched_grade_ok": matched_grade_ok,
        "matched_grade_mismatch": max(0, len(matches) - matched_grade_ok),
        "warnings": "|".join(result.get("warnings", [])),
        "error": result.get("error", ""),
    }


def load_gt(path: Path, width: int, height: int) -> list[GroundTruth]:
    gts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = [float(v) for v in parts]
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height
        gts.append(GroundTruth((x1, y1, x2, y2), {0: "B", 1: "C", 2: "D"}.get(int(cls), str(int(cls)))))
    return gts


def match_predictions(gt: list[GroundTruth], preds: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    candidates = []
    for gi, item in enumerate(gt):
        for pi, pred in enumerate(preds):
            score = iou_xyxy(item.xyxy, tuple(float(v) for v in pred["bbox_xyxy"]))
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
        matches.append({"iou": score, "gt_grade": item.grade, "pred_grade": pred.get("damage_grade", "")})
    return matches


def find_wall_grade_shift_pairs(row: dict[str, str], result: dict[str, Any], iou_threshold: float, ioa_threshold: float) -> list[dict[str, Any]]:
    if row["class_key"] not in {"inner_wall", "rc_wall"}:
        return []
    raw = result.get("raw_crack_detections", [])
    inner = [d for d in raw if d.get("source_model") == "inner_wall"]
    rc = [d for d in raw if d.get("source_model") == "rc_wall"]
    pairs = []
    for i, a in enumerate(inner):
        for j, b in enumerate(rc):
            a_box = tuple(float(v) for v in a["bbox_xyxy"])
            b_box = tuple(float(v) for v in b["bbox_xyxy"])
            iou = iou_xyxy(a_box, b_box)
            ioa = intersection_over_min_area(a_box, b_box)
            if iou < iou_threshold and ioa < ioa_threshold:
                continue
            inner_grade = normalize_grade(a.get("damage_grade", ""))
            rc_grade = normalize_grade(b.get("damage_grade", ""))
            pairs.append(
                {
                    "image": row["image"],
                    "gt_class": row["class_key"],
                    "inner_index": i,
                    "rc_index": j,
                    "iou": round(iou, 4),
                    "ioa_min": round(ioa, 4),
                    "inner_wall_grade": inner_grade,
                    "rc_wall_grade": rc_grade,
                    "inner_wall_conf": round(float(a.get("confidence", 0)), 4),
                    "rc_wall_conf": round(float(b.get("confidence", 0)), 4),
                    "grade_delta_rc_minus_inner": GRADE_RANK.get(rc_grade, 0) - GRADE_RANK.get(inner_grade, 0),
                }
            )
    return pairs


def intersection_over_min_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return 0.0 if denom <= 0 else inter / denom


def normalize_grade(value: str) -> str:
    for grade in ["B", "C", "D"]:
        if grade in str(value):
            return grade
    return str(value)


def summarize(image_rows: list[dict[str, Any]], wall_pairs: list[dict[str, Any]], per_class: int) -> dict[str, Any]:
    by_class = {}
    for class_key in ["tenjo", "inner_wall", "rc_wall", "rc_column"]:
        rows = [r for r in image_rows if r["class_key"] == class_key]
        by_class[class_key] = {
            "images": len(rows),
            "router_hit": sum(int(r["router_hit"]) for r in rows),
            "gt_boxes": sum(int(r["gt_boxes"]) for r in rows),
            "main_predictions": sum(int(r["main_predictions"]) for r in rows),
            "secondary_predictions": sum(int(r["secondary_predictions"]) for r in rows),
            "main_matches_iou50": sum(int(r["main_matches_iou50"]) for r in rows),
            "main_false_negative": sum(int(r["main_false_negative"]) for r in rows),
            "fn_router_miss": sum(int(r["fn_router_miss"]) for r in rows),
            "fn_no_main_output": sum(int(r["fn_no_main_output"]) for r in rows),
            "fn_iou_miss": sum(int(r["fn_iou_miss"]) for r in rows),
            "main_false_positive": sum(int(r["main_false_positive"]) for r in rows),
            "matched_grade_ok": sum(int(r["matched_grade_ok"]) for r in rows),
            "matched_grade_mismatch": sum(int(r["matched_grade_mismatch"]) for r in rows),
        }
    deltas = Counter(int(p["grade_delta_rc_minus_inner"]) for p in wall_pairs)
    return {
        "images": len(image_rows),
        "per_class": per_class,
        "by_class": by_class,
        "wall_grade_shift_pairs": len(wall_pairs),
        "wall_grade_delta_rc_minus_inner": dict(sorted(deltas.items())),
        "known_debug_limitations": [
            "上一期 GPL 判别模型未必使用本次 final dataset 训练，主分支指标只用于调试统计链路。",
            "raw_crack_detections 未做 NMS，适合分析分支冲突；正式输出仍看 crack_detections。",
            "当前为抽样调试结果，不代表完整数据集真实性能。",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
