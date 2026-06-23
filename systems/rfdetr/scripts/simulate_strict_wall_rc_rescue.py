#!/usr/bin/env python3
"""Offline simulation for strict wall<->RC-column rescue filtering."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rfdetr_prod_pipeline.pipeline.display_merge import suppress_overlapping_display_detections
from rfdetr_prod_pipeline.pipeline.result_merge import area_xyxy, grade_level, iou_xyxy


BASELINE = Path("outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/results.jsonl")
FALLBACK = Path("outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_wall_rc_sister_fallback_v1/results.jsonl")
SPLIT = Path("data/pipeline_eval_official_plus_20260615/split.json")
OUT = Path("outputs/rfdetr_prod_pipeline/wall_rc_strict_rescue_sim_20260616")
CLASS_ID_TO_GRADE = {"0": "B", "1": "C", "2": "D"}


VARIANTS = {
    "loose_conf030": {"conf": 0.30, "shape": False, "top_per_region": 2},
    "medium_conf035_shape": {"conf": 0.35, "shape": True, "top_per_region": 1},
    "strict_conf045_shape": {"conf": 0.45, "shape": True, "top_per_region": 1},
    "very_strict_conf055_shape": {"conf": 0.55, "shape": True, "top_per_region": 1},
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {Path(s["eval_image"]).name: s for s in json.loads(SPLIT.read_text(encoding="utf-8"))["samples"]}
    baseline = load_jsonl_by_name(BASELINE)
    fallback = load_jsonl_by_name(FALLBACK)
    gt_by_image = {name: load_gt(samples[name], baseline[name]["image_shape"]) for name in baseline}

    baseline_eval = eval_predictions(baseline, gt_by_image)
    summary: dict[str, Any] = {"baseline": baseline_eval["summary"], "variants": {}}
    for variant_name, cfg in VARIANTS.items():
        rows = []
        per_image = []
        accepted_by_image: dict[str, list[dict[str, Any]]] = {}
        for name, base_row in baseline.items():
            candidates = accepted_rescue_candidates(fallback[name], cfg)
            accepted_by_image[name] = candidates
            display = list(base_row.get("display_crack_detections", [])) + candidates
            display, suppressed = suppress_overlapping_display_detections(display)
            sim_row = dict(base_row)
            sim_row["display_crack_detections"] = display
            sim_row["strict_rescue_candidates"] = candidates
            sim_row["strict_rescue_suppressed"] = suppressed
            rows.append(sim_row)
        sim = {Path(r["image"]).name: r for r in rows}
        sim_eval = eval_predictions(sim, gt_by_image)
        diff = diff_images(baseline_eval["per_image"], sim_eval["per_image"])
        summary["variants"][variant_name] = {
            "config": cfg,
            "summary": sim_eval["summary"],
            "accepted_rescue_detections": sum(len(v) for v in accepted_by_image.values()),
            "images_with_rescue": sum(1 for v in accepted_by_image.values() if v),
            "diff": diff["summary"],
        }
        write_csv(OUT / f"{variant_name}_per_image.csv", diff["rows"])
        write_jsonl(OUT / f"{variant_name}_sim_results.jsonl", rows)
        print_variant(variant_name, summary["variants"][variant_name])

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


def load_jsonl_by_name(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[Path(row["image"]).name] = row
    return out


def accepted_rescue_candidates(row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    h, w = row["image_shape"][:2]
    for rec in row.get("raw_crack_detections", []):
        if not rec.get("is_fallback"):
            continue
        reasons = rec.get("fallback_reasons") or []
        if not any(str(reason).startswith("wall_rc_sister:") for reason in reasons):
            continue
        if float(rec.get("confidence") or 0.0) < float(cfg["conf"]):
            continue
        if cfg["shape"] and not shape_ok(rec, w, h):
            continue
        router_region_index = int(rec.get("router_region_index", -1))
        source_router_class = str(rec.get("source_router_class") or "")
        buckets[(router_region_index, source_router_class)].append(rec)

    accepted = []
    for items in buckets.values():
        items = sorted(items, key=rescue_priority, reverse=True)[: int(cfg["top_per_region"])]
        accepted.extend(to_display_record(item) for item in items)
    return accepted


def shape_ok(rec: dict[str, Any], width: int, height: int) -> bool:
    x1, y1, x2, y2 = [float(v) for v in rec.get("bbox_xyxy", [0, 0, 0, 0])]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_ratio = bw * bh / max(float(width * height), 1.0)
    aspect_h_over_w = bh / max(bw, 1e-6)
    source_router_class = str(rec.get("source_router_class") or "")
    if source_router_class == "RC柱":
        return 0.01 <= area_ratio <= 0.55 and aspect_h_over_w >= 1.05
    if source_router_class == "壁类":
        return 0.02 <= area_ratio <= 0.85
    return False


def rescue_priority(rec: dict[str, Any]) -> tuple[float, float]:
    box = tuple(float(v) for v in rec.get("bbox_xyxy", [0, 0, 0, 0]))
    return (float(rec.get("confidence") or 0.0), area_xyxy(box))


def to_display_record(rec: dict[str, Any]) -> dict[str, Any]:
    source_router_class = str(rec.get("source_router_class") or "")
    grade = grade_level(str(rec.get("damage_grade") or ""))
    if source_router_class == "壁类":
        structure_type = "壁類"
        damage_grade = f"壁-{grade}"
    elif source_router_class == "RC柱":
        structure_type = "RC柱"
        damage_grade = grade
    else:
        structure_type = source_router_class or None
        damage_grade = grade
    return {
        "status": "strict_wall_rc_rescue",
        "structure_type": structure_type,
        "damage_grade": damage_grade,
        "raw_damage_grade": rec.get("damage_grade"),
        "confidence": rec.get("confidence"),
        "bbox_xyxy": rec.get("bbox_xyxy"),
        "source_model": rec.get("source_model"),
        "source_router_class": source_router_class,
        "reason": "strict wall/RC sister rescue simulation",
        "fallback_reasons": rec.get("fallback_reasons") or [],
    }


def load_gt(sample: dict[str, Any], shape: list[int]) -> list[dict[str, Any]]:
    h, w = shape[:2]
    out = []
    for line in Path(sample["eval_label"]).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[:5]
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        out.append(
            {
                "grade": CLASS_ID_TO_GRADE.get(cls, cls),
                "bbox_xyxy": [
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ],
            }
        )
    return out


def eval_predictions(rows: dict[str, dict[str, Any]], gt_by_image: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_image = {}
    totals = Counter()
    by_component = defaultdict(Counter)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["samples"]
    component_by_name = {Path(s["eval_image"]).name: s["component"] for s in split}
    for name, row in rows.items():
        metrics = match(gt_by_image[name], row.get("display_crack_detections", []))
        component = component_by_name[name]
        per_image[name] = {"image": name, "component": component, **metrics}
        for key in ["tp", "fp", "fn", "pred", "gt"]:
            totals[key] += metrics[key]
            by_component[component][key] += metrics[key]
    return {
        "summary": {"overall": prf(totals), "by_component": {k: prf(v) for k, v in by_component.items()}},
        "per_image": per_image,
    }


def match(gt: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float = 0.50) -> dict[str, int]:
    used = set()
    tp = 0
    for item in gt:
        best = None
        best_iou = 0.0
        for idx, pred in enumerate(preds):
            if idx in used:
                continue
            if grade_level(str(pred.get("damage_grade", ""))) != item["grade"]:
                continue
            iou = iou_xyxy(tuple(item["bbox_xyxy"]), tuple(float(v) for v in pred.get("bbox_xyxy", [0, 0, 0, 0])))
            if iou >= iou_threshold and iou > best_iou:
                best = idx
                best_iou = iou
        if best is not None:
            used.add(best)
            tp += 1
    return {"tp": tp, "fp": len(preds) - len(used), "fn": len(gt) - tp, "pred": len(preds), "gt": len(gt)}


def prf(counter: Counter) -> dict[str, Any]:
    tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "pred": counter["pred"], "precision": precision, "recall": recall, "f1": f1}


def diff_images(base: dict[str, dict[str, Any]], sim: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    improved = worsened = fp_up = fp_down = 0
    for name, before in base.items():
        after = sim[name]
        row = {
            "image": name,
            "component": before["component"],
            "base_tp": before["tp"],
            "base_fp": before["fp"],
            "base_fn": before["fn"],
            "sim_tp": after["tp"],
            "sim_fp": after["fp"],
            "sim_fn": after["fn"],
            "delta_tp": after["tp"] - before["tp"],
            "delta_fp": after["fp"] - before["fp"],
            "delta_fn": after["fn"] - before["fn"],
        }
        rows.append(row)
        if row["delta_tp"] > 0 or row["delta_fn"] < 0:
            improved += 1
        if row["delta_tp"] < 0 or row["delta_fn"] > 0:
            worsened += 1
        if row["delta_fp"] > 0:
            fp_up += 1
        if row["delta_fp"] < 0:
            fp_down += 1
    return {"rows": rows, "summary": {"improved_images": improved, "worsened_images": worsened, "fp_up_images": fp_up, "fp_down_images": fp_down}}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_variant(name: str, result: dict[str, Any]) -> None:
    overall = result["summary"]["overall"]
    print(
        name,
        "accepted",
        result["accepted_rescue_detections"],
        "images",
        result["images_with_rescue"],
        "TP/FP/FN",
        overall["tp"],
        overall["fp"],
        overall["fn"],
        "P/R/F1",
        round(overall["precision"], 4),
        round(overall["recall"], 4),
        round(overall["f1"], 4),
        "diff",
        result["diff"],
    )


if __name__ == "__main__":
    main()
