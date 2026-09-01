#!/usr/bin/env python3
"""Generate before/after visual evidence for the Router precision ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_rfdetr_threshold_sweep import box_iou  # noqa: E402


CLASS_NAMES = ["天井", "壁类", "RC柱", "ブレース", "柱脚"]
CLASS_COLORS = ["#3977c3", "#7651b5", "#008d89", "#df8c22", "#8b5a2b"]
GT_COLOR = "#18a558"
TP_COLOR = "#177ddc"
FP_COLOR = "#e43d30"
FN_COLOR = "#f49b23"
PANEL_BG = "#f5f7fa"
TEXT_COLOR = "#17202a"
MUTED_COLOR = "#5f6b76"


@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float | None = None


@dataclass
class MatchResult:
    pred_status: list[str]
    target_status: list[str]
    tp: int
    fp: int
    fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="systems/rfdetr/router/configs/router_5class_precision_ensemble_20260831.yaml",
    )
    parser.add_argument(
        "--cache-3class",
        default="outputs/router_precision_20260831/predictions_cv2_production_devices_primary_3class.json",
    )
    parser.add_argument(
        "--cache-brace",
        default="outputs/router_precision_20260831/predictions_cv2_production_devices_primary_historical5.json",
    )
    parser.add_argument(
        "--dataset-dir",
        default="handoff_20260707_rfdetr_main/data/router_5class_reviewed_dedup_test_as_valid",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--cases-per-group", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="docs/reports/assets/router_precision_ensemble_20260901",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def deserialize_boxes(rows: list[dict[str, Any]], prediction: bool) -> list[Box]:
    return [
        Box(
            cls=int(row["cls"]),
            conf=float(row["conf"]) if prediction else None,
            xyxy=tuple(float(value) for value in row["xyxy"]),
        )
        for row in rows
    ]


def load_cache(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {row["image"]: row for row in payload["images"]}


def primary_signature(rows: list[dict[str, Any]]) -> list[tuple[int, float, tuple[float, ...]]]:
    return [
        (int(row["cls"]), round(float(row["conf"]), 9), tuple(round(float(v), 6) for v in row["xyxy"]))
        for row in rows
    ]


def match_boxes(targets: list[Box], predictions: list[Box], threshold: float) -> MatchResult:
    target_status = ["fn"] * len(targets)
    pred_status = ["fp"] * len(predictions)
    for pred_idx, pred in enumerate(predictions):
        candidates = [
            (box_iou(pred.xyxy, target.xyxy), target_idx)
            for target_idx, target in enumerate(targets)
            if target_status[target_idx] == "fn" and target.cls == pred.cls
        ]
        if not candidates:
            continue
        best_iou, best_idx = max(candidates, key=lambda item: item[0])
        if best_iou >= threshold:
            pred_status[pred_idx] = "tp"
            target_status[best_idx] = "tp"
    return MatchResult(
        pred_status=pred_status,
        target_status=target_status,
        tp=pred_status.count("tp"),
        fp=pred_status.count("fp"),
        fn=target_status.count("fn"),
    )


def baseline_predictions(primary: list[Box], thresholds: list[float]) -> list[Box]:
    return sorted(
        [box for box in primary if box.conf is not None and box.conf >= thresholds[box.cls]],
        key=lambda box: float(box.conf),
        reverse=True,
    )


def ensemble_predictions(
    primary: list[Box],
    confirmation_3class: list[Box],
    confirmation_brace: list[Box],
    config: dict[str, Any],
) -> list[Box]:
    selected: list[Box] = []
    for candidate in primary:
        name = CLASS_NAMES[candidate.cls]
        point = config["operating_points"][name]
        threshold_5 = float(point["threshold_5"])
        if candidate.conf is None or candidate.conf < threshold_5:
            continue
        if point["mode"] == "threshold":
            selected.append(candidate)
            continue
        bypass = float(point["bypass_5"])
        if candidate.conf >= bypass:
            selected.append(candidate)
            continue
        if name == "ブレース":
            support_rows = confirmation_brace
            support_threshold = float(point["threshold_confirmation"])
        else:
            support_rows = confirmation_3class
            support_threshold = float(point["threshold_3"])
        supported = any(
            support.cls == candidate.cls
            and support.conf is not None
            and support.conf >= support_threshold
            and box_iou(candidate.xyxy, support.xyxy) >= float(point["gate_iou"])
            for support in support_rows
        )
        if supported:
            selected.append(candidate)
    return sorted(selected, key=lambda box: float(box.conf), reverse=True)


def aggregate(cases: list[dict[str, Any]], key: str) -> dict[str, Any]:
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    for case in cases:
        predictions = case[key]
        result = case[f"{key}_match"]
        for pred, status in zip(predictions, result.pred_status, strict=True):
            per_class[CLASS_NAMES[pred.cls]][status] += 1
        for target, status in zip(case["targets"], result.target_status, strict=True):
            if status == "fn":
                per_class[CLASS_NAMES[target.cls]]["fn"] += 1
    overall = {metric: sum(row[metric] for row in per_class.values()) for metric in ("tp", "fp", "fn")}
    for row in [*per_class.values(), overall]:
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
    return {"per_class": per_class, "overall": overall}


def case_delta(case: dict[str, Any]) -> dict[str, int]:
    before = case["baseline_match"]
    after = case["ensemble_match"]
    return {
        "tp_added": after.tp - before.tp,
        "fp_removed": before.fp - after.fp,
        "fn_removed": before.fn - after.fn,
    }


def has_cross_class_correction(case: dict[str, Any]) -> bool:
    delta = case_delta(case)
    if delta["tp_added"] <= 0 or delta["fp_removed"] <= 0:
        return False
    before_fp = {
        box.cls
        for box, status in zip(case["baseline"], case["baseline_match"].pred_status, strict=True)
        if status == "fp"
    }
    after_tp = {
        box.cls
        for box, status in zip(case["ensemble"], case["ensemble_match"].pred_status, strict=True)
        if status == "tp"
    }
    return bool(before_fp and after_tp and before_fp != after_tp)


def rank_cases(cases: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "recognition_corrections":
        candidates = [case for case in cases if has_cross_class_correction(case)]
        score = lambda case: (
            case_delta(case)["tp_added"] + case_delta(case)["fp_removed"],
            case_delta(case)["tp_added"],
            case_delta(case)["fp_removed"],
        )
    elif group == "missed_recovered":
        candidates = [case for case in cases if case_delta(case)["tp_added"] > 0]
        score = lambda case: (
            case_delta(case)["tp_added"],
            case_delta(case)["fp_removed"],
            -case["ensemble_match"].fp,
        )
    elif group == "false_positives_removed":
        candidates = [case for case in cases if case_delta(case)["fp_removed"] > 0]
        score = lambda case: (
            case_delta(case)["fp_removed"],
            case_delta(case)["tp_added"],
            -case["ensemble_match"].fn,
        )
    else:
        raise ValueError(group)
    return sorted(candidates, key=score, reverse=True)


def fit_image(image: Image.Image, max_width: int = 620, max_height: int = 720) -> tuple[Image.Image, float]:
    scale = min(1.0, max_width / image.width, max_height / image.height)
    if scale == 1.0:
        return image.convert("RGB"), scale
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS), scale


def scale_box(box: Box, scale: float) -> Box:
    return Box(box.cls, tuple(value * scale for value in box.xyxy), box.conf)


def dashed_rectangle(draw: ImageDraw.ImageDraw, xyxy: tuple[float, ...], color: str, width: int = 4) -> None:
    x1, y1, x2, y2 = [round(value) for value in xyxy]
    dash = 12
    for x in range(x1, x2, dash * 2):
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
    for y in range(y1, y2, dash * 2):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, color: str) -> None:
    font = load_font(19, bold=True)
    bbox = draw.textbbox(xy, value, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill="#ffffff", outline=color, width=2)
    draw.text(xy, value, font=font, fill=color)


def draw_panel(
    image: Image.Image,
    title: str,
    targets: list[Box],
    predictions: list[Box] | None,
    match: MatchResult | None,
) -> Image.Image:
    header = 72
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header), fill=PANEL_BG)
    draw.text((14, 10), title, font=load_font(25, bold=True), fill=TEXT_COLOR)
    if match is not None:
        summary = f"TP {match.tp}   FP {match.fp}   FN {match.fn}"
        draw.text((14, 42), summary, font=load_font(18), fill=MUTED_COLOR)

    for idx, target in enumerate(targets):
        shifted = tuple(value + (header if position % 2 else 0) for position, value in enumerate(target.xyxy))
        if match is not None and match.target_status[idx] == "fn":
            dashed_rectangle(draw, shifted, FN_COLOR, width=5)
            label(draw, (round(shifted[0]) + 3, round(shifted[1]) + 3), f"FN GT {CLASS_NAMES[target.cls]}", FN_COLOR)
        else:
            draw.rectangle(shifted, outline=GT_COLOR, width=4)
            if predictions is None:
                label(draw, (round(shifted[0]) + 3, round(shifted[1]) + 3), f"GT {CLASS_NAMES[target.cls]}", GT_COLOR)

    if predictions is not None and match is not None:
        for prediction, status in zip(predictions, match.pred_status, strict=True):
            shifted = tuple(value + (header if position % 2 else 0) for position, value in enumerate(prediction.xyxy))
            color = TP_COLOR if status == "tp" else FP_COLOR
            draw.rectangle(shifted, outline=color, width=4)
            confidence = float(prediction.conf) if prediction.conf is not None else 0.0
            label(
                draw,
                (round(shifted[0]) + 3, max(header + 2, round(shifted[1]) - 25)),
                f"{status.upper()} {CLASS_NAMES[prediction.cls]} {confidence:.2f}",
                color,
            )
    return canvas


def save_case(case: dict[str, Any], output_path: Path, group_title: str) -> None:
    source, scale = fit_image(Image.open(case["image_path"]).convert("RGB"))
    targets = [scale_box(box, scale) for box in case["targets"]]
    baseline = [scale_box(box, scale) for box in case["baseline"]]
    ensemble = [scale_box(box, scale) for box in case["ensemble"]]
    panels = [
        draw_panel(source, "Ground truth", targets, None, None),
        draw_panel(source, "改善前：单模型", targets, baseline, case["baseline_match"]),
        draw_panel(source, "改善后：确认式融合", targets, ensemble, case["ensemble_match"]),
    ]
    gap = 14
    title_h = 88
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + title_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    delta = case_delta(case)
    title = f"{group_title}｜{case['image']}"
    subtitle = (
        f"变化：TP {delta['tp_added']:+d}，FP {-delta['fp_removed']:+d}，FN {-delta['fn_removed']:+d}"
        "　　绿色=GT，蓝色=TP，红色=FP，橙色虚线=FN"
    )
    draw.text((18, 10), title, font=load_font(27, bold=True), fill=TEXT_COLOR)
    draw.text((18, 48), subtitle, font=load_font(19), fill=MUTED_COLOR)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_h))
        x += panel.width + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def thumbnail(path: Path, width: int = 900) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = min(1.0, width / image.width)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def save_contact_sheet(paths: list[Path], output_path: Path, title: str) -> None:
    images = [thumbnail(path) for path in paths]
    if not images:
        return
    columns = 2
    gap = 18
    title_h = 70
    rows = math.ceil(len(images) / columns)
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    canvas = Image.new("RGB", (columns * cell_w + gap * (columns - 1), title_h + rows * cell_h + gap * (rows - 1)), "white")
    ImageDraw.Draw(canvas).text((16, 14), title, font=load_font(30, bold=True), fill=TEXT_COLOR)
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        x = col * (cell_w + gap)
        y = title_h + row * (cell_h + gap)
        canvas.paste(image, (x, y))
    canvas.save(output_path, optimize=True)


def save_metric_chart(baseline: dict[str, Any], ensemble: dict[str, Any], output_path: Path) -> None:
    cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk_font.exists():
        font_manager.fontManager.addfont(str(cjk_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(cjk_font)).get_name()
        plt.rcParams["axes.unicode_minus"] = False
    labels = [*CLASS_NAMES, "整体"]
    baseline_rows = [baseline["per_class"][name] for name in CLASS_NAMES] + [baseline["overall"]]
    ensemble_rows = [ensemble["per_class"][name] for name in CLASS_NAMES] + [ensemble["overall"]]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, metric, title in zip(axes, ("precision", "recall"), ("Precision 对比", "Recall 对比"), strict=True):
        ax.bar([value - 0.19 for value in x], [row[metric] for row in baseline_rows], 0.38, label="改善前", color="#9aa5b1")
        ax.bar([value + 0.19 for value in x], [row[metric] for row in ensemble_rows], 0.38, label="改善后", color="#2f80c9")
        if metric == "precision":
            ax.axhline(0.90, color="#d64545", linestyle="--", linewidth=1.5, label="目标 0.90")
        ax.set_xticks(x, labels)
        ax.set_ylim(0.65, 1.02)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for index, row in enumerate(ensemble_rows):
            ax.text(index + 0.19, row[metric] + 0.008, f"{row[metric]:.3f}", ha="center", fontsize=8)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left")
    fig.suptitle("RF-DETR Router：确认式融合前后指标", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = ["group", "rank", "image", "tp_before", "fp_before", "fn_before", "tp_after", "fp_after", "fn_after", "tp_added", "fp_removed", "fn_removed", "asset"]
    with (output_dir / "selected_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selected_cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    payload_3, cache_3 = load_cache(Path(args.cache_3class))
    payload_brace, cache_brace = load_cache(Path(args.cache_brace))
    if payload_3.get("image_decoder") != "opencv" or payload_brace.get("image_decoder") != "opencv":
        raise ValueError("visual evidence must use OpenCV production caches")
    if set(cache_3) != set(cache_brace):
        raise ValueError("cache image sets differ")

    image_dir = Path(args.dataset_dir) / args.split / "images"
    cases = []
    for image_name in sorted(cache_3):
        row_3 = cache_3[image_name]
        row_brace = cache_brace[image_name]
        if primary_signature(row_3["predictions_5class"]) != primary_signature(row_brace["predictions_5class"]):
            raise ValueError(f"primary cache differs for {image_name}")
        targets = deserialize_boxes(row_3["targets"], prediction=False)
        primary = deserialize_boxes(row_3["predictions_5class"], prediction=True)
        confirm_3 = deserialize_boxes(row_3["predictions_confirmation"], prediction=True)
        confirm_brace = deserialize_boxes(row_brace["predictions_confirmation"], prediction=True)
        baseline = baseline_predictions(primary, [float(value) for value in config["same_path_single_model_baseline"]["thresholds"]])
        ensemble = ensemble_predictions(primary, confirm_3, confirm_brace, config)
        cases.append(
            {
                "image": image_name,
                "image_path": image_dir / image_name,
                "targets": targets,
                "baseline": baseline,
                "ensemble": ensemble,
                "baseline_match": match_boxes(targets, baseline, args.match_iou),
                "ensemble_match": match_boxes(targets, ensemble, args.match_iou),
            }
        )

    baseline_summary = aggregate(cases, "baseline")
    ensemble_summary = aggregate(cases, "ensemble")
    expected_before = config["same_path_single_model_baseline"]["overall"]
    expected_after = config["frozen_test_metrics"]["overall"]
    for key in ("tp", "fp", "fn"):
        if baseline_summary["overall"][key] != int(expected_before[key]):
            raise AssertionError(("baseline", key, baseline_summary["overall"][key], expected_before[key]))
        if ensemble_summary["overall"][key] != int(expected_after[key]):
            raise AssertionError(("ensemble", key, ensemble_summary["overall"][key], expected_after[key]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_titles = {
        "recognition_corrections": "错误识别被纠正",
        "missed_recovered": "漏检被找回",
        "false_positives_removed": "误报被消除",
    }
    selected_names: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    contact_paths: dict[str, list[Path]] = {}
    for group, title in group_titles.items():
        ranked = rank_cases(cases, group)
        selected = []
        for case in ranked:
            if case["image"] in selected_names:
                continue
            selected.append(case)
            selected_names.add(case["image"])
            if len(selected) >= args.cases_per_group:
                break
        # If a small category has too few unique cases, permit reuse rather than leave the sheet empty.
        if len(selected) < args.cases_per_group:
            for case in ranked:
                if case in selected:
                    continue
                selected.append(case)
                if len(selected) >= args.cases_per_group:
                    break
        group_dir = output_dir / group
        contact_paths[group] = []
        for rank, case in enumerate(selected, 1):
            asset = group_dir / f"{rank:02d}_{Path(case['image']).stem}.png"
            save_case(case, asset, title)
            contact_paths[group].append(asset)
            delta = case_delta(case)
            manifest_rows.append(
                {
                    "group": group,
                    "rank": rank,
                    "image": case["image"],
                    "tp_before": case["baseline_match"].tp,
                    "fp_before": case["baseline_match"].fp,
                    "fn_before": case["baseline_match"].fn,
                    "tp_after": case["ensemble_match"].tp,
                    "fp_after": case["ensemble_match"].fp,
                    "fn_after": case["ensemble_match"].fn,
                    **delta,
                    "asset": str(asset.relative_to(output_dir)),
                }
            )
        save_contact_sheet(contact_paths[group], output_dir / f"contact_{group}.png", title)

    save_metric_chart(baseline_summary, ensemble_summary, output_dir / "metrics_before_after.png")
    write_manifest(manifest_rows, output_dir)
    summary = {
        "images": len(cases),
        "match_iou": args.match_iou,
        "baseline": baseline_summary,
        "ensemble": ensemble_summary,
        "selected_cases": len(manifest_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["baseline"]["overall"], ensure_ascii=False))
    print(json.dumps(summary["ensemble"]["overall"], ensure_ascii=False))
    print(f"visualizations: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
