#!/usr/bin/env python3
"""Search precision-first operating points for the 3/5-class router pair.

The 5-class model remains the source of every output box.  The 3-class model
can corroborate old-class boxes, but never contributes boxes on its own.  This
keeps the ensemble useful for a precision-first requirement: disagreement can
remove a candidate, while the old model cannot introduce a new false positive.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Callable

import torch
import cv2
from PIL import Image

from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    IMAGE_EXTS,
    Prediction,
    Target,
    box_iou,
    detections_to_predictions,
    match_counts,
    merge_counts,
    metric,
    read_targets,
)


CLASS_NAMES = ["天井", "壁类", "RC柱", "ブレース", "柱脚"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-5class", required=True)
    parser.add_argument(
        "--checkpoint-confirmation",
        "--checkpoint-3class",
        dest="checkpoint_confirmation",
        required=True,
    )
    parser.add_argument("--confirmation-num-classes", type=int, default=3)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirmation-device", default=None)
    parser.add_argument("--image-decoder", choices=("pil", "opencv"), default="pil")
    parser.add_argument("--inference-floor", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--gate-class", type=int, default=1)
    parser.add_argument("--active-classes", default="0,1,2,3,4")
    parser.add_argument("--cache-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def prediction_from_dict(row: dict) -> Prediction:
    return Prediction(cls=int(row["cls"]), conf=float(row["conf"]), xyxy=tuple(row["xyxy"]))


def target_from_dict(row: dict) -> Target:
    return Target(cls=int(row["cls"]), xyxy=tuple(row["xyxy"]))


def serialize_prediction(row: Prediction) -> dict:
    data = asdict(row)
    data.pop("matched", None)
    return data


def serialize_target(row: Target) -> dict:
    data = asdict(row)
    data.pop("matched", None)
    return data


def infer_checkpoint(
    checkpoint: str,
    image_paths: list[Path],
    floor: float,
    num_classes: int,
    device: str,
    image_decoder: str = "pil",
) -> list[list[Prediction]]:
    model = from_checkpoint_matched(checkpoint)
    model_ctx = getattr(model, "model", None)
    if model_ctx is not None and hasattr(model_ctx, "device"):
        model_ctx.device = torch.device(device)

    cached: list[list[Prediction]] = []
    for index, image_path in enumerate(image_paths, 1):
        if image_decoder == "opencv":
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"unreadable image: {image_path}")
            image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        else:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
        detections = model.predict(image, threshold=floor, include_source_image=False)
        predictions, _ = detections_to_predictions(detections, floor, num_classes)
        cached.append(predictions)
        if index % 50 == 0 or index == len(image_paths):
            print(f"{Path(checkpoint).name}: {index}/{len(image_paths)}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cached


def build_cache(args: argparse.Namespace) -> list[dict]:
    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / args.split / "images"
    label_dir = dataset_dir / args.split / "labels"
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix in IMAGE_EXTS and not path.name.startswith("._")
    )
    targets = []
    for image_path in image_paths:
        if args.image_decoder == "opencv":
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"unreadable image: {image_path}")
            height, width = image_bgr.shape[:2]
            size = (width, height)
        else:
            with Image.open(image_path) as handle:
                size = handle.size
        targets.append(read_targets(label_dir / f"{image_path.stem}.txt", *size))

    predictions_5 = infer_checkpoint(
        args.checkpoint_5class,
        image_paths,
        args.inference_floor,
        len(CLASS_NAMES),
        args.device,
        args.image_decoder,
    )
    predictions_confirmation = infer_checkpoint(
        args.checkpoint_confirmation,
        image_paths,
        args.inference_floor,
        args.confirmation_num_classes,
        getattr(args, "confirmation_device", None) or args.device,
        args.image_decoder,
    )
    rows = [
        {
            "image": path.name,
            "targets": [serialize_target(row) for row in target_rows],
            "predictions_5class": [serialize_prediction(row) for row in pred5_rows],
            "predictions_confirmation": [serialize_prediction(row) for row in confirmation_rows],
        }
        for path, target_rows, pred5_rows, confirmation_rows in zip(
            image_paths, targets, predictions_5, predictions_confirmation, strict=True
        )
    ]
    cache_path = Path(args.cache_json)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "checkpoint_5class": args.checkpoint_5class,
                "checkpoint_confirmation": args.checkpoint_confirmation,
                "confirmation_num_classes": args.confirmation_num_classes,
                "dataset_dir": args.dataset_dir,
                "split": args.split,
                "inference_floor": args.inference_floor,
                "image_decoder": args.image_decoder,
                "device": args.device,
                "confirmation_device": getattr(args, "confirmation_device", None) or args.device,
                "images": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return deserialize_rows(rows)


def deserialize_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "image": row["image"],
            "targets": [target_from_dict(item) for item in row["targets"]],
            "predictions_5class": [prediction_from_dict(item) for item in row["predictions_5class"]],
            "predictions_confirmation": [
                prediction_from_dict(item)
                for item in row.get("predictions_confirmation", row.get("predictions_3class", []))
            ],
        }
        for row in rows
    ]


def load_or_build_cache(args: argparse.Namespace) -> list[dict]:
    cache_path = Path(args.cache_json)
    if args.reuse_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        expected = {
            "checkpoint_5class": args.checkpoint_5class,
            "checkpoint_confirmation": args.checkpoint_confirmation,
            "dataset_dir": args.dataset_dir,
            "split": args.split,
            "image_decoder": args.image_decoder,
            "device": args.device,
            "confirmation_device": getattr(args, "confirmation_device", None) or args.device,
        }
        actual_confirmation = payload.get("checkpoint_confirmation", payload.get("checkpoint_3class"))
        mismatches = [
            key for key, value in expected.items()
            if (actual_confirmation if key == "checkpoint_confirmation" else payload.get(key)) != value
        ]
        cached_num_classes = payload.get("confirmation_num_classes", 3)
        if cached_num_classes != args.confirmation_num_classes:
            mismatches.append("confirmation_num_classes")
        if mismatches:
            raise ValueError("prediction cache does not match: " + ", ".join(mismatches))
        print(f"reusing prediction cache: {cache_path}", flush=True)
        return deserialize_rows(payload["images"])
    return build_cache(args)


def counts_for_selector(
    rows: list[dict],
    selector: Callable[[dict], list[Prediction]],
    match_iou: float,
) -> dict[int, dict[str, int]]:
    total = {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(len(CLASS_NAMES))}
    for row in rows:
        merge_counts(total, match_counts(row["targets"], selector(row), match_iou, len(CLASS_NAMES)))
    return total


def class_result(counts: dict[int, dict[str, int]], cls: int) -> dict:
    values = counts[cls]
    precision, recall, f1 = metric(values["tp"], values["fp"], values["fn"])
    return {**values, "precision": precision, "recall": recall, "f1": f1}


def candidate_thresholds(rows: list[dict], cls: int, floor: float) -> list[float]:
    scores = sorted(
        {
            pred.conf
            for row in rows
            for pred in row["predictions_5class"]
            if pred.cls == cls and pred.conf >= floor
        }
    )
    if not scores:
        return [floor]
    thresholds = [floor]
    thresholds.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:], strict=False))
    thresholds.append(math.nextafter(scores[-1], math.inf))
    return sorted(set(thresholds))


def search_threshold_only(
    rows: list[dict], args: argparse.Namespace, active_classes: list[int]
) -> tuple[list[dict], dict[int, dict]]:
    output_rows: list[dict] = []
    best: dict[int, dict] = {}
    for cls in active_classes:
        for threshold in candidate_thresholds(rows, cls, args.inference_floor):
            counts = counts_for_selector(
                rows,
                lambda row, cls=cls, threshold=threshold: [
                    pred
                    for pred in row["predictions_5class"]
                    if pred.cls == cls and pred.conf >= threshold
                ],
                args.match_iou,
            )
            result = {"mode": "threshold", "class_id": cls, "class_name": CLASS_NAMES[cls], "threshold": threshold}
            result.update(class_result(counts, cls))
            output_rows.append(result)
        valid = [
            row for row in output_rows
            if row["class_id"] == cls and row["precision"] >= args.target_precision and row["tp"] > 0
        ]
        if valid:
            best[cls] = max(valid, key=lambda row: (row["recall"], row["f1"], row["precision"]))
    return output_rows, best


def support_score(prediction: Prediction, old_predictions: list[Prediction], cls: int, iou: float) -> float:
    return max(
        (
            old.conf
            for old in old_predictions
            if old.cls == cls and box_iou(prediction.xyxy, old.xyxy) >= iou
        ),
        default=0.0,
    )


def gate_selector(
    row: dict,
    cls: int,
    threshold_5: float,
    threshold_3: float,
    gate_iou: float,
    bypass_5: float,
) -> list[Prediction]:
    selected = []
    for pred in row["predictions_5class"]:
        if pred.cls != cls or pred.conf < threshold_5:
            continue
        if pred.conf >= bypass_5 or support_score(
            pred, row["predictions_confirmation"], cls, gate_iou
        ) >= threshold_3:
            selected.append(pred)
    return selected


def blend_selector(
    row: dict,
    cls: int,
    gate_iou: float,
    weight_3: float,
    threshold: float,
) -> list[Prediction]:
    selected = []
    for pred in row["predictions_5class"]:
        if pred.cls != cls:
            continue
        old_score = support_score(pred, row["predictions_confirmation"], cls, gate_iou)
        combined = (1.0 - weight_3) * pred.conf + weight_3 * old_score
        if combined >= threshold:
            selected.append(pred)
    return selected


def evaluate_gate_grid(
    rows: list[dict],
    args: argparse.Namespace,
    grid: dict[str, list[float]],
    stage: str,
) -> list[dict]:
    cls = args.gate_class
    results = []
    combinations = list(product(grid["threshold_5"], grid["threshold_3"], grid["gate_iou"], grid["bypass_5"]))
    for index, (threshold_5, threshold_3, gate_iou, bypass_5) in enumerate(combinations, 1):
        if bypass_5 < threshold_5:
            continue
        counts = counts_for_selector(
            rows,
            lambda row, values=(threshold_5, threshold_3, gate_iou, bypass_5): gate_selector(
                row, cls, *values
            ),
            args.match_iou,
        )
        result = {
            "mode": "gate",
            "stage": stage,
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "threshold_5": threshold_5,
            "threshold_3": threshold_3,
            "gate_iou": gate_iou,
            "bypass_5": bypass_5,
        }
        result.update(class_result(counts, cls))
        results.append(result)
        if index % 5000 == 0:
            print(f"gate {stage}: {index}/{len(combinations)}", flush=True)
    return results


def search_gate(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    coarse = {
        "threshold_5": [round(value / 100, 2) for value in range(30, 91, 5)],
        "threshold_3": [round(value / 100, 2) for value in range(10, 96, 5)],
        "gate_iou": [round(value / 100, 2) for value in range(20, 81, 10)],
        "bypass_5": [round(value / 100, 2) for value in range(65, 101, 5)] + [1.01],
    }
    results = evaluate_gate_grid(rows, args, coarse, "coarse")
    valid = [row for row in results if row["precision"] >= args.target_precision and row["tp"] > 0]
    if not valid:
        return results, None
    coarse_best = max(valid, key=lambda row: (row["recall"], row["f1"], row["precision"]))

    def around(value: float, radius: float, step: float, low: float, high: float) -> list[float]:
        start = max(low, value - radius)
        stop = min(high, value + radius)
        count = int(round((stop - start) / step))
        return sorted({round(start + index * step, 4) for index in range(count + 1)} | {value})

    fine = {
        "threshold_5": around(coarse_best["threshold_5"], 0.06, 0.01, args.inference_floor, 1.0),
        "threshold_3": around(coarse_best["threshold_3"], 0.10, 0.01, args.inference_floor, 1.0),
        "gate_iou": around(coarse_best["gate_iou"], 0.10, 0.025, 0.05, 0.95),
        "bypass_5": around(coarse_best["bypass_5"], 0.10, 0.01, args.inference_floor, 1.01),
    }
    fine_results = evaluate_gate_grid(rows, args, fine, "fine")
    results.extend(fine_results)
    valid = [row for row in results if row["precision"] >= args.target_precision and row["tp"] > 0]
    return results, max(valid, key=lambda row: (row["recall"], row["f1"], row["precision"]))


def search_blend(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    cls = args.gate_class
    results = []
    for gate_iou, weight_3, threshold in product(
        [round(value / 100, 2) for value in range(20, 81, 5)],
        [round(value / 100, 2) for value in range(10, 91, 5)],
        [round(value / 100, 2) for value in range(20, 96)],
    ):
        counts = counts_for_selector(
            rows,
            lambda row, values=(gate_iou, weight_3, threshold): blend_selector(row, cls, *values),
            args.match_iou,
        )
        result = {
            "mode": "blend",
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "gate_iou": gate_iou,
            "weight_3": weight_3,
            "threshold": threshold,
        }
        result.update(class_result(counts, cls))
        results.append(result)
    valid = [row for row in results if row["precision"] >= args.target_precision and row["tp"] > 0]
    best = max(valid, key=lambda row: (row["recall"], row["f1"], row["precision"])) if valid else None
    return results, best


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def final_selector(best: dict[int, dict], gated_best: dict) -> Callable[[dict], list[Prediction]]:
    def select(row: dict) -> list[Prediction]:
        selected = []
        for cls, candidate in best.items():
            if cls == gated_best["class_id"]:
                continue
            selected.extend(
                pred
                for pred in row["predictions_5class"]
                if pred.cls == cls and pred.conf >= candidate["threshold"]
            )
        if gated_best["mode"] == "threshold":
            selected.extend(
                pred
                for pred in row["predictions_5class"]
                if pred.cls == gated_best["class_id"] and pred.conf >= gated_best["threshold"]
            )
        elif gated_best["mode"] == "gate":
            selected.extend(
                gate_selector(
                    row,
                    gated_best["class_id"],
                    gated_best["threshold_5"],
                    gated_best["threshold_3"],
                    gated_best["gate_iou"],
                    gated_best["bypass_5"],
                )
            )
        else:
            selected.extend(
                blend_selector(
                    row,
                    gated_best["class_id"],
                    gated_best["gate_iou"],
                    gated_best["weight_3"],
                    gated_best["threshold"],
                )
            )
        return sorted(selected, key=lambda pred: pred.conf, reverse=True)

    return select


def main() -> int:
    args = parse_args()
    active_classes = sorted({int(value) for value in args.active_classes.split(",") if value.strip()})
    if not active_classes or any(not 0 <= cls < len(CLASS_NAMES) for cls in active_classes):
        raise ValueError("active classes must be class IDs between 0 and 4")
    if args.gate_class not in active_classes:
        raise ValueError("gate class must be included in active classes")
    if not 0 <= args.gate_class < min(args.confirmation_num_classes, len(CLASS_NAMES)):
        raise ValueError("gate class is not emitted by the confirmation model")
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_or_build_cache(args)
    print(f"cached rows: {len(rows)}", flush=True)
    if args.cache_only:
        return 0
    threshold_rows, threshold_best = search_threshold_only(rows, args, active_classes)
    write_csv(output_dir / "threshold_only.csv", threshold_rows)
    if len(threshold_best) != len(active_classes):
        missing = sorted(set(active_classes) - set(threshold_best))
        raise RuntimeError(f"no threshold-only solution at target precision for classes: {missing}")

    gate_rows, gate_best = search_gate(rows, args)
    write_csv(output_dir / "wall_gate.csv", gate_rows)
    blend_rows, blend_best = search_blend(rows, args)
    write_csv(output_dir / "wall_blend.csv", blend_rows)

    gated_candidates = [threshold_best[args.gate_class]]
    gated_candidates.extend(candidate for candidate in (gate_best, blend_best) if candidate is not None)
    gated_best = max(gated_candidates, key=lambda row: (row["recall"], row["f1"], row["precision"]))
    final_counts = counts_for_selector(rows, final_selector(threshold_best, gated_best), args.match_iou)
    per_class = {
        CLASS_NAMES[cls]: {
            **class_result(final_counts, cls),
            "selection": gated_best if cls == args.gate_class else threshold_best[cls],
        }
        for cls in active_classes
    }
    overall_tp = sum(row["tp"] for row in per_class.values())
    overall_fp = sum(row["fp"] for row in per_class.values())
    overall_fn = sum(row["fn"] for row in per_class.values())
    overall_precision, overall_recall, overall_f1 = metric(overall_tp, overall_fp, overall_fn)
    summary = {
        "target_precision_per_class": args.target_precision,
        "images": len(rows),
        "checkpoint_5class": args.checkpoint_5class,
        "checkpoint_confirmation": args.checkpoint_confirmation,
        "confirmation_num_classes": args.confirmation_num_classes,
        "gate_class_candidates": gated_candidates,
        "selected_gate_strategy": gated_best,
        "per_class": per_class,
        "overall": {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
