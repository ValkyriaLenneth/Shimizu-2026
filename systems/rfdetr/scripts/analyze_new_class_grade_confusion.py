#!/usr/bin/env python3
"""Split each missed ground-truth box into a detection miss or a grade confusion.

The distinction matters because the fixes are different. If a D box is never
detected at all, the problem is detection - more data, better scale handling. If
the box is found but labelled C, the problem is grade discrimination, and no
amount of detection work will fix it; that calls for label review or a different
head.

For every GT box the script asks, at the operating thresholds:

    matched        - a prediction of the same grade overlaps it at IoU >= match_iou
    grade_confused - some prediction overlaps it, but all of them have other grades
    missed         - nothing overlaps it at all

and reports the confusion direction for the confused ones, so "D read as C" is
visible separately from "D read as B".
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from checkpoint_resolution import resolution_from_checkpoint

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="0.3,0.35,0.4", help="per-class B,C,D")
    parser.add_argument("--iou-threshold", type=float, default=0.229)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate at most N images, sampled evenly; 0 means all")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        cls = int(fields[0])
        cx, cy, bw, bh = (float(v) for v in fields[1:])
        out.append(
            (
                cls,
                (
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ),
            )
        )
    return out


def main() -> None:
    args = parse_args()
    import numpy as np
    from PIL import Image
    from rfdetr import RFDETRMedium

    thresholds = [float(v) for v in args.thresholds.split(",")]
    if len(thresholds) != 3:
        raise ValueError("--thresholds needs three comma-separated values")
    min_threshold = min(thresholds)

    # Resolution is not stored in the checkpoint args; recover it from the
    # positional-encoding tensor so eval preprocessing matches training.
    _res = resolution_from_checkpoint(args.checkpoint)
    _res_kw = {"resolution": _res} if _res is not None else {}
    if _res is not None:
        print(f"  [resolution] building model at {_res} px (from checkpoint)")
    model = RFDETRMedium(
        pretrain_weights=args.checkpoint, num_classes=3, device=args.device, **_res_kw
    )

    split_dir = Path(args.dataset_dir) / args.split
    images = sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit and len(images) > args.limit:
        # Even stride rather than the first N, so the sample is not biased toward
        # one alphabetical region of the dataset.
        step = len(images) / args.limit
        images = [images[int(i * step)] for i in range(args.limit)]

    outcome: Counter = Counter()
    per_grade: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    examples: list[dict] = []
    # Relative box area by outcome, to test directly whether misses are the small
    # boxes. If the small-object explanation is right, missed boxes should be
    # systematically smaller than matched ones.
    areas_by_outcome: dict[str, list[float]] = defaultdict(list)

    for image_path in images:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            detections = model.predict(image, threshold=min_threshold)

        preds = []
        xyxy = np.asarray(detections.xyxy)
        conf = np.asarray(detections.confidence)
        cls_ids = np.asarray(detections.class_id)
        for box, score, cls in zip(xyxy, conf, cls_ids, strict=False):
            cls = int(cls)
            if cls not in GRADES:
                continue
            if float(score) < thresholds[cls]:
                continue
            preds.append((cls, tuple(float(v) for v in box)))

        for gt_cls, gt_box in load_gt(split_dir / "labels" / f"{image_path.stem}.txt", width, height):
            grade = GRADES[gt_cls]
            same = [p for p in preds if p[0] == gt_cls and iou(gt_box, p[1]) >= args.iou_threshold]
            other = [p for p in preds if p[0] != gt_cls and iou(gt_box, p[1]) >= args.iou_threshold]
            if same:
                state = "matched"
            elif other:
                state = "grade_confused"
                best = max(other, key=lambda p: iou(gt_box, p[1]))
                confusion[f"{grade}->{GRADES[best[0]]}"] += 1
                if len(examples) < 40:
                    examples.append(
                        {
                            "image": image_path.name,
                            "gt_grade": grade,
                            "predicted_as": GRADES[best[0]],
                            "iou": round(iou(gt_box, best[1]), 3),
                        }
                    )
            else:
                state = "missed"
            outcome[state] += 1
            per_grade[grade][state] += 1
            gx1, gy1, gx2, gy2 = gt_box
            areas_by_outcome[state].append(
                max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1) / float(width * height)
            )

    total = sum(outcome.values())
    print(f"checkpoint : {args.checkpoint}")
    print(f"dataset    : {args.dataset_dir} [{args.split}]  {len(images)} images, {total} GT boxes")
    print(f"thresholds : B/C/D = {thresholds}, match IoU {args.iou_threshold}\n")

    print(f"  {'grade':<7} {'GT':>4} {'matched':>8} {'confused':>9} {'missed':>7}   {'recall':>7}")
    for grade in ("B", "C", "D"):
        counts = per_grade[grade]
        n = sum(counts.values())
        if not n:
            continue
        recall = counts["matched"] / n
        print(
            f"  {grade:<7} {n:>4} {counts['matched']:>8} {counts['grade_confused']:>9} "
            f"{counts['missed']:>7}   {recall:>7.3f}"
        )
    print(
        f"  {'ALL':<7} {total:>4} {outcome['matched']:>8} {outcome['grade_confused']:>9} "
        f"{outcome['missed']:>7}   {outcome['matched'] / total if total else 0:>7.3f}"
    )

    if confusion:
        print("\n  grade confusion directions (box found, wrong grade):")
        for key, value in confusion.most_common():
            print(f"    {key}: {value}")

    print("\n  reading: 'confused' boxes are detected but graded wrong - a discrimination")
    print("  problem. 'missed' boxes are not found at all - a detection problem.")

    def describe(values: list[float]) -> str:
        if not values:
            return "n=0"
        ordered = sorted(values)
        mid = ordered[len(ordered) // 2]
        return (
            f"n={len(ordered):>3} median={mid:.4f} "
            f"min={ordered[0]:.4f} max={ordered[-1]:.4f}"
        )

    print("\n  relative GT box area by outcome (tests the small-object explanation):")
    for state in ("matched", "grade_confused", "missed"):
        if areas_by_outcome.get(state):
            print(f"    {state:<15} {describe(areas_by_outcome[state])}")
    matched_areas = sorted(areas_by_outcome.get("matched", []))
    missed_areas = sorted(areas_by_outcome.get("missed", []))
    if matched_areas and missed_areas:
        ratio = (matched_areas[len(matched_areas) // 2] / missed_areas[len(missed_areas) // 2]
                 if missed_areas[len(missed_areas) // 2] else float("inf"))
        print(f"    matched median is {ratio:.1f}x the missed median")
        # How much of the miss count sits in the smallest area decile overall?
        every = sorted(matched_areas + missed_areas)
        cut = every[max(0, len(every) // 4 - 1)]
        small_missed = sum(1 for a in missed_areas if a <= cut)
        small_total = sum(1 for a in every if a <= cut)
        print(
            f"    smallest quartile (area <= {cut:.4f}): {small_missed}/{small_total} boxes missed"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "dataset_dir": args.dataset_dir,
                    "split": args.split,
                    "thresholds": thresholds,
                    "iou_threshold": args.iou_threshold,
                    "outcome": dict(outcome),
                    "per_grade": {k: dict(v) for k, v in per_grade.items()},
                    "confusion": dict(confusion),
                    "areas_by_outcome": {k: sorted(v) for k, v in areas_by_outcome.items()},
                    "examples": examples,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
