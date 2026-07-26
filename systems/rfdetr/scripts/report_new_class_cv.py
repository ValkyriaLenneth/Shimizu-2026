#!/usr/bin/env python3
"""Pool the 5-fold cross-validation results into one unbiased estimate.

Sums tp/fp/fn over all folds at the fixed per-category thresholds, so precision,
recall and F1 rest on the full 477 / 320 boxes rather than the 39 / 38 of a single
split. Also prints per-fold numbers, because the spread across folds is the honest
measure of how much a single-split number can be trusted.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

GRADES = {0: "B", 1: "C", 2: "D"}
CATEGORIES = {"brace": "ブレース", "column_base": "柱脚"}
# Fixed thresholds carried over from the 9:1 crop2+lr3e-5 runs, matching run_new_classes_cv.sh
FIXED = {"brace": (0.3, 0.35, 0.4), "column_base": (0.25, 0.5, 0.45)}

SINGLE_SPLIT = {
    "brace": {"f1": 0.635, "recall": 0.692, "precision": 0.587},
    "column_base": {"f1": 0.507, "recall": 0.447, "precision": 0.586},
}
DELIVERED = [
    ("天井", 0.650, 0.812, 0.722),
    ("RC壁", 0.722, 0.812, 0.765),
    ("内壁", 0.811, 0.909, 0.857),
    ("RC柱", 0.661, 0.826, 0.735),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/rfdetr_single_crack/cv")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def metric(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def read_fold(path: Path, thresholds: tuple[float, float, float]) -> dict | None:
    """Pull the single row matching the fixed threshold triple."""
    if not path.exists():
        return None
    want = tuple(round(float(v), 6) for v in thresholds)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        try:
            got = (
                round(float(row["threshold_class_0"]), 6),
                round(float(row["threshold_class_1"]), 6),
                round(float(row["threshold_class_2"]), 6),
            )
        except (KeyError, ValueError):
            continue
        if got != want:
            continue
        out = {"tp": 0, "fp": 0, "fn": 0, "per_grade": {}}
        for index, grade in GRADES.items():
            tp = int(float(row.get(f"class_{index}_tp", 0) or 0))
            fp = int(float(row.get(f"class_{index}_fp", 0) or 0))
            fn = int(float(row.get(f"class_{index}_fn", 0) or 0))
            out["per_grade"][grade] = {"tp": tp, "fp": fp, "fn": fn}
            out["tp"] += tp
            out["fp"] += fp
            out["fn"] += fn
        return out
    return None


def read_best_epoch(run_dir: Path, thresholds: tuple[float, float, float]) -> dict | None:
    """Deliberately biased upper bound: best epoch per fold at the same thresholds.

    Reported next to the unbiased fixed-epoch figure so the pair brackets the truth.
    Selecting the epoch on the fold's own test is exactly the bias the primary
    number avoids, which is why this is labelled and never used on its own.
    """
    best = None
    for path in sorted(run_dir.glob("cv_grid_epoch_*.csv")):
        data = read_fold(path, thresholds)
        if not data:
            continue
        _, _, f1 = metric(data["tp"], data["fp"], data["fn"])
        if best is None or f1 > best[0]:
            best = (f1, path.stem.replace("cv_grid_epoch_", ""), data)
    if best is None:
        return None
    return {"epoch": best[1], **best[2]}


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)

    print("=" * 92)
    print("5-fold cross-validation: fixed recipe / a-priori fixed epoch / fixed thresholds")
    print("pooled over folds, so the estimate is out-of-fold rather than selected-on-test")
    print("=" * 92)

    for category, label in CATEGORIES.items():
        thresholds = FIXED[category]
        print(f"\n### {label} ({category})   fixed B/C/D thresholds {thresholds}")
        folds = []
        for fold in range(args.folds):
            data = read_fold(root / f"{category}_fold{fold}" / "cv_grid.csv", thresholds)
            folds.append(data)

        done = [f for f in folds if f]
        if not done:
            print("  no fold results yet")
            continue

        print(f"  {'fold':<6} {'tp':>4} {'fp':>4} {'fn':>4} {'prec':>7} {'recall':>7} {'F1':>7}")
        f1s = []
        for index, data in enumerate(folds):
            if not data:
                print(f"  {index:<6} {'-':>4} {'-':>4} {'-':>4} {'pending':>7}")
                continue
            precision, recall, f1 = metric(data["tp"], data["fp"], data["fn"])
            f1s.append(f1)
            print(
                f"  {index:<6} {data['tp']:>4} {data['fp']:>4} {data['fn']:>4} "
                f"{precision:>7.3f} {recall:>7.3f} {f1:>7.3f}"
            )

        tp = sum(f["tp"] for f in done)
        fp = sum(f["fp"] for f in done)
        fn = sum(f["fn"] for f in done)
        precision, recall, f1 = metric(tp, fp, fn)
        print(f"  {'POOLED':<6} {tp:>4} {fp:>4} {fn:>4} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")

        if len(f1s) > 1:
            mean = sum(f1s) / len(f1s)
            spread = (sum((x - mean) ** 2 for x in f1s) / (len(f1s) - 1)) ** 0.5
            print(f"  per-fold F1 mean {mean:.3f} +/- {spread:.3f} (sd), min {min(f1s):.3f}, max {max(f1s):.3f}")
            print(f"  -> a single split can land anywhere in that range purely by which images it got")

        print("  per-grade (pooled):")
        for grade in GRADES.values():
            g_tp = sum(f["per_grade"][grade]["tp"] for f in done)
            g_fp = sum(f["per_grade"][grade]["fp"] for f in done)
            g_fn = sum(f["per_grade"][grade]["fn"] for f in done)
            g_p, g_r, g_f1 = metric(g_tp, g_fp, g_fn)
            print(
                f"    {grade}: tp={g_tp:>3} fp={g_fp:>4} fn={g_fn:>3}  "
                f"P={g_p:.3f} R={g_r:.3f} F1={g_f1:.3f}"
            )

        biased = [read_best_epoch(root / f"{category}_fold{i}", thresholds) for i in range(args.folds)]
        biased_done = [b for b in biased if b]
        if biased_done:
            b_tp = sum(b["tp"] for b in biased_done)
            b_fp = sum(b["fp"] for b in biased_done)
            b_fn = sum(b["fn"] for b in biased_done)
            b_p, b_r, b_f1 = metric(b_tp, b_fp, b_fn)
            picked = ", ".join(f"f{i}:ep{b['epoch']}" for i, b in enumerate(biased) if b)
            print(
                f"  best-epoch-per-fold (BIASED upper bound): P={b_p:.3f} R={b_r:.3f} F1={b_f1:.3f}"
            )
            print(f"    epochs picked on test: {picked}")
            print(f"    honest value sits between {f1:.3f} (unbiased) and {b_f1:.3f} (biased)")

        single = SINGLE_SPLIT.get(category)
        if single:
            print(
                f"  single 9:1 split reported F1 {single['f1']:.3f} "
                f"(R {single['recall']:.3f} / P {single['precision']:.3f}); "
                f"pooled CV F1 {f1:.3f} -> optimism {single['f1'] - f1:+.3f}"
            )

    print("\n" + "-" * 92)
    print("delivered models, for reference (their own test splits hold ~32 boxes each,")
    print("so their numbers carry the same kind of small-sample optimism):")
    print(f"  {'部材':<8} {'Prec':>6} {'Recall':>7} {'F1':>6}")
    for name, precision, recall, f1_value in DELIVERED:
        print(f"  {name:<8} {precision:>6.3f} {recall:>7.3f} {f1_value:>6.3f}")
    print("-" * 92)


if __name__ == "__main__":
    main()
