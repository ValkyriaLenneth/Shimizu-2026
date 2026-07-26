#!/usr/bin/env python3
"""One table comparing every ブレース / 柱脚 experiment on the same footing.

Discovers run directories under the run root, groups them by category, and reports
the metrics that matter for overall performance: threshold-free mAP from the epoch
sweep, and the best-F1 operating point from the per-class threshold grid at match
IoU 0.229.

Runs still in progress fall back to in-training ``val`` metrics and are marked, so
a partial experiment can be read without pretending it is final. Experiments on
different test splits are grouped separately, because recall and F1 are not
comparable across splits.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

CATEGORIES = {"brace": "ブレース", "column_base": "柱脚"}
GRADES = ("B", "C", "D")

DELIVERED_BAND = "delivered band: F1 0.72-0.86, mAP50 0.73-0.78"

# Human labels for the dataset suffixes we have run, so the table reads clearly.
SPLIT_LABELS = {
    "bcd_20260725_test_as_valid": ("8:2", "base"),
    "bcd_20260725_split91_test_as_valid": ("9:1", "base"),
    "bcd_20260725_split91_crop2_test_as_valid": ("9:1", "crop2"),
    "jointft_bcd_20260725_split91_crop2_test_as_valid": ("9:1", "crop2 + joint-ft"),
    "lr3e5_bcd_20260725_split91_crop3_test_as_valid": ("9:1", "crop3 + lr3e-5"),
    "lr3e5_bcd_20260725_split91_crop2_test_as_valid": ("9:1", "crop2 + lr3e-5"),
    "lr1e5_bcd_20260725_split91_crop3_test_as_valid": ("9:1", "crop3 + lr1e-5"),
    "bcd_20260725_split91_crop3_test_as_valid": ("9:1", "crop3"),
    "lr3e5_bcd_20260725_split91_crop2dboost_test_as_valid": ("9:1", "crop2+Dboost lr3e-5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/rfdetr_single_crack")
    parser.add_argument("--precision-floor", type=float, default=0.60)
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def f1_of(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def sweep_best(run_dir: Path) -> dict | None:
    path = run_dir / "test_results.csv"
    if not path.exists():
        return None
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        epoch = to_float(row.get("epoch"))
        recall = to_float(row.get("test/recall"))
        if epoch is None or recall is None:
            continue
        precision = to_float(row.get("test/precision")) or 0.0
        rows.append(
            {
                "epoch": int(epoch),
                "map50": to_float(row.get("test/mAP_50")) or 0.0,
                "map5095": to_float(row.get("test/mAP_50_95")) or 0.0,
                "f1": f1_of(precision, recall),
            }
        )
    if not rows:
        return None
    best = max(rows, key=lambda r: r["map50"])
    return {"n": len(rows), "source": "test", **best}


def val_best(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        epoch = to_float(row.get("epoch"))
        recall = to_float(row.get("val/recall"))
        if epoch is None or recall is None:
            continue
        precision = to_float(row.get("val/precision")) or 0.0
        rows.append(
            {
                "epoch": int(epoch),
                "map50": to_float(row.get("val/mAP_50")) or 0.0,
                "map5095": to_float(row.get("val/mAP_50_95")) or 0.0,
                "f1": f1_of(precision, recall),
            }
        )
    if not rows:
        return None
    best = max(rows, key=lambda r: r["map50"])
    return {"n": len(rows), "source": "val", **best}


def grid_best(run_dir: Path, floor: float) -> dict | None:
    """Best-F1 grid point across all graded checkpoints, plus the P>=floor point."""
    best_f1: dict | None = None
    best_at_floor: dict | None = None
    for path in sorted(run_dir.glob("class_threshold_grid_*.csv")):
        for row in csv.DictReader(path.open(encoding="utf-8")):
            recall = to_float(row.get("recall"))
            precision = to_float(row.get("precision"))
            f1 = to_float(row.get("f1"))
            if recall is None or precision is None or f1 is None:
                continue
            entry = {
                "checkpoint": path.stem.replace("class_threshold_grid_", ""),
                "thresholds": row.get("thresholds", ""),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
            for index, grade in enumerate(GRADES):
                entry[grade] = to_float(row.get(f"class_{index}_recall")) or 0.0
            if best_f1 is None or f1 > best_f1["f1"]:
                best_f1 = entry
            if precision >= floor and (best_at_floor is None or recall > best_at_floor["recall"]):
                best_at_floor = entry
    if best_f1 is None:
        return None
    return {"best_f1": best_f1, "at_floor": best_at_floor}


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    floor = args.precision_floor

    pattern = re.compile(r"^(brace|column_base)_medium_(.+)$")
    found: dict[str, list[tuple[str, Path]]] = {c: [] for c in CATEGORIES}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            found[match.group(1)].append((match.group(2), path))

    print("=" * 104)
    print("ブレース / 柱脚 experiment comparison - overall performance")
    print(f"{DELIVERED_BAND}   |   precision floor {floor:.2f}   |   grid match IoU 0.229")
    print("=" * 104)

    for category, label in CATEGORIES.items():
        print(f"\n### {label} ({category})")
        if not found[category]:
            print("  no runs found")
            continue
        header = (
            f"  {'split':<5} {'variant':<18} {'src':<5} {'eps':>4} "
            f"{'mAP50':>7} {'mAP@.5:.95':>10} {'bestF1':>7} {'R':>6} {'P':>6} {'B/C/D recall':>16}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for suffix, run_dir in found[category]:
            split, variant = SPLIT_LABELS.get(suffix, ("?", suffix))
            sweep = sweep_best(run_dir) or val_best(run_dir)
            grid = grid_best(run_dir, floor)
            if sweep is None:
                print(f"  {split:<5} {variant:<18} {'-':<5} {'-':>4}   (no metrics yet)")
                continue
            marker = sweep["source"]
            if grid:
                point = grid["best_f1"]
                bcd = f"{point['B']:.2f}/{point['C']:.2f}/{point['D']:.2f}"
                print(
                    f"  {split:<5} {variant:<18} {marker:<5} {sweep['n']:>4} "
                    f"{sweep['map50']:>7.3f} {sweep['map5095']:>10.3f} "
                    f"{point['f1']:>7.3f} {point['recall']:>6.3f} {point['precision']:>6.3f} {bcd:>16}"
                )
                if grid["at_floor"]:
                    at = grid["at_floor"]
                    print(
                        f"  {'':<5} {'':<18} {'':<5} {'':>4} {'':>7} {'':>10} "
                        f"  -> at P>={floor:.2f}: R={at['recall']:.3f} P={at['precision']:.3f} "
                        f"F1={at['f1']:.3f} thr {at['thresholds']}"
                    )
                else:
                    print(
                        f"  {'':<5} {'':<18} {'':<5} {'':>4} {'':>7} {'':>10} "
                        f"  -> no grid point reaches P>={floor:.2f}"
                    )
            else:
                print(
                    f"  {split:<5} {variant:<18} {marker:<5} {sweep['n']:>4} "
                    f"{sweep['map50']:>7.3f} {sweep['map5095']:>10.3f} "
                    f"{sweep['f1']:>7.3f} {'-':>6} {'-':>6} {'grid pending':>16}"
                )
        print("  src=test: post-training epoch sweep on the official test split")
        print("  src=val : still training, in-training val metrics, not final")

    joint = root / "joint_bcd_medium_20260725_split91_crop2"
    if joint.is_dir():
        best = sweep_best(joint) or val_best(joint)
        if best:
            print(
                f"\n  [joint pretrain corpus, not deployed] {best['source']} "
                f"best mAP50 {best['map50']:.3f} @ep{best['epoch']} over {best['n']} epochs"
            )

    print("\n" + "-" * 104)
    print("Every metric here, mAP included, is only comparable within one split column:")
    print("the 8:2 and 9:1 test sets are different images (58/45 vs 29/21), so numbers")
    print("across split rows are not a like-for-like comparison. Within a split, mAP50 is")
    print("the preferred signal because it does not depend on where thresholds were placed.")
    print("Crop augmentation only alters train, so a crop2 run is scored on the same test")
    print("split as the 9:1 base run and the two are directly comparable.")
    print("-" * 104)


if __name__ == "__main__":
    main()
