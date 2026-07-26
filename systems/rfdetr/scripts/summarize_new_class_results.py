#!/usr/bin/env python3
"""Overall-performance report for the two new downstream categories.

Leads with threshold-free and balanced metrics - mAP and best F1 - because those
say whether the model is actually good, whereas a recall number can always be
bought by lowering thresholds until precision collapses. The recall-first
operating point is still printed, but last.

Sources, both produced by the established downstream selection procedure:

* ``test_results.csv`` from ``sweep_rfdetr_router_test.py`` - every saved epoch
  checkpoint reloaded and force-evaluated on the official test split. The
  automatic ``checkpoint_best_total.pth`` is chosen by mAP over a different eval
  path and is deliberately ignored here.
* ``class_threshold_grid_*.csv`` from ``evaluate_rfdetr_class_threshold_grid.py``
  at match IoU 0.229 - per-class B/C/D thresholds, which is the protocol the
  delivered client numbers use and therefore the only comparable one.

For each checkpoint three operating points are reported:

    best F1            - the honest "how good is this model" point
    max R at P>=floor  - the most recall obtainable while precision stays usable
    recall-first       - highest recall reaching the target, else highest recall
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TARGET_RECALL = 0.80
PRECISION_FLOOR = 0.60
GRADES = ("B", "C", "D")
CATEGORIES = {"brace": "ブレース", "column_base": "柱脚"}
EXPERIMENT = "medium"  # default; override with --experiment
SUFFIX = "bcd_20260725_split91_test_as_valid"  # default; override with --suffix

# Delivered reference models, from final_release_20260615/.../selected_thresholds.csv
# plus the mAP50 recorded in the per-category manifests where available.
DELIVERED = [
    ("天井", 0.650, 0.812, 0.722, None, "0.25/0.35/0.35"),
    ("RC壁", 0.722, 0.812, 0.765, None, "0.28/0.45/0.25"),
    ("内壁", 0.811, 0.909, 0.857, 0.7842, "0.25/0.40/0.40"),
    ("RC柱", 0.661, 0.826, 0.735, 0.7255, "-"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/rfdetr_single_crack")
    parser.add_argument("--suffix", default=SUFFIX)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--precision-floor", type=float, default=PRECISION_FLOOR)
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_sweep(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "test_results.csv"
    if not path.exists():
        return []
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        epoch = to_float(row.get("epoch"))
        recall = to_float(row.get("test/recall"))
        if epoch is None or recall is None:
            continue
        precision = to_float(row.get("test/precision")) or 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "epoch": int(epoch),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "map50": to_float(row.get("test/mAP_50")) or 0.0,
                "map5095": to_float(row.get("test/mAP_50_95")) or 0.0,
            }
        )
    return rows


def read_grid(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        recall = to_float(row.get("recall"))
        precision = to_float(row.get("precision"))
        if recall is None or precision is None:
            continue
        entry: dict[str, float | str] = {
            "thresholds": row.get("thresholds", ""),
            "recall": recall,
            "precision": precision,
            "f1": to_float(row.get("f1")) or 0.0,
        }
        for index, grade in enumerate(GRADES):
            entry[grade] = to_float(row.get(f"class_{index}_recall")) or 0.0
        rows.append(entry)
    return rows


def operating_points(rows: list[dict[str, float | str]], floor: float) -> dict[str, dict | None]:
    """best-F1, max-recall-at-precision-floor, and recall-first points."""
    if not rows:
        return {"best_f1": None, "at_floor": None, "recall_first": None}
    best_f1 = max(rows, key=lambda r: float(r["f1"]))
    usable = [r for r in rows if float(r["precision"]) >= floor]
    at_floor = max(usable, key=lambda r: float(r["recall"])) if usable else None
    qualifying = [r for r in rows if float(r["recall"]) >= TARGET_RECALL]
    if qualifying:
        recall_first = max(qualifying, key=lambda r: float(r["precision"]))
    else:
        recall_first = max(rows, key=lambda r: float(r["recall"]))
    return {"best_f1": best_f1, "at_floor": at_floor, "recall_first": recall_first}


def fmt(point: dict | None, floor: float) -> str:
    if point is None:
        return f"      (no grid point reaches precision {floor:.2f})"
    return (
        f"      thr {str(point['thresholds']):<16} "
        f"F1={float(point['f1']):.3f}  R={float(point['recall']):.3f}  P={float(point['precision']):.3f}  "
        f"B/C/D={float(point['B']):.2f}/{float(point['C']):.2f}/{float(point['D']):.2f}"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    floor = args.precision_floor

    print("=" * 88)
    print(f"ブレース / 柱脚 downstream RF-DETR - overall performance (experiment: {args.experiment})")
    print(f"dataset suffix: {args.suffix}")
    print("=" * 88)

    for category, label in CATEGORIES.items():
        run_dir = root / f"{category}_{args.experiment}_{args.suffix}"
        print(f"\n### {label} ({category})")

        sweep = read_sweep(run_dir)
        if not sweep:
            print("  sweep: test_results.csv missing - training or sweep did not finish")
        else:
            print(f"  sweep over {len(sweep)} checkpoints. Ranked by mAP50 (threshold-free):")
            print(f"    {'epoch':>6} {'mAP50':>7} {'mAP50-95':>9} {'F1':>7} {'recall':>7} {'prec':>7}")
            for row in sorted(sweep, key=lambda r: -r["map50"])[: args.top]:
                print(
                    f"    {row['epoch']:>6} {row['map50']:>7.3f} {row['map5095']:>9.3f} "
                    f"{row['f1']:>7.3f} {row['recall']:>7.3f} {row['precision']:>7.3f}"
                )
            best_map = max(sweep, key=lambda r: r["map50"])
            best_f1_sweep = max(sweep, key=lambda r: r["f1"])
            print(
                f"    best mAP50 ep{best_map['epoch']} = {best_map['map50']:.3f}   "
                f"best sweep F1 ep{best_f1_sweep['epoch']} = {best_f1_sweep['f1']:.3f}"
            )

        grids = sorted(run_dir.glob("class_threshold_grid_*.csv"))
        if not grids:
            print("  per-class threshold grid: not produced yet")
            continue

        print(f"  per-class threshold grid at match IoU 0.229, precision floor {floor:.2f}:")
        overall_best_f1 = None
        for path in grids:
            tag = path.stem.replace("class_threshold_grid_", "")
            rows = read_grid(path)
            if not rows:
                continue
            points = operating_points(rows, floor)
            print(f"    {tag}  ({len(rows)} grid points)")
            print(f"      -- best F1")
            print(fmt(points["best_f1"], floor))
            print(f"      -- max recall at P>={floor:.2f}")
            print(fmt(points["at_floor"], floor))
            print(f"      -- recall-first")
            print(fmt(points["recall_first"], floor))
            if points["best_f1"] and (
                overall_best_f1 is None or float(points["best_f1"]["f1"]) > float(overall_best_f1[1]["f1"])
            ):
                overall_best_f1 = (tag, points["best_f1"])

        if overall_best_f1:
            tag, point = overall_best_f1
            print(
                f"  => headline: {tag} at thr {point['thresholds']} -> "
                f"F1 {float(point['f1']):.3f}, R {float(point['recall']):.3f}, P {float(point['precision']):.3f}"
            )

    print("\n" + "-" * 88)
    print("delivered reference models (the target to match):")
    print(f"  {'部材':<8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'mAP50':>7}  {'B/C/D thr':<16}")
    for name, precision, recall, f1, map50, thr in DELIVERED:
        map_text = f"{map50:.3f}" if map50 is not None else "   -  "
        print(f"  {name:<8} {precision:>6.3f} {recall:>7.3f} {f1:>6.3f} {map_text:>7}  {thr:<16}")
    print("  F1 band to beat: 0.72 - 0.86;  mAP50 reference: 0.73 - 0.78")
    print("-" * 88)


if __name__ == "__main__":
    main()
