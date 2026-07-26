#!/usr/bin/env python3
"""Leaderboard for the fixed-split iteration bench.

Ranks every experiment recorded in the bench results table, per category, so a
run of experiments can be read at a glance.

Two guards are printed with the table rather than left implicit:

* **Resolution floor.** Fold-to-fold spread was measured at 0.070 F1, so on a single
  split a gap smaller than about 0.05 is not evidence of anything. Rows within that
  band of the leader are marked as tied.
* **Selection creep.** Every experiment scored against the same test set makes the
  best-so-far figure a little more optimistic. The count of experiments is shown so
  the size of that effect stays visible.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

RESOLUTION_FLOOR = 0.05
CATEGORY_LABELS = {"brace": "ブレース", "column_base": "柱脚"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/rfdetr_new_classes/bench/results.csv")
    parser.add_argument("--sort", default="f1", choices=["f1", "recall", "precision"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.results)
    if not path.exists():
        print(f"no results yet at {path}")
        return

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print("results table is empty")
        return

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    print("=" * 104)
    print("fixed-split iteration bench - fold 3, test 117 boxes (ブレース) / 60 boxes (柱脚)")
    print(f"differences below {RESOLUTION_FLOOR:.2f} F1 are not resolvable on a single split")
    print("=" * 104)

    for category, label in CATEGORY_LABELS.items():
        entries = by_category.get(category, [])
        if not entries:
            continue
        # Keep the latest row per experiment name, so a re-run supersedes its predecessor.
        latest: dict[str, dict] = {}
        for row in entries:
            latest[row["name"]] = row
        ranked = sorted(latest.values(), key=lambda r: -float(r[args.sort]))
        best = float(ranked[0][args.sort])

        print(f"\n### {label} ({category})   {len(latest)} experiments")
        print(
            f"  {'name':<16} {'view':<22} {'aug':<26} {'F1':>7} {'P':>7} {'R':>7} "
            f"{'B':>6} {'C':>6} {'D':>6}"
        )
        print("  " + "-" * 102)
        for row in ranked:
            gap = best - float(row[args.sort])
            mark = " " if gap == 0 else ("~" if gap < RESOLUTION_FLOOR else " ")
            view = row["view"].replace("_20260725", "")
            aug = row["aug"] if row["aug"] != "none" else "-"
            print(
                f" {mark}{row['name']:<16} {view:<22} {aug:<26} "
                f"{float(row['f1']):>7.3f} {float(row['precision']):>7.3f} {float(row['recall']):>7.3f} "
                f"{float(row['B_recall']):>6.3f} {float(row['C_recall']):>6.3f} {float(row['D_recall']):>6.3f}"
            )
        tied = [r["name"] for r in ranked if 0 < best - float(r[args.sort]) < RESOLUTION_FLOOR]
        if tied:
            print(f"  ~ within noise of the leader, not distinguishable: {', '.join(tied)}")

    total = len({r["name"] for r in rows})
    print("\n" + "-" * 104)
    print(f"{total} distinct experiments scored against this one test set.")
    print("The leader's figure carries selection optimism that grows with that count;")
    print("confirm any winner with run_new_classes_cv.sh before reporting it.")
    print("-" * 104)


if __name__ == "__main__":
    main()
