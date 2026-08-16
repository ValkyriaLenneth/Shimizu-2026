#!/usr/bin/env python3
"""Re-derive every number in FREEZE.md from the frozen threshold-grid CSVs.

Why this exists
---------------
FREEZE.md pins an operating point by its per-class threshold triple, not by a
remembered number. That only helps if the mapping from triple to metrics can be
recomputed on demand, so this script is the executable half of the freeze: it
reads the CSVs in ../results, re-selects the frozen rows, and prints the same
tables the document carries. Run it before quoting any figure from FREEZE.md.

It needs no GPU, no dataset and no checkpoints -- the grids were already scored
on the frozen test split by ensemble_wbf_eval.py / tta_wbf_eval.py, which write
one row per threshold triple. Reproducing the *scoring* (rather than the
selection) requires the missing assets listed in FREEZE.md section 5.

The grids themselves are not in git -- they live in the 2026-08-15 handoff
package, so point --results-dir at that package's results/ directory:

    python3 systems/rfdetr/scripts/verify_brace_freeze.py \
      --results-dir <handoff_20260815_brace_recall_freeze>/results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Class index -> grade, from
# systems/rfdetr/recognition_models/brace/configs/rfdetr_brace_baseline.yaml
GRADES = ["B", "C", "D"]

# The frozen points. Each is addressed by (csv, threshold triple) so the lookup
# is exact rather than a re-run of the original selection heuristic.
FROZEN = [
    ("FP-1 delivered 0.723", "br_brl_ens_tta.csv", "0.3,0.15,0.12"),
    ("FP-2 all-four >= 0.70", "br_brl_ens_tta.csv", "0.12,0.15,0.18"),
    ("FP-3 recall-max @P>=0.30", "br_brl_ens_tta.csv", "0.1,0.15,0.07"),
    ("REF no-BRL 2model+TTA", "br_ens2tta.csv", "0.4,0.2,0.3"),
]

TARGET = 0.70


def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def find(table: list[dict], triple: str) -> dict:
    for row in table:
        if row["thresholds"].strip('"') == triple:
            return row
    raise SystemExit(f"threshold triple {triple!r} not present in the grid")


def describe(row: dict) -> dict:
    out = {
        "thresholds": row["thresholds"].strip('"'),
        "overall_recall": float(row["recall"]),
        "overall_precision": float(row["precision"]),
        "overall_f1": float(row["f1"]),
    }
    for i, g in enumerate(GRADES):
        tp = int(row[f"class_{i}_tp"])
        fn = int(row[f"class_{i}_fn"])
        out[g] = {
            "recall": float(row[f"class_{i}_recall"]),
            "precision": float(row[f"class_{i}_precision"]),
            "tp": tp,
            "total": tp + fn,
        }
    return out


def ceilings(table: list[dict]) -> dict:
    """Highest recall each grade reaches anywhere in the grid.

    A grade whose ceiling sits just above the target has no margin: the boxes it
    never recovers are invisible to threshold tuning, so only a better model
    moves them.
    """
    out = {}
    for i, g in enumerate(GRADES):
        best = max(table, key=lambda r: float(r[f"class_{i}_recall"]))
        tp = int(best[f"class_{i}_tp"])
        out[g] = {
            "ceiling": float(best[f"class_{i}_recall"]),
            "tp": tp,
            "total": tp + int(best[f"class_{i}_fn"]),
        }
    out["overall"] = {"ceiling": max(float(r["recall"]) for r in table)}
    return out


def feasible(table: list[dict]) -> tuple[int, dict | None]:
    """Points where all three grades and the overall recall clear the target."""
    ok = [
        r for r in table
        if float(r["recall"]) >= TARGET
        and all(float(r[f"class_{i}_recall"]) >= TARGET for i in range(3))
    ]
    if not ok:
        return 0, None
    return len(ok), max(ok, key=lambda r: float(r["precision"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="results/ of the 2026-08-15 brace freeze handoff package")
    ap.add_argument("--write-json", default=None)
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    if not (rdir / "br_brl_ens_tta.csv").exists():
        raise SystemExit(f"{rdir} does not hold br_brl_ens_tta.csv; see the module docstring")

    report: dict = {"target": TARGET, "frozen_points": {}, "grids": {}}

    print("== frozen operating points ==")
    header = f"{'point':26s} {'thr(B,C,D)':>15} {'R':>6} {'P':>6}   B          C          D"
    print(header)
    for name, fname, triple in FROZEN:
        d = describe(find(rows(rdir / fname), triple))
        report["frozen_points"][name] = {"csv": fname, **d}
        cells = " ".join(
            f"{d[g]['recall']:.3f}({d[g]['tp']:>2}/{d[g]['total']:<2})" for g in GRADES
        )
        print(f"{name:26s} {d['thresholds']:>15} {d['overall_recall']:>6.3f} "
              f"{d['overall_precision']:>6.3f}  {cells}")

    print("\n== per-grade recall ceiling over the whole grid ==")
    for fname in ("br_brl_ens_tta.csv", "br_ens2tta.csv"):
        table = rows(rdir / fname)
        c = ceilings(table)
        n, best = feasible(table)
        report["grids"][fname] = {
            "ceilings": c,
            "feasible_points": n,
            "best_feasible": describe(best) if best else None,
        }
        cells = " ".join(
            f"{g}={c[g]['ceiling']:.3f}({c[g]['tp']}/{c[g]['total']})" for g in GRADES
        )
        print(f"{fname:22s} {cells}  overall={c['overall']['ceiling']:.3f}")
        if best:
            print(f"{'':22s} all-four>={TARGET:.2f}: {n} points, "
                  f"best precision {float(best['precision']):.3f}")
        else:
            print(f"{'':22s} all-four>={TARGET:.2f}: no feasible point")

    if args.write_json:
        Path(args.write_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.write_json}")


if __name__ == "__main__":
    main()
