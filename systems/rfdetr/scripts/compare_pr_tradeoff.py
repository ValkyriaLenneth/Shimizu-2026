#!/usr/bin/env python3
"""Compare experiments by their whole recall-precision tradeoff, not one point.

A single operating point hides the thing that matters here. The client requirement
is recall-first with precision merely "respectable", and the two categories were
already delivered at different precision floors because their curves have
different shapes (2026-07-26: relaxing brace from 0.60 to 0.50 buys +0.096 recall
and F1 *rises*, while the same relaxation on column_base buys one box). An
intervention that lifts recall by 0.05 while costing 0.08 precision may be
exactly what this project wants, or worthless, depending on where the curve sits
- and reporting only "recall at P>=0.60" cannot tell the two apart.

So this prints, for every experiment, the maximum recall achievable at each of
several precision floors, computed over the union of every threshold combination
scored for that experiment. Where an experiment has no point at a floor, that is
itself the finding: the curve never reaches there.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

FLOORS = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", action="append", required=True,
                   metavar="LABEL=GLOB",
                   help="experiment label and a glob of its threshold-grid CSVs, repeatable")
    p.add_argument("--title", default="")
    p.add_argument("--floors", default=",".join(str(f) for f in FLOORS))
    return p.parse_args()


def rows_for(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        try:
            out += list(csv.DictReader(open(path, encoding="utf-8")))
        except OSError:
            continue
    return out


def main() -> int:
    args = parse_args()
    floors = [float(x) for x in args.floors.split(",") if x.strip()]
    specs = []
    for s in args.spec:
        if "=" not in s:
            raise SystemExit(f"--spec expects LABEL=GLOB, got {s!r}")
        label, pattern = s.split("=", 1)
        specs.append((label, rows_for(pattern)))

    if args.title:
        print(f"\n{args.title}")
    print("=" * (26 + 9 * len(floors)))
    header = f"{'实验':<24}" + "".join(f"{'P>=' + f'{f:.2f}':>9}" for f in floors)
    print(header)
    print("-" * (26 + 9 * len(floors)))
    for label, rows in specs:
        if not rows:
            print(f"{label:<24}" + "".join(f"{'-':>9}" for _ in floors))
            continue
        cells = []
        for f in floors:
            ok = [r for r in rows if float(r["precision"]) >= f]
            if ok:
                b = max(ok, key=lambda r: float(r["recall"]))
                cells.append(f"{float(b['recall']):>9.3f}")
            else:
                cells.append(f"{'—':>9}")
        print(f"{label:<24}" + "".join(cells))
    print("-" * (26 + 9 * len(floors)))
    print("  '—' = 该实验的 PR 曲线从未达到该精度")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
