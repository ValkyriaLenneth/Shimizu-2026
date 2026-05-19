#!/usr/bin/env python3
"""Summarize router training result CSV files."""

from __future__ import annotations

import csv
from pathlib import Path


RESULTS = {
    "full": Path("../coarse_router_yolov9/runs/train/gelan_c_router_3class_full_e50/results.csv"),
    "cleaned": Path("../coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50/results.csv"),
}


def load_last(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return rows[-1] if rows else None


def main() -> int:
    for name, path in RESULTS.items():
        row = load_last(path)
        print(f"[{name}] {path}")
        if row is None:
            print("  missing")
            continue
        print(
            "  epoch={epoch} precision={precision} recall={recall} mAP50={map50} mAP50-95={map5095}".format(
                epoch=row.get("epoch", "").strip(),
                precision=row.get("metrics/precision", "").strip(),
                recall=row.get("metrics/recall", "").strip(),
                map50=row.get("metrics/mAP_0.5", "").strip(),
                map5095=row.get("metrics/mAP_0.5:0.95", "").strip(),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

