#!/usr/bin/env python3
"""Summarize the new-class RF-DETR runs against the client recall requirement.

Two modes:

* default - print the recall-first picture per run: best recall epoch, whether it
  clears the 0.80 requirement, and whether recall has stopped improving, which is
  the signal to stop a run early rather than walk it to the epoch ceiling.
* ``--list-top-checkpoints RUN_DIR`` - print the top-N epoch checkpoint paths
  ranked by swept ``test/recall``, for feeding into the per-class threshold grid.

Ranking uses ``test_results.csv`` from ``sweep_rfdetr_router_test.py`` when it
exists, because that is the recall-first ranking after checkpoint reload. The
automatic ``checkpoint_best_total.pth`` is selected by mAP and is deliberately
ignored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TARGET_RECALL = 0.80
PRECISION_FLOOR = 0.60
RUN_ROOT = Path("outputs/rfdetr_single_crack")
CATEGORIES = ("brace", "column_base")
EXPERIMENT = "medium"  # default; override with --experiment
SUFFIX = "bcd_20260725_split91_test_as_valid"  # default; override with --suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--suffix", default=SUFFIX)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--list-top-checkpoints", default="", help="run dir to list candidates for")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--stall-epochs",
        type=int,
        default=25,
        help="Flag a run as stalled when recall has not improved for this many epochs.",
    )
    return parser.parse_args()


def number(row: dict[str, str], key: str) -> float | None:
    """metrics.csv rows are written incrementally, so fields can be blank."""
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_rows(path: Path, recall_key: str, prefix: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        recall = number(row, recall_key)
        epoch = number(row, "epoch")
        if recall is None or epoch is None:
            continue
        rows.append(
            {
                "epoch": int(epoch),
                "recall": recall,
                "precision": number(row, f"{prefix}/precision") or 0.0,
                "map50": number(row, f"{prefix}/mAP_50") or 0.0,
            }
        )
    return rows


def load_run(run_dir: Path) -> tuple[list[dict[str, float]], str]:
    """Prefer the post-training sweep; fall back to per-epoch val metrics."""
    sweep = run_dir / "test_results.csv"
    if sweep.exists():
        rows = load_rows(sweep, "test/recall", "test")
        if rows:
            return rows, "sweep(test)"
    metrics = run_dir / "metrics.csv"
    if metrics.exists():
        return load_rows(metrics, "val/recall", "val"), "val"
    return [], "none"


def list_top_checkpoints(run_dir: Path, top: int) -> int:
    rows, source = load_run(run_dir)
    if not rows or source != "sweep(test)":
        return 1
    ranked = sorted(rows, key=lambda r: (-r["recall"], -r["precision"]))[:top]
    for row in ranked:
        candidate = run_dir / "epoch_pth" / f"checkpoint_epoch_{row['epoch']:03d}.pth"
        if candidate.exists():
            print(candidate)
    return 0


def main() -> int:
    args = parse_args()

    if args.list_top_checkpoints:
        return list_top_checkpoints(Path(args.list_top_checkpoints), args.top)

    root = Path(args.run_root)
    print(f"target: recall >= {TARGET_RECALL:.2f} (recall-first), precision floor {PRECISION_FLOOR:.2f}")
    print("delivered reference: tenjo R 0.875 | rc_wall 0.812 | inner_wall 0.848 | rc_column 0.826\n")

    for category in CATEGORIES:
        run_dir = root / f"{category}_{args.experiment}_{args.suffix}"
        rows, source = load_run(run_dir)
        if not rows:
            print(f"{category:12s} {args.experiment:6s}  no completed epoch yet")
            continue

        best = max(rows, key=lambda r: r["recall"])
        last = rows[-1]
        since_best = last["epoch"] - best["epoch"]
        verdict = "MEETS TARGET" if best["recall"] >= TARGET_RECALL else "below target"
        if best["recall"] >= TARGET_RECALL and best["precision"] < PRECISION_FLOOR:
            verdict += " (precision under floor)"
        stalled = "  STALLED" if since_best >= args.stall_epochs else ""

        print(
            f"{category:12s} {args.experiment:6s}  [{source}] epochs={len(rows):3d}  "
            f"best recall={best['recall']:.3f} @ep{best['epoch']:<3d} "
            f"prec={best['precision']:.3f} mAP50={best['map50']:.3f}  "
            f"({since_best} since best){stalled}  {verdict}"
        )
        step = max(1, len(rows) // 8)
        marks = [r for i, r in enumerate(rows) if i % step == 0][:9]
        print(f"{'':21s}curve: " + "  ".join(f"e{r['epoch']}:{r['recall']:.2f}" for r in marks))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
