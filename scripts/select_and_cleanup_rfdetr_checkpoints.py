#!/usr/bin/env python3
"""Select RF-DETR checkpoint candidates and optionally delete the rest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ALWAYS_KEEP = {
    "checkpoint_best_total.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_ema.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metrics-file", default="test_results.csv")
    parser.add_argument("--metric", default="test/precision")
    parser.add_argument("--secondary-metric", default="test/mAP_50")
    parser.add_argument("--recall-metric", default="test/recall")
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument(
        "--keep-epochs",
        default="",
        help="Comma-separated epoch ids to keep in addition to metric-selected epochs.",
    )
    parser.add_argument("--keep-last", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-json", default="")
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_epoch_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            epoch_value = parse_float(raw.get("epoch"))
            if epoch_value is None:
                continue
            epoch = int(epoch_value)
            metric_values = {key: parse_float(value) for key, value in raw.items()}
            if any(key.startswith(("test/", "val/")) for key in metric_values):
                rows[epoch] = {"epoch": epoch, **metric_values}
    return [rows[k] for k in sorted(rows)]


def parse_epoch_list(value: str) -> set[int]:
    epochs: set[int] = set()
    if not value:
        return epochs
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        epochs.add(int(item))
    return epochs


def checkpoints_for_epoch(run_dir: Path, epoch: int) -> list[Path]:
    return [
        run_dir / "epoch_pth" / f"checkpoint_epoch_{epoch:03d}.pth",
        run_dir / f"checkpoint_{epoch}.ckpt",
        run_dir / f"checkpoint_epoch={epoch}.ckpt",
    ]


def choose_epochs(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get(args.metric) is not None
        and (row.get(args.recall_metric) is None or row[args.recall_metric] >= args.min_recall)
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row.get(args.metric) if row.get(args.metric) is not None else -1.0,
            row.get(args.secondary_metric) if row.get(args.secondary_metric) is not None else -1.0,
            row.get(args.recall_metric) if row.get(args.recall_metric) is not None else -1.0,
        ),
        reverse=True,
    )
    return ranked[: args.top_k]


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / args.metrics_file
    rows = load_epoch_metrics(metrics_path)
    selected_rows = choose_epochs(rows, args)
    selected_epochs = {int(row["epoch"]) for row in selected_rows}
    selected_epochs.update(parse_epoch_list(args.keep_epochs))

    if args.keep_last and rows:
        selected_epochs.add(int(rows[-1]["epoch"]))

    keep_paths = {run_dir / name for name in ALWAYS_KEEP if (run_dir / name).exists()}
    for epoch in selected_epochs:
        for ckpt in checkpoints_for_epoch(run_dir, epoch):
            if ckpt.exists():
                keep_paths.add(ckpt)

    all_checkpoints = sorted(
        [
            *run_dir.glob("checkpoint_*.ckpt"),
            *run_dir.glob("checkpoint_epoch=*.ckpt"),
            *run_dir.glob("checkpoint_best_*.pth"),
            *(run_dir / "epoch_pth").glob("checkpoint_epoch_*.pth"),
        ]
    )
    delete_paths = [path for path in all_checkpoints if path not in keep_paths]
    reclaimed_bytes = sum(path.stat().st_size for path in delete_paths if path.exists())

    summary = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "metric": args.metric,
        "secondary_metric": args.secondary_metric,
        "recall_metric": args.recall_metric,
        "min_recall": args.min_recall,
        "explicit_keep_epochs": sorted(parse_epoch_list(args.keep_epochs)),
        "selected_epochs": sorted(selected_epochs),
        "selected_rows": selected_rows,
        "keep": [str(path) for path in sorted(keep_paths)],
        "delete": [str(path) for path in delete_paths],
        "reclaimed_bytes": reclaimed_bytes if not args.dry_run else 0,
        "would_reclaim_bytes": reclaimed_bytes,
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        for path in delete_paths:
            path.unlink()

    out_path = Path(args.write_json) if args.write_json else run_dir / "checkpoint_selection_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
