#!/usr/bin/env python3
"""Summarize router epoch tradeoffs against a base model.

The coarse router is used to choose downstream models, so this report gives
Precision-aware and Recall-aware rankings instead of relying on YOLO fitness.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-results",
        default="coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv2_gemini_nb2_mix_e50_b48_lowlr/test_results.csv",
    )
    parser.add_argument("--base-log", default="outputs/router_eval/augv3_epoch14_test.log")
    parser.add_argument("--base-name", default="base_v3_epoch14")
    parser.add_argument(
        "--run-dir",
        default="coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv2_gemini_nb2_mix_e50_b48_lowlr",
    )
    parser.add_argument("--out-md", default="outputs/router_eval/gemini_mix_e50_tradeoff_summary.md")
    parser.add_argument("--out-csv", default="outputs/router_eval/gemini_mix_e50_tradeoff_summary.csv")
    parser.add_argument("--recall-tolerance", type=float, default=0.005)
    parser.add_argument("--hard-recall-drop", type=float, default=0.02)
    return parser.parse_args()


def parse_base_log(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if re.search(r"^\s*all\s+", line):
            parts = line.split()
            if len(parts) >= 7:
                return {
                    "precision": float(parts[3]),
                    "recall": float(parts[4]),
                    "map50": float(parts[5]),
                    "map50_95": float(parts[6]),
                }
    raise ValueError(f"failed to parse base metrics from {path}")


def load_epochs(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "map50": float(row["map50"]),
                    "map50_95": float(row["map50_95"]),
                }
            )
    return rows


def add_scores(row: dict[str, float], base: dict[str, float], args: argparse.Namespace) -> None:
    # Position is rescued by downstream expansion, but wrong-class routing is not.
    # Use several explicit policies so the tradeoff is visible.
    row["precision_first_score"] = (
        0.40 * row["precision"] + 0.25 * row["map50"] + 0.25 * row["recall"] + 0.10 * row["map50_95"]
    )
    row["balanced_router_score"] = (
        0.30 * row["precision"] + 0.30 * row["recall"] + 0.25 * row["map50"] + 0.15 * row["map50_95"]
    )
    row["recall_guard_precision_score"] = row["precision"] if row["recall"] >= base["recall"] - args.recall_tolerance else -1.0
    row["delta_precision"] = row["precision"] - base["precision"]
    row["delta_recall"] = row["recall"] - base["recall"]
    row["delta_map50"] = row["map50"] - base["map50"]
    row["delta_map50_95"] = row["map50_95"] - base["map50_95"]
    row["passes_recall_guard"] = row["recall"] >= base["recall"] - args.recall_tolerance
    row["hard_recall_drop"] = row["recall"] < base["recall"] - args.hard_recall_drop


def fmt(v: float) -> str:
    return f"{v:.4f}"


def markdown_table(rows: list[dict[str, float]], columns: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(label for label, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, int):
                cells.append(str(value))
            elif isinstance(value, float):
                cells.append(fmt(value))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    base = parse_base_log(Path(args.base_log))
    epochs = load_epochs(Path(args.test_results))
    for row in epochs:
        add_scores(row, base, args)

    base_row = {
        "epoch": "base",
        **base,
        "precision_first_score": 0.40 * base["precision"] + 0.25 * base["map50"] + 0.25 * base["recall"] + 0.10 * base["map50_95"],
        "balanced_router_score": 0.30 * base["precision"] + 0.30 * base["recall"] + 0.25 * base["map50"] + 0.15 * base["map50_95"],
        "recall_guard_precision_score": base["precision"],
        "delta_precision": 0.0,
        "delta_recall": 0.0,
        "delta_map50": 0.0,
        "delta_map50_95": 0.0,
        "passes_recall_guard": True,
        "hard_recall_drop": False,
        "weights": "base",
    }

    for row in epochs:
        row["weights"] = str(Path(args.run_dir) / "weights" / f"epoch{row['epoch']}.pt")

    eligible = [r for r in epochs if r["passes_recall_guard"]]
    best_precision_first = max(epochs, key=lambda r: r["precision_first_score"])
    best_balanced = max(epochs, key=lambda r: r["balanced_router_score"])
    best_recall_guard = max(eligible, key=lambda r: r["precision"]) if eligible else None
    best_recall = max(epochs, key=lambda r: r["recall"])
    best_map50 = max(epochs, key=lambda r: r["map50"])

    recommendation = "keep_base"
    recommended = base_row
    reason = "No epoch beats the base under the recall guard."
    if best_recall_guard and best_recall_guard["precision"] > base["precision"] and best_recall_guard["map50"] >= base["map50"] - 0.01:
        recommendation = "candidate_epoch"
        recommended = best_recall_guard
        reason = "This epoch keeps recall within tolerance and improves precision."

    summary_rows = [
        {"name": "base", **base_row},
        {"name": "precision_first", **best_precision_first},
        {"name": "balanced", **best_balanced},
        {"name": "recall_guard_precision", **best_recall_guard} if best_recall_guard else None,
        {"name": "best_recall", **best_recall},
        {"name": "best_map50", **best_map50},
    ]
    summary_rows = [r for r in summary_rows if r is not None]

    columns = [
        ("name", "name"),
        ("epoch", "epoch"),
        ("P", "precision"),
        ("R", "recall"),
        ("mAP50", "map50"),
        ("mAP50-95", "map50_95"),
        ("dP", "delta_precision"),
        ("dR", "delta_recall"),
        ("guard", "passes_recall_guard"),
    ]

    top_balanced = [{"name": "balanced", **r} for r in sorted(epochs, key=lambda r: r["balanced_router_score"], reverse=True)[:10]]
    top_precision_first = [
        {"name": "precision_first", **r}
        for r in sorted(epochs, key=lambda r: r["precision_first_score"], reverse=True)[:10]
    ]
    top_guard = [
        {"name": "recall_guard", **r}
        for r in sorted(eligible, key=lambda r: (r["precision"], r["map50"]), reverse=True)[:10]
    ]

    md = [
        "# Router Gemini Mix Tradeoff Summary",
        "",
        f"Base: `{args.base_name}` from `{args.base_log}`",
        f"Run: `{args.run_dir}`",
        "",
        "## Recommendation",
        "",
        f"- Decision: `{recommendation}`",
        f"- Recommended epoch: `{recommended['epoch']}`",
        f"- Reason: {reason}",
        f"- Recall guard: epoch recall must be >= base recall - {args.recall_tolerance:.3f}",
        "",
        "## Key Candidates",
        "",
        markdown_table(summary_rows, columns),
        "",
        "## Top Balanced Router Score",
        "",
        markdown_table(top_balanced, columns),
        "",
        "## Top Precision-First Score",
        "",
        markdown_table(top_precision_first, columns),
        "",
        "## Top Recall-Guard Precision",
        "",
        markdown_table(top_guard, columns) if top_guard else "No epochs passed the recall guard.",
        "",
    ]

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    all_rows = [base_row] + epochs
    fieldnames = [
        "epoch",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "precision_first_score",
        "balanced_router_score",
        "recall_guard_precision_score",
        "delta_precision",
        "delta_recall",
        "delta_map50",
        "delta_map50_95",
        "passes_recall_guard",
        "hard_recall_drop",
        "weights",
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n".join(md[:28]))
    print(f"\nWrote: {out_md}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
