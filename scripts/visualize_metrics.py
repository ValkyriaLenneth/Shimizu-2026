from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metrics import plot_confusion_matrix, plot_history, plot_precision_recall_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history = run_dir / "metrics" / "history.csv"
    if history.exists():
        plot_history(history, run_dir / "figures")

    for split_dir in [p for p in (run_dir / "metrics").glob("*") if p.is_dir()]:
        if (split_dir / "confusion_matrix.csv").exists():
            fig_dir = run_dir / "figures" / split_dir.name
            plot_confusion_matrix(split_dir, fig_dir)
            if (split_dir / "per_class_metrics.csv").exists():
                plot_precision_recall_f1(split_dir, fig_dir)

    reports = []
    for report in (run_dir / "metrics").glob("*/per_class_metrics.csv"):
        df = pd.read_csv(report)
        df.insert(0, "split", report.parent.name)
        reports.append(df)
    if reports:
        pd.concat(reports, ignore_index=True).to_csv(
            run_dir / "metrics" / "per_class_metrics_all_splits.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"Updated figures under {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
