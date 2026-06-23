from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from common import save_json


def compute_classification_outputs(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_dir: str | Path,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    save_json(report, output_dir / "classification_report.json")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    per_class_rows = []
    for name in class_names:
        values = report[name]
        per_class_rows.append(
            {
                "class": name,
                "precision": values["precision"],
                "recall_r1": values["recall"],
                "f1": values["f1-score"],
                "support": values["support"],
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall_r1": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall_r1": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def plot_confusion_matrix(metrics_dir: str | Path, figures_dir: str | Path) -> None:
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = pd.read_csv(metrics_dir / "confusion_matrix.csv", index_col=0)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def plot_history(history_csv: str | Path, figures_dir: str | Path) -> None:
    history = pd.read_csv(history_csv)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if {"train_loss", "val_loss"}.issubset(history.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(history["epoch"], history["train_loss"], label="train_loss")
        plt.plot(history["epoch"], history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "loss.png", dpi=160)
        plt.close()

    metric_cols = [
        c
        for c in ["accuracy", "macro_precision", "macro_recall_r1", "macro_f1"]
        if c in history.columns
    ]
    if metric_cols:
        plt.figure(figsize=(8, 5))
        for col in metric_cols:
            plt.plot(history["epoch"], history[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "metrics.png", dpi=160)
        plt.close()


def plot_precision_recall_f1(metrics_dir: str | Path, figures_dir: str | Path) -> None:
    df = pd.read_csv(Path(metrics_dir) / "per_class_metrics.csv")
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, df["precision"], width=width, label="precision")
    plt.bar(x, df["recall_r1"], width=width, label="recall_r1")
    plt.bar(x + width, df["f1"], width=width, label="f1")
    plt.xticks(x, df["class"])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / "precision_recall_f1.png", dpi=160)
    plt.close()
