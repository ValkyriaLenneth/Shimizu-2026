#!/usr/bin/env python3
"""Run RF-DETR router test evaluation over saved epoch checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default="data/rfdetr_router_base_aug_v2")
    parser.add_argument("--epochs", default="", help="comma-separated epoch list; default: all epoch_pth checkpoints")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_epoch(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def selected_checkpoints(run_dir: Path, epochs_arg: str) -> list[Path]:
    epoch_dir = run_dir / "epoch_pth"
    if epochs_arg.strip():
        epochs = {int(item) for item in epochs_arg.split(",") if item.strip()}
        return [epoch_dir / f"checkpoint_epoch_{epoch:03d}.pth" for epoch in sorted(epochs)]
    return sorted(epoch_dir.glob("checkpoint_epoch_*.pth"), key=parse_epoch)


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(float(row["epoch"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_test_objects(
    checkpoint_path: Path,
    dataset_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[Any, Any]:
    import rfdetr

    from checkpoint_resolution import from_checkpoint_matched
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    from rfdetr.datasets import build_dataset
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = dict(checkpoint["args"])
    args["dataset_dir"] = str(dataset_dir.resolve())
    args["dataset_file"] = "yolo"
    args["output_dir"] = str(checkpoint_path.parent.parent.resolve())
    args["run_test"] = False
    args["tensorboard"] = False
    args["wandb"] = False
    args["mlflow"] = False
    args["clearml"] = False
    args["augmentation_backend"] = "cpu"
    args["multi_scale"] = False
    args["expanded_scales"] = False
    if batch_size > 0:
        args["batch_size"] = batch_size
    if num_workers >= 0:
        args["num_workers"] = num_workers

    train_config = TrainConfig(**args)
    model = from_checkpoint_matched(checkpoint_path)
    module = RFDETRModelModule(model.model_config, train_config)
    datamodule = RFDETRDataModule(model.model_config, train_config)

    ns = _namespace_from_configs(model.model_config, train_config)
    datamodule._dataset_test = build_dataset("test", ns, model.model_config.resolution)
    return module, datamodule


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from pytorch_lightning import Trainer
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

    module, datamodule = build_test_objects(
        checkpoint_path,
        dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trainer = Trainer(
        accelerator="gpu" if args.device.startswith("cuda") else "cpu",
        devices=[int(args.device.split(":", 1)[1])] if args.device.startswith("cuda:") else 1,
        precision=args.precision,
        callbacks=[
            COCOEvalCallback(
                max_dets=module.train_config.eval_max_dets,
                segmentation=module.model_config.segmentation_head,
                eval_interval=1,
                log_per_class_metrics=True,
            )
        ],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    result = trainer.test(module, datamodule=datamodule, verbose=False)
    row: dict[str, Any] = dict(result[0]) if result else {}
    row["epoch"] = parse_epoch(checkpoint_path)
    row["checkpoint"] = str(checkpoint_path)
    return {key: to_plain(value) for key, value in row.items()}


def to_plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else run_dir / "test_results.csv"

    checkpoints = selected_checkpoints(run_dir, args.epochs)
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))

    existing = load_existing(output_csv)
    rows = [existing[epoch] for epoch in sorted(existing)]
    by_epoch = {int(float(row["epoch"])): row for row in rows}

    for checkpoint_path in checkpoints:
        epoch = parse_epoch(checkpoint_path)
        if args.skip_existing and epoch in by_epoch:
            print(f"[skip] epoch {epoch}")
            continue
        print(f"[eval] epoch {epoch}: {checkpoint_path}")
        row = evaluate_checkpoint(checkpoint_path, dataset_dir, args)
        by_epoch[epoch] = row
        write_rows(output_csv, [by_epoch[key] for key in sorted(by_epoch)])
        print(
            "[done] epoch {epoch} test/precision={precision} test/recall={recall} test/mAP_50={map50}".format(
                epoch=epoch,
                precision=row.get("test/precision"),
                recall=row.get("test/recall"),
                map50=row.get("test/mAP_50"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
