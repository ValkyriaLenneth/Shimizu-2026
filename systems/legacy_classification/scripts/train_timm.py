from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import ensure_dir, load_json, load_yaml, save_json, seed_everything, timestamp
from metrics import compute_classification_outputs, plot_confusion_matrix, plot_history, plot_precision_recall_f1


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, processed_root: str | Path, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.processed_root = Path(processed_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.processed_root / row["processed_rel_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["class_idx"])


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    kwargs = {
        "pretrained": bool(model_cfg.get("pretrained", True)),
        "num_classes": int(model_cfg["num_classes"]),
    }
    if "drop_rate" in model_cfg:
        kwargs["drop_rate"] = float(model_cfg["drop_rate"])
    if "drop_path_rate" in model_cfg:
        kwargs["drop_path_rate"] = float(model_cfg["drop_path_rate"])
    return timm.create_model(model_cfg["name"], **kwargs)


def make_transforms(model: nn.Module, cfg: dict):
    data_cfg = resolve_data_config({"input_size": (3, cfg["model"]["img_size"], cfg["model"]["img_size"])}, model=model)
    train_tf = create_transform(
        **data_cfg,
        is_training=True,
        auto_augment=cfg.get("augment", {}).get("auto_augment"),
        color_jitter=cfg.get("augment", {}).get("color_jitter", 0.0),
        re_prob=cfg.get("augment", {}).get("random_erasing", 0.0),
    )
    eval_tf = create_transform(**data_cfg, is_training=False)
    return train_tf, eval_tf, data_cfg


def make_optimizer(model: nn.Module, cfg: dict):
    train_cfg = cfg["train"]
    if train_cfg.get("optimizer", "adamw").lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    if train_cfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            momentum=0.9,
            weight_decay=float(train_cfg["weight_decay"]),
        )
    raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")


def make_scheduler(optimizer, cfg: dict):
    train_cfg = cfg["train"]
    if train_cfg.get("scheduler", "cosine").lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    return None


def class_weights(manifest_csv: str | Path, num_classes: int) -> torch.Tensor:
    df = pd.read_csv(manifest_csv)
    counts = df["class_idx"].value_counts().sort_index()
    weights = [len(df) / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train(train)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, targets)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(loader.dataset), y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_to_idx = load_json(cfg["dataset"]["class_to_idx"])
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    run_dir = Path(cfg["output_root"]) / f"{timestamp()}_{cfg['run_name']}"
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_dir = ensure_dir(run_dir / "metrics")
    figures_dir = ensure_dir(run_dir / "figures")
    logs_dir = ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "predictions")
    shutil.copy2(args.config, run_dir / "config.yaml")
    save_json({"class_to_idx": class_to_idx, "class_names": class_names}, run_dir / "class_mapping.json")

    model = build_model(cfg).to(device)
    train_tf, eval_tf, data_cfg = make_transforms(model, cfg)
    save_json(data_cfg, run_dir / "data_config.json")

    processed_root = cfg["dataset"]["processed_root"]
    manifest_dir = Path(cfg["dataset"]["manifest_dir"])
    dataset_name = Path(processed_root).name
    train_csv = manifest_dir / f"{dataset_name}_train.csv"
    val_csv = manifest_dir / f"{dataset_name}_val.csv"

    train_ds = ManifestDataset(train_csv, processed_root, train_tf)
    val_ds = ManifestDataset(val_csv, processed_root, eval_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    weight = None
    if cfg["train"].get("class_weighted_loss", False):
        weight = class_weights(train_csv, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(logs_dir / "tensorboard"))

    best_metric_name = cfg["train"].get("best_metric", "macro_f1")
    best_metric = -1.0
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    bad_epochs = 0
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, train=True)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, optimizer, device, scaler=None, train=False)
        if scheduler is not None:
            scheduler.step()

        summary = compute_classification_outputs(y_true, y_pred, class_names, metrics_dir / "val_latest")
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], **summary}
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_dir / "history.csv", index=False)

        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        metric = float(summary[best_metric_name])
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} {best_metric_name}={metric:.4f}")
        torch.save({"model": model.state_dict(), "config": cfg, "class_to_idx": class_to_idx, "data_config": data_cfg}, checkpoints_dir / "last.pth")
        if metric > best_metric:
            best_metric = metric
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg, "class_to_idx": class_to_idx, "data_config": data_cfg}, checkpoints_dir / "best.pth")
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    plot_history(metrics_dir / "history.csv", figures_dir)
    plot_confusion_matrix(metrics_dir / "val_latest", figures_dir)
    plot_precision_recall_f1(metrics_dir / "val_latest", figures_dir)
    save_json({"best_metric": best_metric, "best_metric_name": best_metric_name}, metrics_dir / "best_summary.json")
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
