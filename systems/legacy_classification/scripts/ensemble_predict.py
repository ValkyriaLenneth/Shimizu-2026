from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import ensure_dir, load_json, load_yaml, timestamp
from metrics import compute_classification_outputs, plot_confusion_matrix, plot_precision_recall_f1


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: Path, processed_root: Path, transform):
        self.df = pd.read_csv(manifest_csv)
        self.processed_root = processed_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.processed_root / row["processed_rel_path"]).convert("RGB")
        return self.transform(image), int(row["class_idx"]), row["processed_rel_path"]


def load_member(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu")
    cfg = ckpt["config"]
    class_to_idx = ckpt["class_to_idx"]
    model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    transform = create_transform(**ckpt["data_config"], is_training=False)
    return model, transform, cfg, class_to_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ensemble.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = []
    for member_cfg in cfg["members"]:
        run_dir = Path(member_cfg["run_dir"])
        if "REPLACE_WITH" in str(run_dir):
            raise ValueError("Update configs/ensemble.yaml with real run_dir values before running ensemble.")
        model, transform, train_cfg, class_to_idx = load_member(run_dir, device)
        members.append({"model": model, "transform": transform, "cfg": train_cfg, "weight": float(member_cfg["weight"])})

    class_to_idx = load_json(cfg["class_to_idx"])
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    first_cfg = members[0]["cfg"]
    processed_root = Path(first_cfg["dataset"]["processed_root"])
    dataset_name = processed_root.name
    manifest_csv = Path(cfg["manifest_dir"]) / f"{dataset_name}_{args.split}.csv"

    run_dir = ensure_dir(Path(cfg["output_root"]) / f"{timestamp()}_{cfg['ensemble_name']}")
    metrics_dir = ensure_dir(run_dir / "metrics" / args.split)
    figures_dir = ensure_dir(run_dir / "figures" / args.split)
    pred_dir = ensure_dir(run_dir / "predictions")

    # Use each member's own preprocessing, then average probabilities.
    member_loaders = []
    for member in members:
        ds = ManifestDataset(manifest_csv, processed_root, member["transform"])
        member_loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True))

    y_true, y_pred, rows = [], [], []
    iterators = [iter(loader) for loader in member_loaders]
    total_batches = len(member_loaders[0])
    with torch.no_grad():
        for _ in tqdm(range(total_batches), desc=f"Ensemble {args.split}"):
            batch_probs = None
            targets = None
            rel_paths = None
            for member, iterator in zip(members, iterators):
                images, batch_targets, batch_paths = next(iterator)
                logits = member["model"](images.to(device, non_blocking=True))
                probs = logits.softmax(dim=1).cpu() * member["weight"]
                batch_probs = probs if batch_probs is None else batch_probs + probs
                targets = batch_targets
                rel_paths = batch_paths
            preds = batch_probs.argmax(dim=1)
            y_true.extend(targets.tolist())
            y_pred.extend(preds.tolist())
            for rel_path, target, pred, prob in zip(rel_paths, targets.tolist(), preds.tolist(), batch_probs.tolist()):
                rows.append(
                    {
                        "path": rel_path,
                        "true_idx": target,
                        "true_class": class_names[target],
                        "pred_idx": pred,
                        "pred_class": class_names[pred],
                        "confidence": max(prob),
                        **{f"prob_{name}": prob[i] for i, name in enumerate(class_names)},
                    }
                )

    pd.DataFrame(rows).to_csv(pred_dir / f"{args.split}_predictions.csv", index=False, encoding="utf-8-sig")
    summary = compute_classification_outputs(y_true, y_pred, class_names, metrics_dir)
    plot_confusion_matrix(metrics_dir, figures_dir)
    plot_precision_recall_f1(metrics_dir, figures_dir)
    print(summary)
    print(f"Ensemble run saved to {run_dir}")


if __name__ == "__main__":
    main()
