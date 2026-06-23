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

from common import load_json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu")
    cfg = ckpt["config"]
    class_to_idx = load_json(run_dir / "class_mapping.json")["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    transform = create_transform(**ckpt["data_config"], is_training=False)
    processed_root = Path(cfg["dataset"]["processed_root"])
    dataset_name = processed_root.name
    manifest_csv = Path(cfg["dataset"]["manifest_dir"]) / f"{dataset_name}_{args.split}.csv"
    ds = ManifestDataset(manifest_csv, processed_root, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    y_true, y_pred, rows = [], [], []
    with torch.no_grad():
        for images, targets, rel_paths in tqdm(loader, desc=f"Evaluating {args.split}"):
            logits = model(images.to(device, non_blocking=True))
            probs = logits.softmax(dim=1).cpu()
            preds = probs.argmax(dim=1)
            y_true.extend(targets.tolist())
            y_pred.extend(preds.tolist())
            for rel_path, target, pred, prob in zip(rel_paths, targets.tolist(), preds.tolist(), probs.tolist()):
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

    metrics_dir = run_dir / "metrics" / args.split
    figures_dir = run_dir / "figures" / args.split
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(pred_dir / f"{args.split}_predictions.csv", index=False, encoding="utf-8-sig")
    summary = compute_classification_outputs(y_true, y_pred, class_names, metrics_dir)
    plot_confusion_matrix(metrics_dir, figures_dir)
    plot_precision_recall_f1(metrics_dir, figures_dir)
    print(summary)


if __name__ == "__main__":
    main()
