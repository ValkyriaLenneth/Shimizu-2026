from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import create_transform

from common import load_yaml


def load_single_run(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu")
    cfg = ckpt["config"]
    class_to_idx = ckpt["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    transform = create_transform(**ckpt["data_config"], is_training=False)
    return model, transform, class_names


def predict_image(image_path: Path, members: list[dict], device: torch.device, top_k: int) -> dict:
    probs_sum = None
    class_names = members[0]["class_names"]
    for member in members:
        image = Image.open(image_path).convert("RGB")
        tensor = member["transform"](image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = member["model"](tensor).softmax(dim=1).cpu().squeeze(0) * member["weight"]
        probs_sum = probs if probs_sum is None else probs_sum + probs

    values, indices = torch.topk(probs_sum, k=min(top_k, len(class_names)))
    result = {
        "image": str(image_path),
        "pred_class": class_names[int(indices[0])],
        "confidence": float(values[0]),
    }
    for rank, (idx, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        result[f"top{rank}_class"] = class_names[idx]
        result[f"top{rank}_prob"] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--image-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--ensemble-config")
    parser.add_argument("--output", default="outputs/inference_predictions.csv")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide --image or --image-dir.")
    if not args.run_dir and not args.ensemble_config:
        raise ValueError("Provide --run-dir or --ensemble-config.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = []
    if args.run_dir:
        model, transform, class_names = load_single_run(Path(args.run_dir), device)
        members.append({"model": model, "transform": transform, "class_names": class_names, "weight": 1.0})
    else:
        cfg = load_yaml(args.ensemble_config)
        total_weight = sum(float(m["weight"]) for m in cfg["members"])
        for member_cfg in cfg["members"]:
            model, transform, class_names = load_single_run(Path(member_cfg["run_dir"]), device)
            members.append(
                {
                    "model": model,
                    "transform": transform,
                    "class_names": class_names,
                    "weight": float(member_cfg["weight"]) / total_weight,
                }
            )

    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths.extend([p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in exts])

    rows = [predict_image(path, members, device, args.top_k) for path in image_paths]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
