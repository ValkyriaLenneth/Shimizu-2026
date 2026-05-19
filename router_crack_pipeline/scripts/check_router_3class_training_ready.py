#!/usr/bin/env python3
"""Preflight checks for 3-class router YOLO training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

DATASETS = {
    "full": Path("coarse_router_yolov9/datasets/coarse_router_3class_full"),
    "cleaned": Path("coarse_router_yolov9/datasets/coarse_router_3class_cleaned"),
}
REQUIRED_CLASSES = {"0", "1", "2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-torch", action="store_true", help="do not fail when torch is missing")
    return parser.parse_args()


def check_torch(skip: bool) -> bool:
    try:
        import torch  # type: ignore
    except Exception as exc:
        print(f"[FAIL] torch import failed: {exc}")
        return bool(skip)
    print(f"[OK] torch {torch.__version__}")
    print(f"[OK] cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("[FAIL] expected at least 2 CUDA devices for parallel training")
        return False
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[OK] gpu{i}: {props.name}, memory={props.total_memory / 1024**3:.1f} GiB")
    return True


def check_nvidia_smi() -> None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"], text=True)
    except Exception as exc:
        print(f"[WARN] nvidia-smi unavailable: {exc}")
        return
    print("[INFO] nvidia-smi:")
    for line in out.strip().splitlines():
        print(f"  {line}")


def check_dataset(name: str, root: Path) -> bool:
    ok = True
    print(f"\n[DATASET] {name}: {root}")
    for rel in ["data.yaml", "summary.json", "manifest.csv"]:
        path = root / rel
        if path.exists():
            print(f"[OK] {rel}")
        else:
            print(f"[FAIL] missing {rel}")
            ok = False
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        print(f"[INFO] images={summary.get('images')} boxes={summary.get('boxes')} class_counts={summary.get('class_counts')}")
    total_images = 0
    total_labels = 0
    class_counts: Counter[str] = Counter()
    bad_labels = []
    bad_images = []
    for split in ["train", "val", "test"]:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted([p for p in image_dir.glob("*") if p.is_file()]) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        print(f"[INFO] {split}: images={len(images)} labels={len(labels)}")
        if len(images) != len(labels):
            print(f"[FAIL] {split} image/label count mismatch")
            ok = False
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)[:5]
        missing_images = sorted(label_stems - image_stems)[:5]
        if missing_labels or missing_images:
            print(f"[FAIL] {split} missing_labels={missing_labels} missing_images={missing_images}")
            ok = False
        for label_path in labels:
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or parts[0] not in REQUIRED_CLASSES:
                    bad_labels.append(f"{label_path}:{line_no}:{line}")
                    continue
                class_counts[parts[0]] += 1
                try:
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    bad_labels.append(f"{label_path}:{line_no}:{line}")
                    continue
                if any(x < 0 or x > 1 for x in coords) or coords[2] <= 0 or coords[3] <= 0:
                    bad_labels.append(f"{label_path}:{line_no}:{line}")
        for image_path in images:
            try:
                with Image.open(image_path) as im:
                    im.load()
                if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                    with image_path.open("rb") as f:
                        f.seek(-2, 2)
                        if f.read() != b"\xff\xd9":
                            bad_images.append(f"{image_path}: missing JPEG EOI marker")
            except Exception as exc:
                bad_images.append(f"{image_path}: {exc}")
        total_images += len(images)
        total_labels += len(labels)
    if bad_labels:
        print(f"[FAIL] bad label lines={len(bad_labels)} examples={bad_labels[:5]}")
        ok = False
    else:
        print(f"[OK] label format class_counts={dict(class_counts)}")
    if bad_images:
        print(f"[FAIL] unreadable/corrupt images={len(bad_images)} examples={bad_images[:5]}")
        ok = False
    else:
        print("[OK] image decode and JPEG EOI checks")
    print(f"[INFO] total_images={total_images} total_label_files={total_labels}")
    return ok


def main() -> int:
    args = parse_args()
    ok = True
    check_nvidia_smi()
    ok = check_torch(args.skip_torch) and ok
    for name, root in DATASETS.items():
        ok = check_dataset(name, root) and ok
    for path in [
        Path("coarse_router_yolov9/yolov9/train.py"),
        Path("coarse_router_yolov9/yolov9/models/detect/gelan-c.yaml"),
        Path("coarse_router_yolov9/yolov9/data/hyps/hyp.scratch-high.yaml"),
    ]:
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[FAIL] missing {path}")
            ok = False
    print("\nREADY" if ok else "\nNOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
