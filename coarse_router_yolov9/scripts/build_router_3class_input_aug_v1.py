#!/usr/bin/env python3
"""Build an input-augmented 3-class router dataset.

The script keeps val/test unchanged and expands only the train split. Augmented
images are generated with bbox-aware transforms so YOLO labels stay aligned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
CLASS_NAMES = {0: "天井", 1: "壁类", 2: "RC柱"}
RC_COLUMN_CLASS = 2
WALL_CLASS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source YOLO dataset root.")
    parser.add_argument("--output", required=True, help="Output YOLO dataset root.")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--rc-aug-per-image", type=int, default=2)
    parser.add_argument("--wall-aug-ratio", type=float, default=0.35)
    parser.add_argument("--ceiling-aug-ratio", type=float, default=0.15)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--affine-degrees", type=float, default=4.0)
    parser.add_argument("--affine-scale", type=float, default=0.12)
    parser.add_argument("--affine-translate", type=float, default=0.06)
    parser.add_argument("--photometric-prob", type=float, default=0.75)
    parser.add_argument("--blur-noise-prob", type=float, default=0.35)
    parser.add_argument("--occlusion-prob", type=float, default=0.35)
    parser.add_argument("--jpeg-prob", type=float, default=0.25)
    parser.add_argument("--jpeg-roundtrip-min-quality", type=int, default=55)
    parser.add_argument("--jpeg-roundtrip-max-quality", type=int, default=85)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    src = Path(args.source).resolve()
    out = Path(args.output).resolve()
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output already exists; pass --overwrite to rebuild: {out}")

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(src),
        "output": str(out),
        "seed": args.seed,
        "rc_aug_per_image": args.rc_aug_per_image,
        "wall_aug_ratio": args.wall_aug_ratio,
        "ceiling_aug_ratio": args.ceiling_aug_ratio,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_summary = materialize_original_split(src, out, split, args.link_mode)
        if split == "train":
            augmented = augment_train_split(
                src,
                out,
                rng,
                rc_aug_per_image=args.rc_aug_per_image,
                wall_aug_ratio=args.wall_aug_ratio,
                ceiling_aug_ratio=args.ceiling_aug_ratio,
                jpeg_quality=args.jpeg_quality,
                aug_params={
                    "affine_degrees": args.affine_degrees,
                    "affine_scale": args.affine_scale,
                    "affine_translate": args.affine_translate,
                    "photometric_prob": args.photometric_prob,
                    "blur_noise_prob": args.blur_noise_prob,
                    "occlusion_prob": args.occlusion_prob,
                    "jpeg_prob": args.jpeg_prob,
                    "jpeg_roundtrip_min_quality": args.jpeg_roundtrip_min_quality,
                    "jpeg_roundtrip_max_quality": args.jpeg_roundtrip_max_quality,
                },
            )
            split_summary["augmented_files_added"] = augmented["files"]
            split_summary["augmented_class_counts"] = augmented["class_counts"]
            split_summary["augmentation_types"] = augmented["types"]
        summary["splits"][split] = split_summary

    write_data_yaml(out)
    summary["final_counts"] = summarize_dataset(out)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def materialize_original_split(src: Path, out: Path, split: str, link_mode: str) -> dict[str, object]:
    labels = sorted((src / "labels" / split).glob("*.txt"))
    class_counts: Counter[int] = Counter()
    for label in labels:
        image = find_image(src / "images" / split, label.stem)
        place_file(image, out / "images" / split / f"{label.stem}{image.suffix.lower()}", link_mode)
        place_file(label, out / "labels" / split / label.name, link_mode)
        class_counts.update(cls for cls, *_ in read_yolo_label(label))
    return {
        "original_label_files": len(labels),
        "original_image_files": len(labels),
        "original_class_counts": {str(k): v for k, v in sorted(class_counts.items())},
    }


def augment_train_split(
    src: Path,
    out: Path,
    rng: random.Random,
    rc_aug_per_image: int,
    wall_aug_ratio: float,
    ceiling_aug_ratio: float,
    jpeg_quality: int,
    aug_params: dict[str, float],
) -> dict[str, object]:
    labels = sorted((src / "labels" / "train").glob("*.txt"))
    selected: list[tuple[Path, int]] = []
    for label in labels:
        classes = {cls for cls, *_ in read_yolo_label(label)}
        if RC_COLUMN_CLASS in classes:
            selected.extend((label, idx) for idx in range(rc_aug_per_image))
        elif WALL_CLASS in classes and rng.random() < wall_aug_ratio:
            selected.append((label, 0))
        elif rng.random() < ceiling_aug_ratio:
            selected.append((label, 0))

    type_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    written = 0
    for label, aug_index in selected:
        image_path = find_image(src / "images" / "train", label.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable image: {image_path}")
        boxes = read_yolo_label(label)
        aug_image, aug_boxes, aug_types = apply_random_augmentation(image, boxes, rng, aug_params)
        if not aug_boxes:
            continue
        suffix = f"__augv1_{aug_index:02d}_{written:05d}"
        out_image = out / "images" / "train" / f"{label.stem}{suffix}.jpg"
        out_label = out / "labels" / "train" / f"{label.stem}{suffix}.txt"
        ok = cv2.imwrite(str(out_image), aug_image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError(f"failed to write image: {out_image}")
        write_yolo_label(out_label, aug_boxes)
        type_counts.update(aug_types)
        class_counts.update(cls for cls, *_ in aug_boxes)
        written += 1

    return {
        "files": written,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "types": dict(type_counts),
    }


def apply_random_augmentation(
    image: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    rng: random.Random,
    aug_params: dict[str, float],
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]], list[str]]:
    aug_types: list[str] = []
    out = image.copy()
    aug_boxes = boxes[:]

    out, aug_boxes = affine_transform(out, aug_boxes, rng, aug_params)
    aug_types.append("affine")

    if rng.random() < aug_params["photometric_prob"]:
        out = photometric_transform(out, rng)
        aug_types.append("photometric")
    if rng.random() < aug_params["blur_noise_prob"]:
        out = blur_or_noise(out, rng)
        aug_types.append("blur_or_noise")
    if rng.random() < aug_params["occlusion_prob"]:
        out, aug_boxes = random_occlusion(out, aug_boxes, rng)
        aug_types.append("occlusion")
    if rng.random() < aug_params["jpeg_prob"]:
        out = jpeg_roundtrip(
            out,
            rng,
            int(aug_params["jpeg_roundtrip_min_quality"]),
            int(aug_params["jpeg_roundtrip_max_quality"]),
        )
        aug_types.append("jpeg")

    return out, clip_boxes(aug_boxes), aug_types


def affine_transform(
    image: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    rng: random.Random,
    aug_params: dict[str, float],
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-aug_params["affine_degrees"], aug_params["affine_degrees"])
    scale_delta = aug_params["affine_scale"]
    scale = rng.uniform(1.0 - scale_delta, 1.0 + scale_delta)
    translate = aug_params["affine_translate"]
    tx = rng.uniform(-translate, translate) * width
    ty = rng.uniform(-translate, translate) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    transformed = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_boxes = [transform_box(box, matrix, width, height) for box in boxes]
    return transformed, [box for box in new_boxes if box is not None]


def transform_box(
    box: tuple[int, float, float, float, float],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, float, float, float, float] | None:
    cls, x, y, w, h = box
    x1 = (x - w / 2.0) * width
    y1 = (y - h / 2.0) * height
    x2 = (x + w / 2.0) * width
    y2 = (y + h / 2.0) * height
    corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=np.float32)
    transformed = corners @ matrix.T
    nx1, ny1 = transformed[:, 0].min(), transformed[:, 1].min()
    nx2, ny2 = transformed[:, 0].max(), transformed[:, 1].max()
    nx1, ny1 = max(0.0, float(nx1)), max(0.0, float(ny1))
    nx2, ny2 = min(float(width), float(nx2)), min(float(height), float(ny2))
    if nx2 - nx1 < 3 or ny2 - ny1 < 3:
        return None
    return (cls, (nx1 + nx2) / 2.0 / width, (ny1 + ny2) / 2.0 / height, (nx2 - nx1) / width, (ny2 - ny1) / height)


def photometric_transform(image: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-28, 28)
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.75, 1.20)
    hsv[:, :, 2] *= rng.uniform(0.85, 1.15)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def blur_or_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        return cv2.GaussianBlur(image, (k, k), 0)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(3, 10), image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_occlusion(
    image: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    out = image.copy()
    height, width = out.shape[:2]
    rect_w = int(width * rng.uniform(0.05, 0.18))
    rect_h = int(height * rng.uniform(0.04, 0.16))
    x1 = rng.randint(0, max(0, width - rect_w))
    y1 = rng.randint(0, max(0, height - rect_h))
    patch = out[max(0, y1 - 5) : min(height, y1 + rect_h + 5), max(0, x1 - 5) : min(width, x1 + rect_w + 5)]
    color = tuple(int(v) for v in (patch.mean(axis=(0, 1)) if patch.size else np.array([128, 128, 128])))
    cv2.rectangle(out, (x1, y1), (x1 + rect_w, y1 + rect_h), color, thickness=-1)
    return out, boxes


def jpeg_roundtrip(image: np.ndarray, rng: random.Random, min_quality: int, max_quality: int) -> np.ndarray:
    quality = int(rng.uniform(min_quality, max_quality))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def clip_boxes(boxes: list[tuple[int, float, float, float, float]]) -> list[tuple[int, float, float, float, float]]:
    clipped = []
    for cls, x, y, w, h in boxes:
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        w = min(1.0, max(1e-5, w))
        h = min(1.0, max(1e-5, h))
        if math.isfinite(x + y + w + h):
            clipped.append((cls, x, y, w, h))
    return clipped


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"bad label line in {path}: {line}")
        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def write_yolo_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_image(root: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        for candidate in [root / f"{stem}{suffix}", root / f"{stem}{suffix.upper()}"]:
            if candidate.exists():
                return candidate
    matches = [p for p in root.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES]
    if not matches:
        raise FileNotFoundError(f"missing image for {stem} under {root}")
    return matches[0]


def place_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_data_yaml(out: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    text = f"""path: {out.resolve()}
train: images/train
val: images/val
test: images/test
nc: 3
names:
{names}
"""
    (out / "data.yaml").write_text(text, encoding="utf-8")


def summarize_dataset(root: Path) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in ["train", "val", "test"]:
        image_count = len([p for p in (root / "images" / split).iterdir() if p.is_file() or p.is_symlink()])
        labels = sorted((root / "labels" / split).glob("*.txt"))
        counts: Counter[int] = Counter()
        for label in labels:
            counts.update(cls for cls, *_ in read_yolo_label(label))
        summary[split] = {
            "images": image_count,
            "labels": len(labels),
            "class_counts": {str(k): v for k, v in sorted(counts.items())},
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
