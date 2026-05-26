#!/usr/bin/env python3
"""Build a router YOLO dataset with real splits plus synthetic train images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


MERGED_CLASSES = ["天井", "壁类", "RC柱"]
MERGE_LABEL = {"天井": "天井", "内壁": "壁类", "RC壁": "壁类", "RC柱": "RC柱"}
MERGED_TO_ID = {name: i for i, name in enumerate(MERGED_CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset",
        default="handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2",
    )
    parser.add_argument(
        "--synthetic-results",
        default="outputs/synthetic_router_pipeline_nb2_promptgen_300x4_c10/annotation_results.jsonl",
    )
    parser.add_argument(
        "--out-dataset",
        default="handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2_gemini_nb2_mix",
    )
    parser.add_argument("--seed", type=int, default=2026052604)
    parser.add_argument("--synthetic-max-per-class", type=int, default=0, help="0 means use all available synthetic samples")
    parser.add_argument("--synthetic-fraction", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.995)
    parser.add_argument("--expected-only", action="store_true", help="keep only boxes matching the generated sample class")
    parser.add_argument("--synthetic-only-train", action="store_true", help="use synthetic images only for train; keep real val/test")
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def normalize_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in raw]
    except Exception:
        return None
    if any(math.isnan(v) or math.isinf(v) for v in [ymin, xmin, ymax, xmax]):
        return None
    ymin = max(0.0, min(1000.0, ymin)) / 1000.0
    xmin = max(0.0, min(1000.0, xmin)) / 1000.0
    ymax = max(0.0, min(1000.0, ymax)) / 1000.0
    xmax = max(0.0, min(1000.0, xmax)) / 1000.0
    if ymax <= ymin or xmax <= xmin:
        return None
    return ymin, xmin, ymax, xmax


def yolo_line(merged_label: str, bbox: tuple[float, float, float, float]) -> str:
    ymin, xmin, ymax, xmax = bbox
    bw = xmax - xmin
    bh = ymax - ymin
    return f"{MERGED_TO_ID[merged_label]} {xmin + bw / 2:.6f} {ymin + bh / 2:.6f} {bw:.6f} {bh:.6f}"


def parsed_payload(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("response") or {}).get("parsed") or {}


def synthetic_label_lines(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    expected = row.get("expected_label") or row.get("primary_class") or ""
    expected_merged = MERGE_LABEL.get(expected)
    lines = []
    for element in parsed_payload(row).get("elements") or []:
        label = str(element.get("label") or "")
        merged = MERGE_LABEL.get(label)
        if merged is None:
            continue
        if args.expected_only and expected_merged and merged != expected_merged:
            continue
        conf = element.get("confidence")
        if isinstance(conf, (int, float)) and float(conf) < args.min_confidence:
            continue
        bbox = normalize_bbox(element.get("bbox_2d"))
        if bbox is None:
            continue
        ymin, xmin, ymax, xmax = bbox
        area = (xmax - xmin) * (ymax - ymin)
        if area < args.min_area_ratio or area > args.max_area_ratio:
            continue
        lines.append(yolo_line(merged, bbox))
    return lines


def copy_base_split(base: Path, out: Path, split: str, mode: str, manifest_rows: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    image_dir = base / "images" / split
    for src_img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        src_lbl = base / "labels" / split / f"{src_img.stem}.txt"
        if not src_lbl.exists():
            continue
        dst_img = out / "images" / split / src_img.name
        dst_lbl = out / "labels" / split / src_lbl.name
        link_or_copy(src_img, dst_img, mode)
        link_or_copy(src_lbl, dst_lbl, mode)
        labels = [line.split()[0] for line in src_lbl.read_text(encoding="utf-8").splitlines() if line.strip()]
        for cls_id in labels:
            counts[MERGED_CLASSES[int(cls_id)]] += 1
        manifest_rows.append({
            "split": split,
            "source": "real",
            "dataset_image": str(dst_img.relative_to(out)),
            "dataset_label": str(dst_lbl.relative_to(out)),
            "source_image": str(src_img),
            "expected_label": "",
            "boxes": len(labels),
        })
    return counts


def load_synthetic_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("ok"):
            continue
        image_path = Path(row.get("image_path") or "")
        if not image_path.exists():
            continue
        lines = synthetic_label_lines(row, args)
        if not lines:
            continue
        row["_yolo_lines"] = lines
        rows.append(row)

    rng = random.Random(args.seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(row.get("expected_label") or row.get("primary_class") or "unknown", []).append(row)
    selected = []
    for cls, cls_rows in sorted(by_class.items()):
        rng.shuffle(cls_rows)
        n = len(cls_rows)
        if args.synthetic_fraction < 1.0:
            n = max(1, int(round(n * args.synthetic_fraction)))
        if args.synthetic_max_per_class > 0:
            n = min(n, args.synthetic_max_per_class)
        selected.extend(cls_rows[:n])
    rng.shuffle(selected)
    return selected


def add_synthetic_train(out: Path, rows: list[dict[str, Any]], mode: str, manifest_rows: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for idx, row in enumerate(rows):
        src_img = Path(row["image_path"])
        expected = row.get("expected_label") or row.get("primary_class") or "unknown"
        stem = f"synthetic_{expected}_{idx:05d}_{src_img.stem}"
        dst_img = out / "images" / "train" / f"{stem}{src_img.suffix.lower()}"
        dst_lbl = out / "labels" / "train" / f"{stem}.txt"
        link_or_copy(src_img, dst_img, mode)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(row["_yolo_lines"]) + "\n", encoding="utf-8")
        for line in row["_yolo_lines"]:
            counts[MERGED_CLASSES[int(line.split()[0])]] += 1
        manifest_rows.append({
            "split": "train",
            "source": "synthetic",
            "dataset_image": str(dst_img.relative_to(out)),
            "dataset_label": str(dst_lbl.relative_to(out)),
            "source_image": str(src_img),
            "expected_label": expected,
            "boxes": len(row["_yolo_lines"]),
        })
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dataset)
    out = Path(args.out_dataset)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    split_counts = {}
    for split in ["train", "val", "test"]:
        if split == "train" and args.synthetic_only_train:
            split_counts[split] = Counter()
            continue
        split_counts[split] = copy_base_split(base, out, split, args.link_mode, manifest_rows)

    synthetic_rows = load_synthetic_rows(Path(args.synthetic_results), args)
    synthetic_counts = add_synthetic_train(out, synthetic_rows, args.link_mode, manifest_rows)
    split_counts["train_synthetic"] = synthetic_counts

    (out / "data.yaml").write_text(
        "path: " + str(out.resolve()) + "\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(MERGED_CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(MERGED_CLASSES)),
        encoding="utf-8",
    )
    write_csv(out / "manifest.csv", manifest_rows)

    summary = {
        "base_dataset": str(base),
        "synthetic_results": args.synthetic_results,
        "out_dataset": str(out),
        "seed": args.seed,
        "synthetic_images": len(synthetic_rows),
        "image_counts": {
            split: len(list((out / "images" / split).iterdir())) for split in ["train", "val", "test"]
        },
        "box_counts": {key: dict(value) for key, value in split_counts.items()},
        "options": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
