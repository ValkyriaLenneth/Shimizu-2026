#!/usr/bin/env python3
"""Build a 5-class RF-DETR router dataset with brace and column-base classes."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
OLD_NAMES = {0: "天井", 1: "壁类", 2: "RC柱"}
NEW_LABEL_TO_CLASS = {
    "天井": 0,
    "内壁": 1,
    "RC壁": 1,
    "壁类": 1,
    "RC柱": 2,
    "ブレース": 3,
    "柱脚": 4,
}
NAMES = ["天井", "壁类", "RC柱", "ブレース", "柱脚"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-yolo-dir",
        default=(
            "final_release_20260615/data/final_download_20260526/handoff_20260519/"
            "shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/"
            "coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2"
        ),
        help="Existing 3-class YOLO router dataset with images/train layout.",
    )
    parser.add_argument(
        "--gemini-results",
        default="outputs/gemini_new_router_classes_20260630/results.jsonl",
    )
    parser.add_argument(
        "--review-items",
        help="Deduplicated review queue JSON. When set, new rows are built from reviewed items instead of Gemini JSONL.",
    )
    parser.add_argument(
        "--review-annotations",
        help="Manual review annotation JSON keyed by review item id.",
    )
    parser.add_argument("--output-dir", default="data/rfdetr_router_5class_brace_columnbase_20260630")
    parser.add_argument("--new-label-policy", choices=["expected-only", "all-router-labels"], default="expected-only")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--valid-source",
        choices=["valid", "test"],
        default="valid",
        help="Use the normal validation split or mirror test into valid for RF-DETR's required val loader.",
    )
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def normalize_old_layout(base_yolo_dir: Path, valid_source: str) -> dict[str, tuple[Path, Path]]:
    valid_name = "test" if valid_source == "test" else "val"
    return {
        "train": (base_yolo_dir / "images" / "train", base_yolo_dir / "labels" / "train"),
        "valid": (base_yolo_dir / "images" / valid_name, base_yolo_dir / "labels" / valid_name),
        "test": (base_yolo_dir / "images" / "test", base_yolo_dir / "labels" / "test"),
    }


def copy_old_dataset(base_yolo_dir: Path, output_dir: Path, mode: str, valid_source: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, (image_dir, label_dir) in normalize_old_layout(base_yolo_dir, valid_source).items():
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"missing old router split: {image_dir} / {label_dir}")
        image_count = label_count = 0
        class_counts: Counter[int] = Counter()
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"missing label for {image_path}: {label_path}")
            dst_name = f"old3__{image_path.name}"
            link_file(image_path, output_dir / split / "images" / dst_name, mode)
            link_file(label_path, output_dir / split / "labels" / f"{Path(dst_name).stem}.txt", mode)
            image_count += 1
            label_count += 1
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    class_counts[int(parts[0])] += 1
        summary[split] = {
            "images": image_count,
            "labels": label_count,
            "boxes": {str(cls): class_counts.get(cls, 0) for cls in range(len(NAMES))},
        }
    return summary


def clamp_box_1000(box: list[Any]) -> tuple[float, float, float, float] | None:
    if len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    ymin = max(0.0, min(1000.0, ymin))
    xmin = max(0.0, min(1000.0, xmin))
    ymax = max(0.0, min(1000.0, ymax))
    xmax = max(0.0, min(1000.0, xmax))
    if ymax <= ymin or xmax <= xmin:
        return None
    x_center = ((xmin + xmax) / 2.0) / 1000.0
    y_center = ((ymin + ymax) / 2.0) / 1000.0
    width = (xmax - xmin) / 1000.0
    height = (ymax - ymin) / 1000.0
    return x_center, y_center, width, height


def row_to_yolo_lines(row: dict[str, Any], policy: str, min_confidence: float) -> list[str]:
    parsed = ((row.get("response") or {}).get("parsed") or {}) if row.get("ok") else {}
    elements = parsed.get("elements", []) or []
    expected = row.get("expected_label")
    lines: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        label = element.get("label")
        if label not in NEW_LABEL_TO_CLASS:
            continue
        if policy == "expected-only" and label != expected:
            continue
        try:
            confidence = float(element.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue
        yolo = clamp_box_1000(element.get("bbox_2d") or [])
        if yolo is None:
            continue
        cls = NEW_LABEL_TO_CLASS[str(label)]
        x_center, y_center, width, height = yolo
        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def load_new_rows(results_path: Path) -> list[dict[str, Any]]:
    latest_by_image: dict[str, dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("image_path"):
                    latest_by_image[str(row["image_path"])] = row
    return list(latest_by_image.values())


def load_reviewed_rows(items_path: Path, review_path: Path | None) -> list[dict[str, Any]]:
    items = json.loads(items_path.read_text(encoding="utf-8"))
    reviews = {}
    if review_path and review_path.exists():
        reviews = json.loads(review_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for item in items:
        review = reviews.get(str(item["id"]), {})
        if review.get("status") == "rejected":
            continue
        boxes = review.get("boxes") if "boxes" in review else item.get("boxes", [])
        elements = []
        for box in boxes or []:
            label = box.get("label")
            if label not in NEW_LABEL_TO_CLASS:
                continue
            bbox = box.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            elements.append(
                {
                    "label": label,
                    "bbox_2d": bbox,
                    "confidence": 1.0 if box.get("confidence") is None else box.get("confidence"),
                    "reason": box.get("reason", "manual_review"),
                }
            )
        rows.append(
            {
                "expected_label": item["expected_label"],
                "image_path": item["image_path"],
                "image_rel_path": item.get("image_rel_path", item["image_path"]),
                "ok": True,
                "source": "manual_review_dedup",
                "review_item_id": item["id"],
                "dedup": item.get("dedup", {}),
                "response": {
                    "parsed": {
                        "elements": elements,
                        "image_level_labels": sorted({element["label"] for element in elements}),
                        "notes": review.get("notes", item.get("notes", "")),
                    }
                },
            }
        )
    return rows


def split_new_rows(rows: list[dict[str, Any]], seed: int, train_ratio: float, valid_ratio: float) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("expected_label", "")), []).append(row)

    splits = {"train": [], "valid": [], "test": []}
    for label, label_rows in sorted(by_label.items()):
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = round(n * train_ratio)
        n_valid = round(n * valid_ratio)
        splits["train"].extend(label_rows[:n_train])
        splits["valid"].extend(label_rows[n_train : n_train + n_valid])
        splits["test"].extend(label_rows[n_train + n_valid :])
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def mirror_test_to_valid(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    mirrored = {split: list(rows) for split, rows in splits.items()}
    mirrored["valid"] = list(mirrored["test"])
    return mirrored


def add_new_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    policy: str,
    min_confidence: float,
    mode: str,
) -> dict[str, Any]:
    written = skipped = 0
    class_counts: Counter[int] = Counter()
    skipped_reasons: Counter[str] = Counter()
    for index, row in enumerate(rows):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            skipped += 1
            skipped_reasons["missing_image"] += 1
            continue
        lines = row_to_yolo_lines(row, policy, min_confidence)
        if not lines:
            skipped += 1
            skipped_reasons["no_valid_boxes"] += 1
            continue
        safe_stem = f"new20260630__{split}__{index:05d}__{image_path.stem}"
        dst_image = output_dir / split / "images" / f"{safe_stem}{image_path.suffix.lower()}"
        dst_label = output_dir / split / "labels" / f"{safe_stem}.txt"
        link_file(image_path, dst_image, mode)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
        for line in lines:
            class_counts[int(line.split()[0])] += 1
    return {
        "candidate_rows": len(rows),
        "written_images": written,
        "skipped_rows": skipped,
        "skipped_reasons": dict(skipped_reasons),
        "boxes": {str(cls): class_counts.get(cls, 0) for cls in range(len(NAMES))},
    }


def write_data_yaml(output_dir: Path) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NAMES),
        "names": {i: name for i, name in enumerate(NAMES)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_summary = copy_old_dataset(Path(args.base_yolo_dir), output_dir, args.link_mode, args.valid_source)
    if args.review_items:
        new_rows = load_reviewed_rows(
            Path(args.review_items),
            Path(args.review_annotations) if args.review_annotations else None,
        )
    else:
        new_rows = [row for row in load_new_rows(Path(args.gemini_results)) if row.get("ok")]
    split_rows = split_new_rows(new_rows, args.seed, args.train_ratio, args.valid_ratio)
    if args.valid_source == "test":
        split_rows = mirror_test_to_valid(split_rows)
    new_summary = {
        split: add_new_rows(
            rows=rows,
            output_dir=output_dir,
            split=split,
            policy=args.new_label_policy,
            min_confidence=args.min_confidence,
            mode=args.link_mode,
        )
        for split, rows in split_rows.items()
    }
    write_data_yaml(output_dir)

    summary = {
        "output_dir": str(output_dir),
        "base_yolo_dir": args.base_yolo_dir,
        "gemini_results": args.gemini_results,
        "review_items": args.review_items,
        "review_annotations": args.review_annotations,
        "names": {str(i): name for i, name in enumerate(NAMES)},
        "new_label_policy": args.new_label_policy,
        "min_confidence": args.min_confidence,
        "valid_source": args.valid_source,
        "old_dataset": old_summary,
        "new_dataset": new_summary,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
