#!/usr/bin/env python3
"""Build RF-DETR YOLO views for single-component crack models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


COMPONENTS = {
    "ceiling": {
        "dataset": "tenjo",
        "output": "rfdetr_tenjo_all_non_legacy_test_v1",
        "names": ["天井の損傷程度B", "天井の損傷程度C", "天井の損傷程度D"],
    },
    "interior": {
        "dataset": "inner_wall",
        "output": "rfdetr_inner_wall_all_non_legacy_test_v1",
        "names": ["内壁の損傷程度B", "内壁の損傷程度C", "内壁の損傷程度D"],
    },
    "rc_wall": {
        "dataset": "rc_wall",
        "output": "rfdetr_rc_wall_all_non_legacy_test_v1",
        "names": ["耐震壁の損傷程度B", "耐震壁の損傷程度C", "耐震壁の損傷程度D"],
    },
    "rc_column": {
        "dataset": "rc_column",
        "output": "rfdetr_rc_column_all_non_legacy_test_v1",
        "names": ["RC柱の損傷程度B", "RC柱の損傷程度C", "RC柱の損傷程度D"],
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split", required=True, help="Current final_crack_yolo_20260519/split directory.")
    parser.add_argument("--data-split-json", default="data_split.json")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--components", default="ceiling,interior,rc_wall,rc_column")
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_stem(path_or_stem: str | Path) -> str:
    stem = Path(path_or_stem).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def read_split_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = data.get("parts")
    if not isinstance(parts, dict):
        raise ValueError(f"{path} does not contain a parts mapping")
    return parts


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


def collect_samples(dataset_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix in IMAGE_EXTS):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"missing label for {image_path}: {label_path}")
            samples.append(
                {
                    "source_split": split,
                    "image": image_path,
                    "label": label_path,
                    "stem": image_path.stem,
                    "canonical_stem": canonical_stem(image_path),
                }
            )
    return samples


def count_boxes(label_paths: list[Path]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for label_path in label_paths:
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def write_yaml(path: Path, names: list[str]) -> None:
    data = {
        "path": str(path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (path / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_component(
    component_key: str,
    source_root: Path,
    output_root: Path,
    official_test_stems: set[str],
    mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    info = COMPONENTS[component_key]
    dataset_dir = source_root / info["dataset"]
    output_dir = output_root / info["output"]
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)

    samples = collect_samples(dataset_dir)
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_canonical[sample["canonical_stem"]].append(sample)

    missing = sorted(stem for stem in official_test_stems if stem not in by_canonical)
    if missing:
        raise FileNotFoundError(f"{component_key} official test stems missing from source data: {missing[:20]}")

    train_samples = [sample for sample in samples if sample["canonical_stem"] not in official_test_stems]
    test_samples = []
    duplicate_test_stems: dict[str, list[str]] = {}
    for stem in sorted(official_test_stems):
        matches = by_canonical[stem]
        if len(matches) > 1:
            duplicate_test_stems[stem] = [m["stem"] for m in matches]
        test_samples.append(matches[0])

    split_samples = {
        "train": train_samples,
        "valid": test_samples,
        "test": test_samples,
    }

    for split, selected in split_samples.items():
        for sample in selected:
            link_file(sample["image"], output_dir / split / "images" / sample["image"].name, mode)
            link_file(sample["label"], output_dir / split / "labels" / sample["label"].name, mode)

    write_yaml(output_dir, info["names"])

    summary: dict[str, Any] = {
        "component_key": component_key,
        "source_dataset": str(dataset_dir),
        "output_dir": str(output_dir),
        "official_test_stems": len(official_test_stems),
        "duplicate_test_stems": duplicate_test_stems,
        "splits": {},
    }
    for split, selected in split_samples.items():
        label_paths = [sample["label"] for sample in selected]
        boxes = count_boxes(label_paths)
        summary["splits"][split] = {
            "images": len(selected),
            "boxes": {str(cls): boxes.get(cls, 0) for cls in range(3)},
            "source_split_counts": dict(Counter(sample["source_split"] for sample in selected)),
        }

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_split)
    output_root = Path(args.output_root)
    parts = read_split_json(Path(args.data_split_json))
    selected_components = [part.strip() for part in args.components.split(",") if part.strip()]

    all_summaries = {}
    for component_key in selected_components:
        if component_key not in COMPONENTS:
            raise ValueError(f"unknown component {component_key}; expected one of {sorted(COMPONENTS)}")
        if component_key not in parts:
            raise ValueError(f"{args.data_split_json} has no part {component_key}")
        official_test_stems = set(parts[component_key].get("test", []))
        summary = build_component(
            component_key=component_key,
            source_root=source_root,
            output_root=output_root,
            official_test_stems=official_test_stems,
            mode=args.link_mode,
            overwrite=args.overwrite,
        )
        all_summaries[component_key] = summary
        split_bits = ", ".join(
            f"{split}={data['images']} images boxes={data['boxes']}"
            for split, data in summary["splits"].items()
        )
        print(f"{component_key}: {summary['output_dir']} ({split_bits})")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rfdetr_single_crack_views_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
