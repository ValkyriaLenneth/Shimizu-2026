#!/usr/bin/env python3
"""Build the ブレース / 柱脚 downstream B/C/D datasets for RF-DETR.

Input is the reviewed 1:1 manifest produced by
``match_new_class_annotations.py``. One dataset is built per element category,
matching the existing architecture of one recognition model per category.

Data policy agreed with the client on 2026-07-25:

* keep only images carrying at least one B/C/D box - every training image is
  assumed to contain damage, so empty labels are dropped rather than used as
  background negatives
* inside a duplicate cluster keep the representative with the most boxes
  (already elected by the matching step); grade contradictions are documented in
  ``docs/development_records/2026-07-25-new-classes-annotation-match.md``

Split policy:

* ``train`` / ``test`` only, split on ``scene_group_id`` so near-identical views
  of one scene cannot straddle the two splits
* stratified by the rarest grade present in each scene group, which is what
  keeps the scarce D grade represented in ``test``
* ``valid`` is a mirror of ``test``, because RF-DETR always requires a
  validation loader; this follows the existing ``test_as_valid`` convention
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

GRADE_NAMES = {0: "B", 1: "C", 2: "D"}
# Rarest grade first: a scene group is stratified by the scarcest grade it holds.
RARITY_ORDER = [2, 1, 0]

CATEGORY_LABELS = {"brace": "ブレース", "column_base": "柱脚"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/new_class_annotation_match_20260724/manifest.json")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--dataset-suffix", default="bcd_20260725_test_as_valid")
    parser.add_argument("--categories", nargs="*", default=sorted(CATEGORY_LABELS))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    else:
        shutil.copy2(source, target)


def select_records(manifest: dict[str, Any], category: str) -> list[dict[str, Any]]:
    """Keep annotated, deduplicated, non-empty images for one category."""
    return [
        record
        for record in manifest["records"]
        if record["category"] == category
        and record["source"] == "annotated"
        and record["is_representative"]
        and record["box_count"] > 0
    ]


def split_scene_groups(
    records: list[dict[str, Any]], test_ratio: float, seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign each scene group to train or test, stratified by rarest grade."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["scene_group_id"]].append(record)

    tiers: dict[int, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        present = {box[0] for member in members for box in member["boxes"]}
        rarest = next(grade for grade in RARITY_ORDER if grade in present)
        tiers[rarest].append(group_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    tier_report: dict[str, Any] = {}
    for grade in RARITY_ORDER:
        tier_groups = sorted(tiers.get(grade, []))
        rng.shuffle(tier_groups)
        # round() keeps the smallest tier from silently losing its test share.
        n_test = round(len(tier_groups) * test_ratio)
        if tier_groups and n_test == 0:
            n_test = 1
        for index, group_id in enumerate(tier_groups):
            assignment[group_id] = "test" if index < n_test else "train"
        tier_report[GRADE_NAMES[grade]] = {
            "scene_groups": len(tier_groups),
            "to_test": n_test,
            "to_train": len(tier_groups) - n_test,
        }
    return assignment, tier_report


def write_split(
    records: list[dict[str, Any]], dataset_dir: Path, split: str, link_mode: str
) -> dict[str, Any]:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    grade_counts: Counter[str] = Counter()
    images_with_grade: Counter[str] = Counter()
    for record in records:
        image_source = Path(record["image_path"])
        link_file(image_source, images_dir / f"{record['stem']}{image_source.suffix}", link_mode)
        lines = [
            f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
            for class_id, cx, cy, width, height in record["boxes"]
        ]
        (labels_dir / f"{record['stem']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        present = set()
        for box in record["boxes"]:
            grade_counts[GRADE_NAMES[box[0]]] += 1
            present.add(GRADE_NAMES[box[0]])
        for grade in present:
            images_with_grade[grade] += 1

    return {
        "images": len(records),
        "boxes": sum(record["box_count"] for record in records),
        "boxes_by_grade": dict(sorted(grade_counts.items())),
        "images_containing_grade": dict(sorted(images_with_grade.items())),
        "scene_groups": len({record["scene_group_id"] for record in records}),
        "stems": sorted(record["stem"] for record in records),
    }


def build_category(
    manifest: dict[str, Any], category: str, args: argparse.Namespace
) -> dict[str, Any]:
    label = CATEGORY_LABELS[category]
    dataset_name = f"rfdetr_{category}_{args.dataset_suffix}"
    dataset_dir = Path(args.output_root) / dataset_name
    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_dir} exists; pass --overwrite")
        shutil.rmtree(dataset_dir)

    selected = select_records(manifest, category)
    dropped_empty = [
        record
        for record in manifest["records"]
        if record["category"] == category
        and record["source"] == "annotated"
        and record["is_representative"]
        and record["box_count"] == 0
    ]
    assignment, tier_report = split_scene_groups(selected, args.test_ratio, args.seed)

    train_records = [record for record in selected if assignment[record["scene_group_id"]] == "train"]
    test_records = [record for record in selected if assignment[record["scene_group_id"]] == "test"]

    stats = {
        "train": write_split(train_records, dataset_dir, "train", args.link_mode),
        "test": write_split(test_records, dataset_dir, "test", args.link_mode),
    }
    # valid mirrors test: RF-DETR always needs a validation loader.
    stats["valid"] = write_split(test_records, dataset_dir, "valid", args.link_mode)

    names = {index: f"{label}の損傷程度{grade}" for index, grade in GRADE_NAMES.items()}
    (dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(names),
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    leaked = {record["scene_group_id"] for record in train_records} & {
        record["scene_group_id"] for record in test_records
    }
    summary = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "category": category,
        "category_label": label,
        "names": names,
        "source_manifest": args.manifest,
        "policy": {
            "empty_labels": "dropped (images are assumed to contain damage)",
            "duplicate_policy": "keep representative with most boxes",
            "split_unit": "scene_group_id",
            "stratified_by": "rarest grade present in the scene group",
            "valid_source": "test",
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "dropped_empty_label_images": len(dropped_empty),
        "stratification": tier_report,
        "splits": {
            split: {key: value for key, value in info.items() if key != "stems"}
            for split, info in stats.items()
        },
        "scene_group_leakage": sorted(leaked),
    }
    (dataset_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "note": "valid mirrors test; stems are filename stems without extension",
                "splits": {split: info["stems"] for split, info in stats.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    summaries = {}
    for category in args.categories:
        if category not in CATEGORY_LABELS:
            raise ValueError(f"unknown category {category}")
        summaries[category] = build_category(manifest, category, args)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    for category, summary in summaries.items():
        if summary["scene_group_leakage"]:
            print(f"[ERROR] {category}: scene group leaked across splits")
        else:
            print(f"[{category}] no scene-group leakage; dataset at {summary['dataset_dir']}")


if __name__ == "__main__":
    main()
