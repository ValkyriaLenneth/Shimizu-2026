#!/usr/bin/env python3
"""Build k-fold cross-validation dataset views for the new categories.

Why this exists: the single 9:1 test split holds 39 boxes for ブレース and 38 for
柱脚. Six experiments, three checkpoints each, and 3375 threshold combinations per
checkpoint means the reported best is the maximum over roughly 60000 noisy
estimates on 39 boxes, with valid mirroring test. That number is optimistically
biased and cannot separate a real improvement from noise.

Cross-validation fixes the measurement rather than the model: every image serves
as test exactly once, so the pooled test set is the full 438 / 282 boxes instead of
39 / 38, and the estimate is out-of-fold rather than selected-on.

Folds are assigned over ``scene_group_id`` from the match manifest, so
near-identical views of one scene stay inside a single fold - the same leakage
guard the main split uses. Folds are stratified by the rarest grade present in a
scene group, which keeps the scarce D grade represented in every fold.

Each fold is emitted as a standard RF-DETR YOLO view with train/valid/test, valid
mirroring test, and optional train-only crop augmentation applied afterwards by
``build_rfdetr_crop_aug_view.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

GRADE_NAMES = {0: "B", 1: "C", 2: "D"}
RARITY_ORDER = [2, 1, 0]
CATEGORY_LABELS = {"brace": "ブレース", "column_base": "柱脚"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/new_class_annotation_match_20260724/manifest.json")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--categories", nargs="*", default=sorted(CATEGORY_LABELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tag", default="cv5_20260725")
    parser.add_argument("--seed", type=int, default=20260725)
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


def select_records(manifest: dict, category: str) -> list[dict]:
    """Same policy as the main dataset build: annotated, deduplicated, non-empty."""
    return [
        record
        for record in manifest["records"]
        if record["category"] == category
        and record["source"] == "annotated"
        and record["is_representative"]
        and record["box_count"] > 0
    ]


def assign_folds(records: list[dict], folds: int, seed: int) -> tuple[dict[str, int], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["scene_group_id"]].append(record)

    tiers: dict[int, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        present = {box[0] for member in members for box in member["boxes"]}
        rarest = next(grade for grade in RARITY_ORDER if grade in present)
        tiers[rarest].append(group_id)

    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    report: dict[str, dict] = {}
    for grade in RARITY_ORDER:
        tier_groups = sorted(tiers.get(grade, []))
        rng.shuffle(tier_groups)
        # Deal round-robin so each fold gets a near-equal share of this tier.
        for index, group_id in enumerate(tier_groups):
            assignment[group_id] = index % folds
        counts = Counter(assignment[g] for g in tier_groups)
        report[GRADE_NAMES[grade]] = {"scene_groups": len(tier_groups), "per_fold": dict(sorted(counts.items()))}
    return assignment, report


def write_split(records: list[dict], dataset_dir: Path, split: str, link_mode: str) -> dict:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    grades: Counter[str] = Counter()
    for record in records:
        source = Path(record["image_path"])
        link_file(source, images_dir / f"{record['stem']}{source.suffix}", link_mode)
        lines = [
            f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in record["boxes"]
        ]
        (labels_dir / f"{record['stem']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for box in record["boxes"]:
            grades[GRADE_NAMES[box[0]]] += 1
    return {
        "images": len(records),
        "boxes": sum(r["box_count"] for r in records),
        "boxes_by_grade": dict(sorted(grades.items())),
        "scene_groups": len({r["scene_group_id"] for r in records}),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary: dict[str, dict] = {}

    for category in args.categories:
        label = CATEGORY_LABELS[category]
        records = select_records(manifest, category)
        assignment, tier_report = assign_folds(records, args.folds, args.seed)

        per_fold = []
        for fold in range(args.folds):
            name = f"rfdetr_{category}_{args.tag}_fold{fold}_test_as_valid"
            dataset_dir = Path(args.output_root) / name
            if dataset_dir.exists():
                if not args.overwrite:
                    raise FileExistsError(f"{dataset_dir} exists; pass --overwrite")
                shutil.rmtree(dataset_dir)

            test_records = [r for r in records if assignment[r["scene_group_id"]] == fold]
            train_records = [r for r in records if assignment[r["scene_group_id"]] != fold]
            stats = {
                "train": write_split(train_records, dataset_dir, "train", args.link_mode),
                "test": write_split(test_records, dataset_dir, "test", args.link_mode),
            }
            stats["valid"] = write_split(test_records, dataset_dir, "valid", args.link_mode)

            leaked = {r["scene_group_id"] for r in train_records} & {
                r["scene_group_id"] for r in test_records
            }
            names = {i: f"{label}の損傷程度{g}" for i, g in GRADE_NAMES.items()}
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
            fold_summary = {
                "fold": fold,
                "dataset_dir": str(dataset_dir),
                "splits": stats,
                "scene_group_leakage": sorted(leaked),
            }
            (dataset_dir / "build_summary.json").write_text(
                json.dumps(fold_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            per_fold.append(fold_summary)

        pooled_test_boxes = sum(f["splits"]["test"]["boxes"] for f in per_fold)
        pooled_test_images = sum(f["splits"]["test"]["images"] for f in per_fold)
        summary[category] = {
            "label": label,
            "folds": args.folds,
            "usable_images": len(records),
            "pooled_test_images": pooled_test_images,
            "pooled_test_boxes": pooled_test_boxes,
            "stratification": tier_report,
            "per_fold": [
                {
                    "fold": f["fold"],
                    "train_images": f["splits"]["train"]["images"],
                    "test_images": f["splits"]["test"]["images"],
                    "test_boxes": f["splits"]["test"]["boxes"],
                    "test_by_grade": f["splits"]["test"]["boxes_by_grade"],
                    "leakage": f["scene_group_leakage"],
                }
                for f in per_fold
            ],
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for category, info in summary.items():
        bad = [f for f in info["per_fold"] if f["leakage"]]
        state = "LEAKAGE" if bad else "no leakage"
        print(
            f"[{category}] {info['folds']} folds, pooled test "
            f"{info['pooled_test_images']} images / {info['pooled_test_boxes']} boxes, {state}"
        )


if __name__ == "__main__":
    main()
