#!/usr/bin/env python3
"""Freeze the one canonical ブレース / 柱脚 train+test split, and detect later drift.

Why this exists: 36 dataset directories accumulated for these two categories
across three generations of split policy (8:2, 9:1, 5-fold CV) and four train
views (base, crop2, crop3, dboost). Numbers from different generations were
compared against each other in the experiment log, and the builders take
``--overwrite``. Nothing on disk recorded which split a stored number belonged to,
so a rebuild with a different seed or a changed upstream manifest would silently
invalidate every result without changing a single count.

Decision of 2026-07-25: **one fixed train/test split per category**, no
cross-validation. The 8:2 build is the frozen one rather than the 9:1 build,
because the 9:1 test split holds 39 / 38 boxes - and only 4 grade-D boxes for
柱脚, which quantises D recall to steps of 0.25 and cannot support incremental
tuning. 8:2 gives 83 / 72 test boxes at the cost of about 11% of the training
images.

The fingerprint is over label *content* (stem plus the exact box lines at fixed
precision), so a rebuild that reshuffles the split, drops an image or rescales a
coordinate changes the digest. Image identity is a (stem, extension, byte size)
digest, which catches a re-export at another resolution without hashing pixels.

Derived train views (crop, oversampled, class-boosted) are checked rather than
frozen: the requirement on them is that their ``test`` split is byte-identical to
the frozen one, which is what makes a view comparable to the baseline at all.

Usage:

    freeze_new_class_datasets.py --write     # create the lockfile
    freeze_new_class_datasets.py --check     # verify nothing drifted (exit 1 if it did)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

GRADES = {0: "B", 1: "C", 2: "D"}
CATEGORIES = ["brace", "column_base"]

# The frozen split. Everything reported for these two categories is measured here.
CANONICAL_SUFFIX = "bcd_20260725_test_as_valid"
SPLITS = ("train", "test", "valid")
DEFAULT_LOCK = "data/frozen/new_classes_20260725.lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="create or replace the lockfile")
    mode.add_argument("--check", action="store_true", help="verify current state against the lockfile")
    parser.add_argument("--lockfile", default=DEFAULT_LOCK)
    parser.add_argument("--paired-root", default="data/new_classes_paired_20260724")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--check-views",
        nargs="*",
        default=[],
        help="derived train views to check for test-split identity, e.g. bcd_20260725_split91_crop2_test_as_valid",
    )
    return parser.parse_args()


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        boxes.append((int(parts[0]), *(float(v) for v in parts[1:5])))
    return boxes


def fingerprint_split(split_dir: Path) -> dict:
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"missing split: {split_dir}")

    label_hash = hashlib.sha256()
    image_hash = hashlib.sha256()
    grades: Counter[str] = Counter()
    box_total = 0
    stems = sorted(p.stem for p in labels_dir.glob("*.txt"))
    image_by_stem = {p.stem: p for p in images_dir.iterdir() if p.is_file()}

    for stem in stems:
        boxes = read_label(labels_dir / f"{stem}.txt")
        box_total += len(boxes)
        for cls, *_ in boxes:
            grades[GRADES[cls]] += 1
        # Re-emitted at fixed precision so a harmless reformat is not read as
        # drift while a real coordinate change is.
        canonical = ";".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes)
        label_hash.update(f"{stem}|{canonical}\n".encode("utf-8"))
        image = image_by_stem.get(stem)
        size = image.stat().st_size if image else -1
        image_hash.update(f"{stem}|{image.suffix if image else ''}|{size}\n".encode("utf-8"))

    return {
        "images": len(stems),
        "boxes": box_total,
        "boxes_by_grade": dict(sorted(grades.items())),
        "label_digest": label_hash.hexdigest(),
        "image_digest": image_hash.hexdigest(),
    }


def fingerprint_paired(paired_dir: Path) -> dict:
    labels_dir = paired_dir / "labels"
    images_dir = paired_dir / "images"
    digest = hashlib.sha256()
    grades: Counter[str] = Counter()
    annotated = 0
    empty = 0
    stems = sorted(p.stem for p in labels_dir.glob("*.txt"))
    image_by_stem = {p.stem: p for p in images_dir.iterdir() if p.is_file()}
    for stem in stems:
        boxes = read_label(labels_dir / f"{stem}.txt")
        if boxes:
            annotated += 1
        else:
            empty += 1
        for cls, *_ in boxes:
            grades[GRADES[cls]] += 1
        canonical = ";".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes)
        image = image_by_stem.get(stem)
        size = image.stat().st_size if image else -1
        digest.update(f"{stem}|{canonical}|{size}\n".encode("utf-8"))
    return {
        "images": len(stems),
        "images_with_boxes": annotated,
        "images_empty": empty,
        "boxes": sum(grades.values()),
        "boxes_by_grade": dict(sorted(grades.items())),
        "digest": digest.hexdigest(),
    }


def stems_of(split_dir: Path) -> set[str]:
    return {p.stem for p in (split_dir / "labels").glob("*.txt")}


def collect(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    paired_root = Path(args.paired_root)
    state: dict = {
        "policy": "single fixed train/test split per category, no cross-validation",
        "canonical_suffix": CANONICAL_SUFFIX,
        "categories": {},
    }

    for category in CATEGORIES:
        dataset_dir = data_root / f"rfdetr_{category}_{CANONICAL_SUFFIX}"
        splits = {name: fingerprint_split(dataset_dir / name) for name in SPLITS}
        train_stems = stems_of(dataset_dir / "train")
        test_stems = stems_of(dataset_dir / "test")
        valid_stems = stems_of(dataset_dir / "valid")

        entry = {
            "dataset_dir": str(dataset_dir),
            "source_paired": fingerprint_paired(paired_root / category),
            "splits": splits,
            "stem_overlap_train_test": sorted(train_stems & test_stems),
            # The project convention is that valid mirrors the official test split;
            # record whether it actually does rather than trusting the name.
            "valid_mirrors_test": splits["valid"]["label_digest"] == splits["test"]["label_digest"],
            "coverage": {
                "train_plus_test_images": len(train_stems | test_stems),
                "train_plus_test_boxes": splits["train"]["boxes"] + splits["test"]["boxes"],
            },
        }

        views: dict[str, dict] = {}
        for suffix in args.check_views:
            view_dir = data_root / f"rfdetr_{category}_{suffix}"
            if not view_dir.is_dir():
                continue
            view_test = fingerprint_split(view_dir / "test")
            views[suffix] = {
                "train_images": fingerprint_split(view_dir / "train")["images"],
                "test_matches_frozen": view_test["label_digest"] == splits["test"]["label_digest"],
            }
        if views:
            entry["derived_views"] = views

        state["categories"][category] = entry

    return state


def diff(expected: dict, actual: dict, path: str = "") -> list[str]:
    problems: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        here = f"{path}.{key}" if path else key
        if key not in expected:
            problems.append(f"{here}: added (now {actual[key]!r})")
        elif key not in actual:
            problems.append(f"{here}: removed (was {expected[key]!r})")
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            problems.extend(diff(expected[key], actual[key], here))
        elif expected[key] != actual[key]:
            problems.append(f"{here}: was {expected[key]!r}, now {actual[key]!r}")
    return problems


def report_integrity(state: dict) -> list[str]:
    """Invariants the frozen split must satisfy regardless of the lockfile."""
    problems: list[str] = []
    for category, entry in state["categories"].items():
        splits = entry["splits"]
        source = entry["source_paired"]
        coverage = entry["coverage"]

        if entry["stem_overlap_train_test"]:
            problems.append(
                f"{category}: {len(entry['stem_overlap_train_test'])} stems appear in both train and test"
            )
        if not entry["valid_mirrors_test"]:
            problems.append(f"{category}: valid does not mirror test, breaking the downstream convention")
        # train + test must account for every usable source image exactly once. The
        # paired corpus also holds the empty-label images that policy drops, so the
        # target is images_with_boxes rather than images.
        if coverage["train_plus_test_images"] != source["images_with_boxes"]:
            problems.append(
                f"{category}: train+test covers {coverage['train_plus_test_images']} images but the source has "
                f"{source['images_with_boxes']} with boxes"
            )
        if coverage["train_plus_test_boxes"] != source["boxes"]:
            problems.append(
                f"{category}: train+test holds {coverage['train_plus_test_boxes']} boxes "
                f"against {source['boxes']} in source"
            )
        for grade, count in splits["test"]["boxes_by_grade"].items():
            if count == 0:
                problems.append(f"{category}: grade {grade} is absent from the test split")
        for suffix, view in entry.get("derived_views", {}).items():
            if not view["test_matches_frozen"]:
                problems.append(
                    f"{category}: derived view {suffix} has a different test split - "
                    "its numbers are not comparable to the baseline"
                )
    return problems


def main() -> int:
    args = parse_args()
    lock_path = Path(args.lockfile)
    state = collect(args)

    integrity = report_integrity(state)
    for problem in integrity:
        print(f"INTEGRITY: {problem}")

    if args.write:
        if integrity:
            print("\nrefusing to freeze a state that fails its own invariants")
            return 1
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nfroze the canonical split for {len(CATEGORIES)} categories -> {lock_path}")
        for category, entry in state["categories"].items():
            train, test = entry["splits"]["train"], entry["splits"]["test"]
            print(
                f"  {category}: train {train['images']} imgs / {train['boxes']} boxes {train['boxes_by_grade']}"
                f" | test {test['images']} imgs / {test['boxes']} boxes {test['boxes_by_grade']}"
            )
        return 0

    if not lock_path.exists():
        print(f"no lockfile at {lock_path}; run with --write first")
        return 1
    expected = json.loads(lock_path.read_text(encoding="utf-8"))
    problems = diff(expected, state)
    if problems or integrity:
        print(f"\nDRIFT: {len(problems)} difference(s) against {lock_path}")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print(f"\nOK: frozen split matches {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
