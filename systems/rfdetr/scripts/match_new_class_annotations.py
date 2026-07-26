#!/usr/bin/env python3
"""Match the 2026-07-17 ブレース/柱脚 CVAT annotations 1:1 to their raw images.

The two new element categories were delivered as three independent drops:

```text
data/downloads/raw_extract/{ブレース,柱脚}/                  raw images
data/downloads/annot_extract/20260717_.../{5_ブレース,6_柱脚}/ CVAT YOLO 1.1 labels
data/downloads/annot_extract/20260724_.../..._追加分_JSCA講習より/ extra raw images
```

Filenames pair the labels to the raw images, but the raw batches contain the
same photograph more than once - byte-identical copies and, more importantly,
rescaled copies that a SHA256 pass cannot see. Duplicated photographs were
annotated independently, so keeping every copy would feed RF-DETR contradictory
supervision for identical pixels and leak content across the train/valid/test
split.

This script produces the reviewed 1:1 pairing:

1. verify the image <-> label correspondence is a bijection per category
2. cluster images by content (SHA256, then dHash + downscaled-pixel MSE)
3. elect one representative per cluster and mark the rest redundant
4. flag clusters whose duplicate copies carry disagreeing annotations
5. optionally emit a deduplicated ``images/`` + ``labels/`` pairing per category
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

DHASH_SIZE = 8
THUMB_SIZE = 32
CLASS_NAMES = {0: "B", 1: "C", 2: "D"}

# Category layout relative to --downloads-dir.
CATEGORIES: dict[str, dict[str, str]] = {
    "brace": {
        "label": "ブレース",
        "raw": "raw_extract/ブレース",
        "labels": "annot_extract/20260717_アノテーションデータ_ブレース,柱脚/5_ブレース/obj_train_data/obj_train_data",
        "names": "annot_extract/20260717_アノテーションデータ_ブレース,柱脚/5_ブレース/obj.names",
        "extra": "annot_extract/20260724_学習用データ_追加分_ブレース,柱脚/ブレース_追加分_JSCA講習より",
    },
    "column_base": {
        "label": "柱脚",
        "raw": "raw_extract/柱脚",
        "labels": "annot_extract/20260717_アノテーションデータ_ブレース,柱脚/6_柱脚/obj_train_data/obj_train_data",
        "names": "annot_extract/20260717_アノテーションデータ_ブレース,柱脚/6_柱脚/obj.names",
        "extra": "annot_extract/20260724_学習用データ_追加分_ブレース,柱脚/柱脚_追加分_JSCA講習より",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", default="data/downloads")
    parser.add_argument("--output-dir", default="outputs/new_class_annotation_match_20260724")
    parser.add_argument(
        "--dup-mse-threshold",
        type=float,
        default=2.0,
        help="Max 32x32 grayscale MSE for two images to count as the same photograph.",
    )
    parser.add_argument(
        "--dup-dhash-distance",
        type=int,
        default=4,
        help="Max dHash Hamming distance considered before the MSE check runs.",
    )
    parser.add_argument(
        "--scene-dhash-distance",
        type=int,
        default=6,
        help="Max dHash Hamming distance for a weaker same-scene group used only "
        "to keep near-identical photographs inside one split.",
    )
    parser.add_argument(
        "--emit-paired-dir",
        default="",
        help="Optional directory to hardlink the deduplicated image/label pairs into.",
    )
    parser.add_argument(
        "--keep-redundant-in-paired-dir",
        action="store_true",
        help="Emit every copy instead of representatives only (debugging).",
    )
    return parser.parse_args()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_fingerprint(path: Path) -> tuple[int, list[int], tuple[int, int]]:
    """Return (dHash bits, 32x32 grayscale thumbnail, original size)."""
    with Image.open(path) as image:
        size = image.size
        gray = image.convert("L")
        rows = gray.resize((DHASH_SIZE + 1, DHASH_SIZE), Image.LANCZOS)
        thumb = gray.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        row_pixels = list(rows.getdata())
        thumb_pixels = list(thumb.getdata())

    bits = 0
    index = 0
    for row in range(DHASH_SIZE):
        offset = row * (DHASH_SIZE + 1)
        for col in range(DHASH_SIZE):
            if row_pixels[offset + col] < row_pixels[offset + col + 1]:
                bits |= 1 << index
            index += 1
    return bits, thumb_pixels, size


def mean_squared_error(left: list[int], right: list[int]) -> float:
    total = sum((a - b) ** 2 for a, b in zip(left, right))
    return total / len(left)


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Parse a YOLO label file into boxes plus a list of malformed-line messages."""
    boxes: list[tuple[int, float, float, float, float]] = []
    problems: list[str] = []
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            problems.append(f"line {number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError:
            problems.append(f"line {number}: non-numeric field")
            continue
        if class_id not in CLASS_NAMES:
            problems.append(f"line {number}: unknown class id {class_id}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            problems.append(f"line {number}: coordinates outside [0, 1]")
            continue
        if width <= 0.0 or height <= 0.0:
            problems.append(f"line {number}: non-positive box size")
            continue
        boxes.append((class_id, cx, cy, width, height))
    return boxes, problems


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[left_root] = right_root

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            result[self.find(item)].append(item)
        return {root: sorted(members) for root, members in result.items()}


def collect_records(downloads_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read every image, pair it with its label, and report pairing integrity."""
    records: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}

    for category, layout in CATEGORIES.items():
        raw_dir = downloads_dir / layout["raw"]
        labels_dir = downloads_dir / layout["labels"]
        extra_dir = downloads_dir / layout["extra"]
        names_file = downloads_dir / layout["names"]
        for required in (raw_dir, labels_dir):
            if not required.is_dir():
                raise FileNotFoundError(required)

        declared_names = (
            [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if names_file.exists()
            else []
        )

        images = {path.stem: path for path in sorted(raw_dir.glob("*.jpg"))}
        extras = {path.stem: path for path in sorted(extra_dir.glob("*.jpg"))} if extra_dir.is_dir() else {}
        labels = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}

        integrity[category] = {
            "label": layout["label"],
            "declared_class_names": declared_names,
            "annotated_images": len(images),
            "label_files": len(labels),
            "extra_unlabelled_images": len(extras),
            "images_without_label": sorted(set(images) - set(labels)),
            "labels_without_image": sorted(set(labels) - set(images)),
            "extra_stems_colliding_with_annotated": sorted(set(extras) & set(images)),
            "extra_stems_with_label": sorted(set(extras) & set(labels)),
        }
        integrity[category]["is_bijection"] = (
            not integrity[category]["images_without_label"] and not integrity[category]["labels_without_image"]
        )

        for source, pool in (("annotated", images), ("extra_unlabelled", extras)):
            for stem, image_path in pool.items():
                label_path = labels.get(stem) if source == "annotated" else None
                boxes, problems = parse_label_file(label_path) if label_path else ([], [])
                bits, thumb, size = image_fingerprint(image_path)
                records.append(
                    {
                        "key": f"{category}/{stem}",
                        "category": category,
                        "category_label": layout["label"],
                        "stem": stem,
                        "source": source,
                        "image_path": str(image_path),
                        "label_path": str(label_path) if label_path else "",
                        "width": size[0],
                        "height": size[1],
                        "sha256": sha256_of(image_path),
                        "dhash": bits,
                        "_thumb": thumb,
                        "boxes": boxes,
                        "box_count": len(boxes),
                        "class_counts": {CLASS_NAMES[c]: n for c, n in sorted(Counter(b[0] for b in boxes).items())},
                        "label_problems": problems,
                    }
                )

    return records, integrity


def cluster_records(
    records: list[dict[str, Any]],
    dup_mse_threshold: float,
    dup_dhash_distance: int,
    scene_dhash_distance: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[dict[str, Any]]]:
    """Cluster duplicates within a category and collect cross-category matches."""
    by_key = {record["key"]: record for record in records}
    duplicates = UnionFind()
    scenes = UnionFind()
    for key in by_key:
        duplicates.add(key)
        scenes.add(key)

    cross_category: list[dict[str, Any]] = []
    per_category: dict[str, list[str]] = defaultdict(list)
    for record in records:
        per_category[record["category"]].append(record["key"])

    def compare(left_key: str, right_key: str) -> tuple[int, float]:
        left, right = by_key[left_key], by_key[right_key]
        distance = bin(left["dhash"] ^ right["dhash"]).count("1")
        error = (
            0.0
            if left["sha256"] == right["sha256"]
            else mean_squared_error(left["_thumb"], right["_thumb"])
        )
        return distance, error

    # Duplicate and same-scene clustering stay inside a category: the two
    # categories become two independent RF-DETR datasets, so grouping across
    # them would only distort each split.
    for keys in per_category.values():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                left_key, right_key = keys[i], keys[j]
                if by_key[left_key]["sha256"] == by_key[right_key]["sha256"]:
                    duplicates.union(left_key, right_key)
                    scenes.union(left_key, right_key)
                    continue
                distance, error = compare(left_key, right_key)
                if distance <= dup_dhash_distance and error <= dup_mse_threshold:
                    duplicates.union(left_key, right_key)
                if distance <= scene_dhash_distance:
                    scenes.union(left_key, right_key)

    # Cross-category identical content is reported, never merged: the same
    # photograph filed under both categories is a delivery question for the
    # annotation team.
    category_names = sorted(per_category)
    for index, left_category in enumerate(category_names):
        for right_category in category_names[index + 1 :]:
            for left_key in per_category[left_category]:
                for right_key in per_category[right_category]:
                    if by_key[left_key]["sha256"] == by_key[right_key]["sha256"]:
                        cross_category.append({"pair": [left_key, right_key], "match": "sha256"})
                        continue
                    distance, error = compare(left_key, right_key)
                    if distance <= dup_dhash_distance and error <= dup_mse_threshold:
                        cross_category.append(
                            {"pair": [left_key, right_key], "match": "rescaled", "mse": round(error, 3)}
                        )

    return duplicates.groups(), scenes.groups(), cross_category


def elect_representatives(
    records: list[dict[str, Any]],
    duplicate_groups: dict[str, list[str]],
    scene_groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Assign cluster ids, elect representatives, and flag annotation conflicts."""
    by_key = {record["key"]: record for record in records}

    scene_id_of: dict[str, str] = {}
    for index, (_, members) in enumerate(sorted(scene_groups.items(), key=lambda item: item[1][0])):
        scene_id = f"scene-{index:04d}"
        for key in members:
            scene_id_of[key] = scene_id

    conflicts: list[dict[str, Any]] = []
    for index, (_, members) in enumerate(sorted(duplicate_groups.items(), key=lambda item: item[1][0])):
        cluster_id = f"dup-{index:04d}"
        group = [by_key[key] for key in members]

        # Most complete annotation wins; ties go to the highest-resolution copy,
        # then to the lowest stem so the choice is reproducible.
        ranked = sorted(
            group,
            key=lambda record: (
                -record["box_count"],
                -(record["width"] * record["height"]),
                record["stem"],
            ),
        )
        representative = ranked[0]

        annotated = [record for record in group if record["source"] == "annotated"]
        counts = {record["box_count"] for record in annotated}
        signatures = {tuple(sorted(record["boxes"])) for record in annotated}
        positive = [record for record in annotated if record["box_count"] > 0]

        if len(annotated) > 1 and len(counts) > 1 and len(positive) == len(annotated):
            # Every copy was annotated, but with a different number of boxes:
            # a genuine annotator disagreement that a human must settle.
            severity = "disagreement"
        elif len(annotated) > 1 and len(positive) not in (0, len(annotated)):
            # One copy annotated, its twin left empty - the empty copy would
            # otherwise become a false background sample.
            severity = "unannotated_duplicate"
        elif len(annotated) > 1 and len(signatures) > 1:
            severity = "coordinate_drift"
        else:
            severity = ""

        if severity:
            conflicts.append(
                {
                    "cluster_id": cluster_id,
                    "category": representative["category"],
                    "severity": severity,
                    "chosen": representative["key"],
                    "members": [
                        {
                            "key": record["key"],
                            "source": record["source"],
                            "size": f"{record['width']}x{record['height']}",
                            "box_count": record["box_count"],
                            "class_counts": record["class_counts"],
                        }
                        for record in sorted(group, key=lambda item: item["stem"])
                    ],
                }
            )

        for record in group:
            record["duplicate_cluster_id"] = cluster_id
            record["duplicate_cluster_size"] = len(group)
            record["scene_group_id"] = scene_id_of[record["key"]]
            record["is_representative"] = record["key"] == representative["key"]
            record["duplicate_of"] = "" if record["is_representative"] else representative["key"]
            record["cluster_conflict"] = severity

    return conflicts


def summarize(records: list[dict[str, Any]], integrity: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"categories": {}}
    for category, layout in CATEGORIES.items():
        rows = [record for record in records if record["category"] == category]
        annotated = [record for record in rows if record["source"] == "annotated"]
        kept = [record for record in annotated if record["is_representative"]]
        dropped = [record for record in annotated if not record["is_representative"]]
        extra = [record for record in rows if record["source"] == "extra_unlabelled"]

        kept_classes: Counter[str] = Counter()
        for record in kept:
            kept_classes.update(record["class_counts"])
        dropped_boxes = sum(record["box_count"] for record in dropped)

        summary["categories"][category] = {
            "label": layout["label"],
            "pairing_is_bijection": integrity[category]["is_bijection"],
            "annotated_images_delivered": len(annotated),
            "unique_images_after_dedup": len(kept),
            "redundant_copies_dropped": len(dropped),
            "boxes_delivered": sum(record["box_count"] for record in annotated),
            "boxes_on_unique_images": sum(record["box_count"] for record in kept),
            "boxes_on_dropped_copies": dropped_boxes,
            "class_distribution_after_dedup": dict(sorted(kept_classes.items())),
            "empty_labels_delivered": sum(1 for record in annotated if record["box_count"] == 0),
            "empty_labels_after_dedup": sum(1 for record in kept if record["box_count"] == 0),
            "scene_groups_after_dedup": len({record["scene_group_id"] for record in kept}),
            "extra_unlabelled_images": len(extra),
            "malformed_label_lines": sum(len(record["label_problems"]) for record in annotated),
        }
    return summary


def emit_paired_dir(records: list[dict[str, Any]], target: Path, keep_redundant: bool) -> dict[str, int]:
    """Hardlink each kept image next to a label file of the same stem."""
    written: dict[str, int] = {}
    for category in CATEGORIES:
        images_dir = target / category / "images"
        labels_dir = target / category / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for record in records:
            if record["category"] != category or record["source"] != "annotated":
                continue
            if not keep_redundant and not record["is_representative"]:
                continue

            image_source = Path(record["image_path"])
            image_target = images_dir / f"{record['stem']}{image_source.suffix}"
            if image_target.exists() or image_target.is_symlink():
                image_target.unlink()
            try:
                image_target.hardlink_to(image_source)
            except OSError:
                image_target.write_bytes(image_source.read_bytes())

            lines = [
                f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
                for class_id, cx, cy, width, height in record["boxes"]
            ]
            body = "\n".join(lines)
            (labels_dir / f"{record['stem']}.txt").write_text(body + "\n" if body else "", encoding="utf-8")
            count += 1
        written[category] = count
    return written


def main() -> None:
    args = parse_args()
    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, integrity = collect_records(downloads_dir)
    duplicate_groups, scene_groups, cross_category = cluster_records(
        records, args.dup_mse_threshold, args.dup_dhash_distance, args.scene_dhash_distance
    )
    conflicts = elect_representatives(records, duplicate_groups, scene_groups)
    summary = summarize(records, integrity)
    summary["cross_category_duplicate_pairs"] = cross_category
    summary["conflict_counts"] = dict(Counter(item["severity"] for item in conflicts))
    summary["thresholds"] = {
        "dup_mse_threshold": args.dup_mse_threshold,
        "dup_dhash_distance": args.dup_dhash_distance,
        "scene_dhash_distance": args.scene_dhash_distance,
    }

    for record in records:
        record.pop("_thumb", None)

    ordered = sorted(records, key=lambda record: (record["category"], record["stem"]))
    (output_dir / "manifest.json").write_text(
        json.dumps({"summary": summary, "integrity": integrity, "records": ordered}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "match_summary.json").write_text(
        json.dumps({"summary": summary, "integrity": integrity}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "duplicate_conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = [
        "category",
        "category_label",
        "stem",
        "source",
        "image_path",
        "label_path",
        "width",
        "height",
        "sha256",
        "box_count",
        "duplicate_cluster_id",
        "duplicate_cluster_size",
        "is_representative",
        "duplicate_of",
        "scene_group_id",
        "cluster_conflict",
    ]
    with (output_dir / "image_label_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in ordered:
            writer.writerow({key: record.get(key, "") for key in columns})

    if args.emit_paired_dir:
        written = emit_paired_dir(records, Path(args.emit_paired_dir), args.keep_redundant_in_paired_dir)
        summary["paired_dir"] = {"path": args.emit_paired_dir, "written": written}
        (output_dir / "match_summary.json").write_text(
            json.dumps({"summary": summary, "integrity": integrity}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for category, info in integrity.items():
        state = "bijection OK" if info["is_bijection"] else "PAIRING BROKEN"
        print(f"[{category}] {info['label']}: {state}")
    print(f"conflict clusters needing review: {len(conflicts)}")
    print(f"wrote manifest to {output_dir}")


if __name__ == "__main__":
    main()
