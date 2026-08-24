#!/usr/bin/env python3
"""Append reviewed Gemini Router boxes to train while preserving baseline tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from build_rfdetr_router_5class_dataset import IMAGE_EXTS, NEW_LABEL_TO_CLASS, NAMES, row_to_yolo_lines


def link_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)


def sequence_group(row: dict) -> str:
    path = Path(row["image_path"])
    match = re.search(r"(\d+)$", path.stem)
    block = int(match.group(1)) // 10 if match else path.stem
    return f"{row['expected_label']}:{block}"


def is_holdout(group: str, seed: int, fraction: float) -> bool:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
    return value < fraction


def image_hashes(root: Path) -> set[str]:
    result = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._"):
            result.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--gemini-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-dir", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dataset).resolve()
    output = Path(args.output_dir).resolve()
    holdout = Path(args.holdout_dir).resolve()
    for target in (output, holdout):
        if target.exists():
            if not args.overwrite:
                raise FileExistsError(target)
            shutil.rmtree(target)
    output.mkdir(parents=True)
    link_tree(base, output)
    for subdir in (holdout / "test" / "images", holdout / "test" / "labels"):
        subdir.mkdir(parents=True, exist_ok=True)

    base_hashes = image_hashes(base)
    rows = []
    with Path(args.gemini_results).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    counts = Counter()
    records = []
    for row in sorted(rows, key=lambda item: (item["expected_label"], item["image_path"])):
        source = Path(row["image_path"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in base_hashes:
            counts["excluded_duplicate_with_base"] += 1
            continue
        lines = row_to_yolo_lines(row, "expected-only", args.min_confidence)
        if not lines:
            counts["excluded_no_valid_box"] += 1
            continue
        split = "holdout" if is_holdout(sequence_group(row), args.seed, args.holdout_fraction) else "train"
        prefix = f"new20260807__{row['expected_label']}__"
        filename = prefix + source.name
        if split == "train":
            image_target = output / "train" / "images" / filename
            label_target = output / "train" / "labels" / f"{Path(filename).stem}.txt"
        else:
            image_target = holdout / "test" / "images" / filename
            label_target = holdout / "test" / "labels" / f"{Path(filename).stem}.txt"
        os.link(source, image_target)
        label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[f"{split}_{row['expected_label']}_images"] += 1
        counts[f"{split}_{row['expected_label']}_boxes"] += len(lines)
        records.append({"source": str(source), "sha256": digest, "split": split,
                        "expected_label": row["expected_label"], "boxes": len(lines)})

    yaml_text = "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n" + "".join(
        f"  {index}: {name}\n" for index, name in enumerate(NAMES)
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    holdout_yaml = "path: .\ntrain: test/images\nval: test/images\ntest: test/images\nnames:\n" + "".join(
        f"  {index}: {name}\n" for index, name in enumerate(NAMES)
    )
    (holdout / "data.yaml").write_text(holdout_yaml, encoding="utf-8")
    manifest = {"base_dataset": str(base), "gemini_results": args.gemini_results,
                "seed": args.seed, "holdout_fraction": args.holdout_fraction,
                "min_confidence": args.min_confidence, "counts": dict(counts), "records": records}
    (output / "incremental_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
