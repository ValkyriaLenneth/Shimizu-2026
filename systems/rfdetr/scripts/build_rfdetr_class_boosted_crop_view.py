#!/usr/bin/env python3
"""Add class-targeted crops on top of an existing crop view, to fix the imbalance
that uniform crop augmentation introduces.

Uniform cropping amplifies the grades unequally. Measured on the 9:1 crop2 views:

| category | B | C | D |
|---|---|---|---|
| ブレース | 6.2x | 5.7x | 3.9x |
| 柱脚 | 4.1x | 4.0x | 3.9x |

D is amplified least, so the D:B box ratio gets *worse* after cropping - 0.63 to
0.40 for ブレース and 0.20 to 0.19 for 柱脚. The mechanism is geometric: a D box is
large (ブレース D median area is 21x a B box), so a window around it rarely also
contains other boxes, whereas a window around a tiny B box often catches a C as
well. Uniform cropping therefore favours the small-box grades.

This script layers extra crops for chosen classes onto an existing view. It reuses
`build_rfdetr_crop_aug_view.py` to generate them, then copies across only the crop
files for the requested classes, which are identifiable from the
``{stem}__crop_cls{N}_{box}_{variant}`` naming. train only; valid and test are
copied unchanged from the base view.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CROP_RE = re.compile(r"__crop_cls(\d+)_")
GRADE_NAMES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view", required=True, help="existing crop view to layer onto")
    parser.add_argument("--source-view", required=True, help="un-cropped view the extra crops come from")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boost-classes", default="2", help="comma-separated class ids to add crops for")
    parser.add_argument("--crops-per-box", type=int, default=6)
    parser.add_argument("--context", type=float, default=3.0)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            source = src / split / sub
            if not source.is_dir():
                continue
            target = dst / split / sub
            target.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                if not path.is_file():
                    continue
                out = target / path.name
                try:
                    os.link(path, out)
                except OSError:
                    shutil.copy2(path, out)


def count_grades(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for path in labels_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 5:
                counts[GRADE_NAMES.get(int(fields[0]), fields[0])] += 1
    return counts


def main() -> None:
    args = parse_args()
    base = Path(args.base_view)
    source = Path(args.source_view)
    output = Path(args.output_dir)
    boost = [int(v) for v in args.boost_classes.split(",") if v.strip()]

    for required in (base, source):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)

    before = count_grades(base / "train" / "labels")
    copy_tree(base, output)
    for extra in ("data.yaml",):
        if (base / extra).exists():
            shutil.copy2(base / extra, output / extra)

    # Generate the class-targeted crops in a scratch view, then take only those.
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as tmp:
        scratch = Path(tmp) / "boost"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("build_rfdetr_crop_aug_view.py")),
            "--source-dir", str(source),
            "--output-dir", str(scratch),
            "--target-classes", ",".join(str(v) for v in boost),
            "--crops-per-box", str(args.crops_per_box),
            "--context", str(args.context),
            "--min-size", str(args.min_size),
            "--overwrite",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"crop generation failed:\n{result.stderr[-2000:]}")

        added = 0
        images_out = output / "train" / "images"
        labels_out = output / "train" / "labels"
        for image_path in sorted((scratch / "train" / "images").iterdir()):
            match = CROP_RE.search(image_path.name)
            if not match or int(match.group(1)) not in boost:
                continue
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = scratch / "train" / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            target_image = images_out / image_path.name
            if target_image.exists():
                continue
            try:
                os.link(image_path, target_image)
            except OSError:
                shutil.copy2(image_path, target_image)
            shutil.copy2(label_path, labels_out / label_path.name)
            added += 1

    after = count_grades(output / "train" / "labels")
    summary = {
        "output_dir": str(output),
        "base_view": str(base),
        "source_view": str(source),
        "boost_classes": [GRADE_NAMES.get(v, v) for v in boost],
        "crops_per_box": args.crops_per_box,
        "context": args.context,
        "added_crop_images": added,
        "train_images": len(list((output / "train" / "images").iterdir())),
        "boxes_before": dict(sorted(before.items())),
        "boxes_after": dict(sorted(after.items())),
    }
    if before.get("B") and after.get("B"):
        summary["D_over_B_before"] = round(before.get("D", 0) / before["B"], 3)
        summary["D_over_B_after"] = round(after.get("D", 0) / after["B"], 3)
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
