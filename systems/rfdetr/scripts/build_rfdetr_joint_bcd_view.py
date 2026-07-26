#!/usr/bin/env python3
"""Merge several B/C/D category datasets into one joint view for pretraining.

Both new categories carry the same three classes - damage grade B, C and D - so a
single model can be pretrained on their union and then fine-tuned per category.
Deployment stays one model per category; only the initialization is shared.

The point is data volume. ブレース and 柱脚 each have roughly a third of the
training data of the delivered categories, but their grading semantics are the
same, so the union is a legitimate pretraining corpus.

Train splits are merged. valid/test are also merged, but only so RF-DETR has
loaders during pretraining - the per-category test splits remain the reporting
basis, and the fine-tuned models are what get evaluated.

Stems are prefixed with the source category so two categories can never collide.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")
GRADE_NAMES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Category name and dataset dir, repeatable, e.g. brace=data/rfdetr_brace_...",
    )
    parser.add_argument("--output-dir", required=True)
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


def main() -> None:
    args = parse_args()
    sources: dict[str, Path] = {}
    for item in args.source:
        if "=" not in item:
            raise ValueError(f"--source expects NAME=DIR, got {item!r}")
        name, path = item.split("=", 1)
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        sources[name] = directory

    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)

    summary: dict[str, object] = {
        "output_dir": str(output),
        "sources": {name: str(path) for name, path in sources.items()},
        "note": "joint B/C/D pretraining corpus; per-category test splits remain the reporting basis",
        "splits": {},
    }

    for split in SPLITS:
        images_out = output / split / "images"
        labels_out = output / split / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        counts: Counter[str] = Counter()
        per_source: dict[str, int] = {}
        boxes = 0
        for name, directory in sources.items():
            src_images = directory / split / "images"
            src_labels = directory / split / "labels"
            if not src_images.is_dir():
                per_source[name] = 0
                continue
            written = 0
            for image_path in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                label_path = src_labels / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                stem = f"{name}__{image_path.stem}"
                link_file(image_path, images_out / f"{stem}{image_path.suffix}", args.link_mode)
                text = label_path.read_text(encoding="utf-8")
                (labels_out / f"{stem}.txt").write_text(text, encoding="utf-8")
                for line in text.splitlines():
                    fields = line.split()
                    if len(fields) == 5:
                        counts[GRADE_NAMES.get(int(fields[0]), fields[0])] += 1
                        boxes += 1
                written += 1
            per_source[name] = written

        summary["splits"][split] = {
            "images": sum(per_source.values()),
            "boxes": boxes,
            "boxes_by_grade": dict(sorted(counts.items())),
            "images_per_source": per_source,
        }

    names = {index: f"損傷程度{grade}" for index, grade in GRADE_NAMES.items()}
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output),
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
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
