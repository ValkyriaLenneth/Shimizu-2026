#!/usr/bin/env python3
"""Create a hard-linked Router view with selected incremental images repeated."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source.resolve(), target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filename-prefix", default="new20260807__柱脚__")
    parser.add_argument("--incremental-prefix", default="new20260807__")
    parser.add_argument("--incremental-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=20, help="total appearances per matching image")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_dir():
            (output / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            if (
                args.incremental_only
                and relative.parts[:2] in (("train", "images"), ("train", "labels"))
                and not path.name.startswith(args.incremental_prefix)
            ):
                continue
            link(path, output / relative)

    matches = sorted((source / "train" / "images").glob(f"{args.filename_prefix}*"))
    for image in matches:
        label = source / "train" / "labels" / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(label)
        for copy_index in range(1, args.repeat):
            stem = f"oversample{copy_index:02d}__{image.stem}"
            link(image, output / "train" / "images" / f"{stem}{image.suffix}")
            link(label, output / "train" / "labels" / f"{stem}.txt")

    summary = {
        "source": str(source),
        "filename_prefix": args.filename_prefix,
        "repeat": args.repeat,
        "incremental_only": args.incremental_only,
        "matched_images": len(matches),
        "added_train_images": len(matches) * (args.repeat - 1),
    }
    (output / "oversample_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
