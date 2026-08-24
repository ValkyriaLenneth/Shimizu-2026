#!/usr/bin/env python3
"""Deduplicate a folder-labelled Router image drop for Gemini annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_LABELS = {"ブレース_無損傷": "ブレース", "柱脚_無損傷": "柱脚"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    seen: dict[str, dict] = {}
    records = []
    counts = Counter()
    for path in sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        label = SOURCE_LABELS.get(path.parent.name)
        if label is None:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            records.append({"source": str(path), "sha256": digest, "duplicate_of": seen[digest]["staged"]})
            counts[f"{label}_duplicates"] += 1
            continue
        target_dir = output / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        if target.exists():
            target = target_dir / f"{digest[:10]}_{path.name}"
        os.link(path, target)
        item = {"source": str(path), "staged": str(target), "label": label, "sha256": digest}
        seen[digest] = item
        records.append(item)
        counts[f"{label}_unique"] += 1

    manifest = {
        "source_root": str(source),
        "output_root": str(output),
        "counts": dict(counts),
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_root": str(source), "output_root": str(output), "counts": dict(counts)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
