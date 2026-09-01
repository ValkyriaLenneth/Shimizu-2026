#!/usr/bin/env python3
"""Restore absolute image paths in a compact Router annotation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._"):
            index[path.name].append(path.resolve())
    return index


def resolve_image(row: dict, index: dict[str, list[Path]]) -> Path:
    basename = Path(row["image_rel_path"]).name
    expected = str(row["expected_label"])
    candidates = [
        path for path in index.get(basename, [])
        if any(part.startswith(expected) for part in path.parts)
    ]
    if not candidates:
        raise FileNotFoundError(f"no {expected} image for {basename}")
    digests = {sha256(path) for path in candidates}
    if len(digests) != 1:
        raise ValueError(f"ambiguous images with different content: {basename}")
    return candidates[0]


def main() -> int:
    args = parse_args()
    bundle = json.loads(Path(args.bundle_json).read_text(encoding="utf-8"))
    rows = bundle.get("annotations")
    if not isinstance(rows, list):
        raise ValueError("bundle does not contain an annotations list")
    index = image_index(Path(args.image_root))
    restored = []
    for row in rows:
        item = dict(row)
        item["image_path"] = str(resolve_image(row, index))
        restored.append(item)

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in restored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "restored": len(restored)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
