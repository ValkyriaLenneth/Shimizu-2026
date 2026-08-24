#!/usr/bin/env python3
"""Stage images whose first Gemini pass omitted the expected Router class."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    count = 0
    with Path(args.results).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            elements = ((row.get("response") or {}).get("parsed") or {}).get("elements", []) or []
            if any(item.get("label") == row.get("expected_label") for item in elements if isinstance(item, dict)):
                continue
            source = Path(row["image_path"])
            target_dir = output / row["expected_label"]
            target_dir.mkdir(parents=True, exist_ok=True)
            os.link(source, target_dir / source.name)
            count += 1
    print(json.dumps({"staged_missing": count, "output_root": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
