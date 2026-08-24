#!/usr/bin/env python3
"""Merge a general Gemini pass with focused retries for missing Router boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[str, str]:
    return str(row["expected_label"]), Path(row["image_path"]).name


def has_expected(row: dict) -> bool:
    elements = ((row.get("response") or {}).get("parsed") or {}).get("elements", []) or []
    return any(item.get("label") == row.get("expected_label") for item in elements if isinstance(item, dict))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--retry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded-output", required=True)
    args = parser.parse_args()

    primary = {key(row): row for row in load(Path(args.primary))}
    retry = {key(row): row for row in load(Path(args.retry))}
    merged = []
    excluded = []
    replacements = 0
    for item_key, row in sorted(primary.items()):
        selected = row
        if not has_expected(row) and item_key in retry and has_expected(retry[item_key]):
            selected = retry[item_key]
            replacements += 1
        if has_expected(selected):
            merged.append(selected)
        else:
            excluded.append(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.excluded_output).write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary": len(primary), "retry": len(retry), "replacements": replacements,
                      "merged": len(merged), "excluded": len(excluded)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
