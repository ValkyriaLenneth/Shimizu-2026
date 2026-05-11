#!/usr/bin/env python3
"""Create a Gemini annotation copy with inner-wall/RC-wall labels fixed by source class."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUTS = [
    "outputs/gemini_balanced_300x4_3_1_pro/results.jsonl",
    "outputs/gemini_additional_200_each_no_overlap_3_1_pro/results.jsonl",
]


def get_parsed(row: dict) -> dict:
    response = row.get("response") or {}
    parsed = response.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def fix_label(expected_label: str | None, label: str) -> str:
    if expected_label == "内壁" and label == "RC壁":
        return "内壁"
    if expected_label == "RC壁" and label == "内壁":
        return "RC壁"
    return label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input Gemini JSONL files, in priority order.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gemini_wall_label_fixed_3_1_pro",
        help="Directory for fixed results.jsonl and summary.json.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "results.jsonl"
    summary_json = output_dir / "summary.json"

    seen: set[str] = set()
    rows: list[dict] = []
    summary = {
        "inputs": args.inputs,
        "output": str(output_jsonl),
        "unique_images": 0,
        "skipped_duplicates": 0,
        "element_label_changes": Counter(),
        "image_level_label_changes": Counter(),
        "expected_counts": Counter(),
    }

    for input_path in args.inputs:
        with Path(input_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("image_rel_path") or row.get("image_path")
                if key in seen:
                    summary["skipped_duplicates"] += 1
                    continue
                seen.add(key)

                expected_label = row.get("expected_label")
                summary["expected_counts"][expected_label] += 1

                parsed = get_parsed(row)
                for element in parsed.get("elements") or []:
                    label = element.get("label")
                    if not isinstance(label, str):
                        continue
                    fixed = fix_label(expected_label, label)
                    if fixed != label:
                        summary["element_label_changes"][f"{expected_label}:{label}->{fixed}"] += 1
                        element["label"] = fixed

                image_level_labels = parsed.get("image_level_labels")
                if isinstance(image_level_labels, list):
                    fixed_labels = []
                    for label in image_level_labels:
                        fixed = fix_label(expected_label, label) if isinstance(label, str) else label
                        if fixed != label:
                            summary["image_level_label_changes"][f"{expected_label}:{label}->{fixed}"] += 1
                        fixed_labels.append(fixed)
                    parsed["image_level_labels"] = sorted(set(fixed_labels), key=fixed_labels.index)

                response = row.get("response")
                if isinstance(response, dict) and parsed:
                    response["text"] = json.dumps(parsed, ensure_ascii=False, indent=2)

                row["wall_label_fix"] = {
                    "rule": "For expected_label=内壁, RC壁 labels are changed to 内壁; for expected_label=RC壁, 内壁 labels are changed to RC壁.",
                    "source_inputs": args.inputs,
                }
                rows.append(row)

    summary["unique_images"] = len(rows)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    serializable_summary = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in summary.items()
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(serializable_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
