#!/usr/bin/env python3
"""Build a reproducible RF-DETR pipeline evaluation split from release data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMPONENTS = ["tenjo", "inner_wall", "rc_wall", "rc_column"]
SPLIT_PART = {
    "tenjo": "ceiling",
    "inner_wall": "interior",
    "rc_wall": "rc_wall",
    "rc_column": "rc_column",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        default="final_release_20260615",
        help="Extracted final release directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pipeline_eval_official_plus_20260623",
        help="Destination split directory.",
    )
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--additional-ratio",
        type=float,
        default=1.0,
        help="Additional non-official samples per official sample.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    release_root = resolve(Path(args.release_root), repo)
    output_dir = resolve(Path(args.output_dir), repo)
    source_root = (
        release_root
        / "data/final_download_20260526/handoff_20260519"
        / "shimizu_20260519_minimal_repro_package"
    )
    manifest_path = source_root / "data/final_crack_yolo_20260519/manifest.csv"
    split_json_path = release_root / "data/data_split.json"

    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "labels").mkdir(parents=True)

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    split_json = json.loads(split_json_path.read_text(encoding="utf-8"))
    selected = select_rows(rows, split_json, args.seed, args.additional_ratio)
    materialize(selected, source_root, output_dir, args.link_mode)
    write_outputs(selected, source_root, output_dir, args, manifest_path, split_json_path)

    summary = summarize(selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def resolve(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else (repo / path).resolve()


def select_rows(
    rows: list[dict[str, str]],
    split_json: dict[str, Any],
    seed: int,
    additional_ratio: float,
) -> list[dict[str, str]]:
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_component_stem: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        component = row["class_key"]
        if component not in COMPONENTS:
            continue
        by_component[component].append(row)
        by_component_stem[(component, row["original_stem"])] = row

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for component in COMPONENTS:
        part = SPLIT_PART[component]
        official_stems = split_json["parts"][part]["test"]
        official_rows = []
        missing = []
        for stem in official_stems:
            row = by_component_stem.get((component, stem))
            if row is None:
                missing.append(stem)
                continue
            official_rows.append(with_eval_meta(row, "official_test"))
        official_keys = {row["output_stem"] for row in official_rows}
        pool = [
            row
            for row in by_component[component]
            if row["output_stem"] not in official_keys
        ]
        pool = sorted(pool, key=lambda r: (r["source"], r["final_split"], r["output_stem"]))
        rng.shuffle(pool)
        add_count = int(round(len(official_rows) * additional_ratio))
        additional_rows = [with_eval_meta(row, "additional_holdout") for row in pool[:add_count]]
        selected.extend(official_rows + additional_rows)
        if missing:
            print(f"[WARN] {component}: missing official stems: {missing}")
    return selected


def with_eval_meta(row: dict[str, str], eval_group: str) -> dict[str, str]:
    out = dict(row)
    out["eval_group"] = eval_group
    return out


def materialize(
    rows: list[dict[str, str]],
    source_root: Path,
    output_dir: Path,
    link_mode: str,
) -> None:
    for row in rows:
        src_img = source_root / row["image"]
        src_lbl = source_root / row["label"]
        image_name = f"{row['class_key']}__{row['output_stem']}{src_img.suffix.lower()}"
        label_name = f"{row['class_key']}__{row['output_stem']}.txt"
        dst_img = output_dir / "images" / image_name
        dst_lbl = output_dir / "labels" / label_name
        place(src_img, dst_img, link_mode)
        place(src_lbl, dst_lbl, link_mode)
        row["eval_image"] = str(dst_img)
        row["eval_label"] = str(dst_lbl)
        row["eval_image_name"] = image_name
        row["eval_label_name"] = label_name


def place(src: Path, dst: Path, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src, dst)


def write_outputs(
    rows: list[dict[str, str]],
    source_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    split_json_path: Path,
) -> None:
    fieldnames = [
        "eval_group",
        "class_key",
        "class_display",
        "source",
        "source_split",
        "final_split",
        "original_stem",
        "output_stem",
        "box_count",
        "eval_image_name",
        "eval_label_name",
        "eval_image",
        "eval_label",
        "image",
        "label",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    split = {
        "metadata": {
            "seed": args.seed,
            "additional_ratio": args.additional_ratio,
            "link_mode": args.link_mode,
            "source_root": str(source_root),
            "source_manifest": str(manifest_path),
            "source_split_json": str(split_json_path),
        },
        "samples": rows,
        "summary": summarize(rows),
    }
    (output_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(split["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Pipeline Eval Official Plus 2026-06-23",
                "",
                "Reproducible pipeline evaluation split built from the extracted final release data.",
                "",
                f"- seed: `{args.seed}`",
                f"- additional_ratio: `{args.additional_ratio}`",
                f"- source_manifest: `{manifest_path}`",
                f"- source_split_json: `{split_json_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_component: dict[str, Counter[str]] = {component: Counter() for component in COMPONENTS}
    total_boxes: dict[str, int] = {component: 0 for component in COMPONENTS}
    for row in rows:
        component = row["class_key"]
        by_component[component][row["eval_group"]] += 1
        total_boxes[component] += int(row.get("box_count") or 0)
    return {
        "total_images": len(rows),
        "by_component": {
            component: {
                "images": sum(by_component[component].values()),
                "groups": dict(by_component[component]),
                "boxes": total_boxes[component],
            }
            for component in COMPONENTS
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
