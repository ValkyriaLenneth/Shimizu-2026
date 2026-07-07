#!/usr/bin/env python3
"""Build a deduplicated review queue and migrate existing manual reviews."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = Path(__file__).with_name("app.py")
SOURCE_REVIEW = ROOT / "outputs/gemini_new_router_classes_20260630/manual_review/review_annotations.json"
DEDUP_DIR = ROOT / "outputs/gemini_new_router_classes_20260630/manual_review_dedup"
DEDUP_ITEMS = DEDUP_DIR / "dedup_items.json"
DEDUP_REVIEW = DEDUP_DIR / "review_annotations.json"
DEDUP_SUMMARY = DEDUP_DIR / "dedup_summary.json"
DEDUP_GROUPS = DEDUP_DIR / "dedup_groups.json"


def load_app_module():
    spec = importlib.util.spec_from_file_location("gemini_review_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_canonical(group: list[dict], reviewed_ids: set[str]) -> dict:
    reviewed = [item for item in group if item["id"] in reviewed_ids]
    if reviewed:
        return min(reviewed, key=lambda item: int(item["id"]))
    ok_items = [item for item in group if item.get("ok")]
    candidates = ok_items or group
    return min(candidates, key=lambda item: int(item["id"]))


def clone_review_for_item(review: dict, source_item: dict, target_item: dict) -> dict:
    cloned = json.loads(json.dumps(review, ensure_ascii=False))
    cloned["file_name"] = target_item["file_name"]
    cloned["expected_label"] = target_item["expected_label"]
    cloned["dedup_source_id"] = source_item["id"]
    cloned["dedup_source_file_name"] = source_item["file_name"]
    return cloned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed-through-line",
        type=int,
        default=174,
        help="Only migrate manual reviews from original JSONL lines <= this value.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if DEDUP_DIR.exists():
        if not args.overwrite:
            raise FileExistsError(f"{DEDUP_DIR} exists; pass --overwrite")
        shutil.rmtree(DEDUP_DIR)
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)

    app = load_app_module()
    state = app.make_state()
    items = state["items"]
    source_review = {}
    if SOURCE_REVIEW.exists():
        source_review = json.loads(SOURCE_REVIEW.read_text(encoding="utf-8"))

    eligible_review_ids = {
        item["id"]
        for item in items
        if item["id"] in source_review and int(item["line"]) <= args.reviewed_through_line
    }

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    hashes_by_id: dict[str, str] = {}
    for item in items:
        image_hash = sha256_file(Path(item["image_path"]))
        hashes_by_id[item["id"]] = image_hash
        groups[(item["expected_label"], image_hash)].append(item)

    dedup_items = []
    dedup_review = {}
    group_records = []
    migrated_review_count = 0
    duplicate_review_count = 0

    sorted_groups = sorted(groups.items(), key=lambda kv: min(int(item["id"]) for item in kv[1]))
    for dedup_index, ((label, image_hash), group) in enumerate(sorted_groups):
        canonical = choose_canonical(group, eligible_review_ids)
        canonical_copy = json.loads(json.dumps(canonical, ensure_ascii=False))
        old_id = canonical_copy["id"]
        canonical_copy["id"] = str(dedup_index)
        canonical_copy["dedup"] = {
            "sha256": image_hash,
            "canonical_source_id": old_id,
            "canonical_source_line": canonical["line"],
            "duplicate_count": len(group),
            "source_ids": [item["id"] for item in sorted(group, key=lambda item: int(item["id"]))],
            "source_lines": [item["line"] for item in sorted(group, key=lambda item: int(item["id"]))],
            "source_file_names": [item["file_name"] for item in sorted(group, key=lambda item: int(item["id"]))],
        }
        dedup_items.append(canonical_copy)

        reviewed_in_group = [
            item for item in sorted(group, key=lambda item: int(item["id"]))
            if item["id"] in eligible_review_ids
        ]
        if reviewed_in_group:
            source_item = reviewed_in_group[0]
            dedup_review[str(dedup_index)] = clone_review_for_item(
                source_review[source_item["id"]],
                source_item,
                canonical_copy,
            )
            migrated_review_count += 1
            duplicate_review_count += max(0, len(reviewed_in_group) - 1)

        group_records.append(
            {
                "dedup_id": str(dedup_index),
                "expected_label": label,
                "sha256": image_hash,
                "canonical_source_id": old_id,
                "canonical_file_name": canonical["file_name"],
                "source_items": [
                    {
                        "id": item["id"],
                        "line": item["line"],
                        "file_name": item["file_name"],
                        "image_path": item["image_path"],
                        "review_migrated": item["id"] in eligible_review_ids,
                    }
                    for item in sorted(group, key=lambda item: int(item["id"]))
                ],
            }
        )

    summary = {
        "source_rows": len(items),
        "dedup_items": len(dedup_items),
        "removed_duplicate_rows": len(items) - len(dedup_items),
        "duplicate_groups": sum(1 for group in groups.values() if len(group) > 1),
        "by_label": dict(Counter(item["expected_label"] for item in dedup_items)),
        "source_review_count": len(source_review),
        "eligible_review_count": len(eligible_review_ids),
        "migrated_review_count": migrated_review_count,
        "duplicate_review_count_ignored": duplicate_review_count,
        "reviewed_through_line": args.reviewed_through_line,
        "dedup_items": str(DEDUP_ITEMS.relative_to(ROOT)),
        "dedup_review": str(DEDUP_REVIEW.relative_to(ROOT)),
        "dedup_groups": str(DEDUP_GROUPS.relative_to(ROOT)),
    }

    DEDUP_ITEMS.write_text(json.dumps(dedup_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEDUP_REVIEW.write_text(json.dumps(dedup_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEDUP_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEDUP_GROUPS.write_text(json.dumps(group_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
