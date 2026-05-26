#!/usr/bin/env python3
"""Annotate generated POC images with Gemini coarse building-element labels.

Reuses the same prompt and worker as the existing real-image Gemini coarse
annotation flow (`scripts/run_gemini_coarse_test.py` /
`scripts/test_gemini_concurrency.py`) so the output is shape-compatible with
`scripts/visualize_gemini_coarse.py`.

Input is the `generation_results.jsonl` produced by
`scripts/run_gemini_image_generation_poc.py`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_gemini_concurrency import append_jsonl, load_done, utc_now, worker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-results",
        default="outputs/synthetic_router_generation_poc/generation_results.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/synthetic_router_generation_poc",
    )
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_generation_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok") or not row.get("image_path"):
                continue
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY before running.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "annotation_results.jsonl"

    generation_rows = load_generation_rows(Path(args.generation_results))
    if args.limit:
        generation_rows = generation_rows[: args.limit]
    completed = load_done([results_path])
    plan: list[dict] = []
    for row in generation_rows:
        if row["image_path"] in completed:
            continue
        plan.append(
            {
                "expected_dir": row.get("expected_dir"),
                "expected_label": row.get("expected_label"),
                "image_path": row["image_path"],
                "image_rel_path": row.get("image_rel_path") or row["image_path"],
                "prompt_id": row.get("prompt_id"),
                "primary_class": row.get("primary_class"),
                "target_classes": row.get("target_classes"),
                "scenario": row.get("scenario"),
                "generation_model": row.get("model"),
            }
        )
    print(
        f"generation_rows={len(generation_rows)} pending_annotations={len(plan)} concurrency={args.concurrency}",
        flush=True,
    )
    if not plan:
        return 0

    lock = threading.Lock()
    started = time.monotonic()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(worker, row, api_key, args.model, args.timeout, args.max_retries): row
            for row in plan
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["source"] = "gemini_api_synthetic"
            result["imported"] = False
            append_jsonl(results_path, result, lock)
            done += 1
            elapsed = time.monotonic() - started
            rate = done / elapsed * 60 if elapsed else 0
            status = "ok" if result.get("ok") else "err"
            print(
                f"{done}/{len(plan)} {status} rate={rate:.2f}/min last={result.get('image_rel_path')}",
                flush=True,
            )

    summary = summarize(results_path, args)
    (out_dir / "annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize(results_path: Path, args: argparse.Namespace) -> dict:
    rows: list[dict] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    ok = sum(1 for r in rows if r.get("ok"))
    errors = len(rows) - ok
    by_label: dict[str, int] = {}
    label_match: dict[str, int] = {}
    label_total: dict[str, int] = {}
    for row in rows:
        label = row.get("expected_label") or ""
        label_total[label] = label_total.get(label, 0) + 1
        if not row.get("ok"):
            continue
        parsed = (row.get("response") or {}).get("parsed") or {}
        elements = parsed.get("elements", []) or []
        detected = {element.get("label") for element in elements if isinstance(element, dict)}
        for detected_label in detected:
            if detected_label:
                by_label[detected_label] = by_label.get(detected_label, 0) + 1
        if label in detected:
            label_match[label] = label_match.get(label, 0) + 1
    coverage = {
        label: {
            "annotated": label_total.get(label, 0),
            "matched_primary_class": label_match.get(label, 0),
        }
        for label in label_total
    }
    return {
        "model": args.model,
        "concurrency": args.concurrency,
        "finished_at": utc_now(),
        "total": len(rows),
        "ok": ok,
        "errors": errors,
        "detected_label_counts": by_label,
        "primary_class_coverage": coverage,
        "results": str(results_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
