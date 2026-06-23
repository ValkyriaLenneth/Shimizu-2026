#!/usr/bin/env python3
"""Run Gemini coarse building-element annotation for a manifest subset."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_gemini_concurrency import append_jsonl, load_done, utc_now, worker


CLASS_KEY_TO_EXPECTED = {
    "tenjo": ("1_天井", "天井"),
    "inner_wall": ("2_内壁", "内壁"),
    "rc_wall": ("3_RC壁", "RC壁"),
    "rc_column": ("4_RC柱", "RC柱"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/final_crack_yolo_20260519/manifest.csv")
    parser.add_argument("--source", default="data_add100")
    parser.add_argument("--out-dir", default="outputs/gemini_data_add100_3_1_pro_preview_2026-05-19")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exclude-results", action="append", default=[])
    return parser.parse_args()


def build_plan(manifest: Path, source: str, limit: int) -> list[dict[str, str]]:
    rows = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["source"] != source:
                continue
            expected_dir, expected_label = CLASS_KEY_TO_EXPECTED[row["class_key"]]
            image_path = Path(row["image"])
            rows.append(
                {
                    "expected_dir": expected_dir,
                    "expected_label": expected_label,
                    "image_path": str(image_path),
                    "image_rel_path": row["image"],
                    "manifest_class_key": row["class_key"],
                    "manifest_source": row["source"],
                    "manifest_source_split": row["source_split"],
                    "manifest_final_split": row["final_split"],
                    "manifest_label": row["label"],
                    "manifest_output_stem": row["output_stem"],
                }
            )
    rows.sort(key=lambda r: (r["expected_dir"], r["image_rel_path"]))
    if limit:
        rows = rows[:limit]
    return rows


def summarize(path: Path) -> dict[str, object]:
    counts = Counter()
    ok = errors = 0
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts[row.get("expected_label", "")] += 1
                ok += bool(row.get("ok"))
                errors += not bool(row.get("ok"))
    return {"total": ok + errors, "ok": ok, "errors": errors, "by_label": dict(counts)}


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    plan_path = out_dir / "sample_plan.jsonl"
    summary_path = out_dir / "summary.json"

    plan = build_plan(Path(args.manifest), args.source, args.limit)
    with plan_path.open("w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    exclude_paths = [Path(p) for p in args.exclude_results] + [results_path]
    completed = load_done(exclude_paths)
    pending = [row for row in plan if row["image_path"] not in completed]
    print(f"plan={len(plan)} completed={len(plan) - len(pending)} pending={len(pending)} concurrency={args.concurrency}", flush=True)

    lock = threading.Lock()
    started = time.monotonic()
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(worker, row, api_key, args.model, args.timeout, args.max_retries): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["source"] = "gemini_api"
            result["imported"] = False
            append_jsonl(results_path, result, lock)
            done_count += 1
            status = "ok" if result["ok"] else "err"
            if done_count % 10 == 0 or not result["ok"]:
                elapsed = time.monotonic() - started
                rate = done_count / elapsed * 60 if elapsed else 0
                print(f"{done_count}/{len(pending)} {status} rate={rate:.2f}/min last={result['image_rel_path']}", flush=True)

    summary = summarize(results_path)
    summary.update(
        {
            "model": args.model,
            "manifest": args.manifest,
            "source_filter": args.source,
            "concurrency": args.concurrency,
            "finished_at": utc_now(),
            "results": str(results_path),
            "sample_plan": str(plan_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
