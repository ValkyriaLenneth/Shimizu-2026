from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
from pathlib import Path

from run_gemini_coarse_test import CLASS_DIRS, IMAGE_EXTS
from test_gemini_concurrency import append_jsonl, load_done, utc_now, worker


def list_images(root: Path) -> dict[str, list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = {label: [] for label in CLASS_DIRS.values()}
    for dirname, label in CLASS_DIRS.items():
        for path in sorted((root / dirname).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                by_label[label].append(
                    {
                        "expected_dir": dirname,
                        "expected_label": label,
                        "image_path": str(path),
                        "image_rel_path": str(path.relative_to(root)),
                    }
                )
    return by_label


def build_plan(root: Path, exclude_paths: list[Path], per_class: int, seed: int) -> list[dict]:
    done = load_done(exclude_paths)
    rng = random.Random(seed)
    plan: list[dict] = []
    for label, rows in list_images(root).items():
        fresh = [row for row in rows if row["image_path"] not in done]
        target = min(per_class, len(fresh))
        chosen = rng.sample(fresh, target)
        for row in chosen:
            row = dict(row)
            row["requested_per_class"] = per_class
            row["selected_for_label"] = target
            plan.append(row)
        print(f"{label}: remaining={len(fresh)} selected={target}", flush=True)
    rng.shuffle(plan)
    return plan


def summarize(path: Path) -> dict:
    counts: dict[str, int] = {}
    ok = errors = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts[row.get("expected_label", "")] = counts.get(row.get("expected_label", ""), 0) + 1
                ok += bool(row.get("ok"))
                errors += not bool(row.get("ok"))
    return {"total": ok + errors, "ok": ok, "errors": errors, "by_label": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unzip")
    parser.add_argument("--out-dir", default="outputs/gemini_additional_200_each_no_overlap_3_1_pro")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--exclude-results",
        action="append",
        default=["outputs/gemini_balanced_300x4_3_1_pro/results.jsonl"],
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    plan_path = out_dir / "sample_plan.jsonl"
    summary_path = out_dir / "summary.json"

    exclude_paths = [Path(p) for p in args.exclude_results] + [results_path]
    plan = build_plan(Path(args.root), exclude_paths, args.per_class, args.seed)
    with plan_path.open("w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    completed = load_done([results_path])
    pending = [row for row in plan if row["image_path"] not in completed]
    print(f"pending={len(pending)} concurrency={args.concurrency}", flush=True)

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
                print(
                    f"{done_count}/{len(pending)} {status} rate={rate:.2f}/min last={result['image_rel_path']}",
                    flush=True,
                )

    summary = summarize(results_path)
    summary.update(
        {
            "model": args.model,
            "requested_per_class": args.per_class,
            "concurrency": args.concurrency,
            "finished_at": utc_now(),
            "results": str(results_path),
            "sample_plan": str(plan_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
