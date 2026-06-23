from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
from pathlib import Path

from test_gemini_concurrency import append_jsonl, load_done, utc_now, worker
from run_gemini_coarse_test import CLASS_DIRS, IMAGE_EXTS


def list_dataset_images(root: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {label: [] for label in CLASS_DIRS.values()}
    for dirname, label in CLASS_DIRS.items():
        for path in sorted((root / dirname).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows[label].append(
                    {
                        "expected_dir": dirname,
                        "expected_label": label,
                        "image_path": str(path),
                        "image_rel_path": str(path.relative_to(root)),
                    }
                )
    return rows


def compact_existing(row: dict, source_path: str) -> dict:
    response = row.get("response") or {}
    if "parsed" in response:
        compact_response = {
            "parsed": response.get("parsed"),
            "text": response.get("text"),
            "finishReason": response.get("finishReason"),
            "usageMetadata": response.get("usageMetadata"),
            "modelVersion": response.get("modelVersion"),
            "responseId": response.get("responseId"),
        }
    else:
        raw = response.get("raw") or {}
        candidate = (raw.get("candidates") or [{}])[0]
        usage = raw.get("usageMetadata")
        compact_response = {
            "parsed": response.get("parsed"),
            "text": response.get("text"),
            "finishReason": candidate.get("finishReason"),
            "usageMetadata": usage,
            "modelVersion": raw.get("modelVersion"),
            "responseId": raw.get("responseId"),
        }
    return {
        "expected_dir": row.get("expected_dir"),
        "expected_label": row.get("expected_label"),
        "image_path": row.get("image_path"),
        "image_rel_path": row.get("image_rel_path"),
        "model": row.get("model"),
        "ok": bool(row.get("ok")),
        "error": row.get("error"),
        "response": compact_response,
        "source": source_path,
        "imported": True,
        "started_at": None,
        "finished_at": None,
        "latency_sec": None,
        "attempts": row.get("attempts"),
    }


def load_existing_rows(paths: list[Path]) -> dict[str, dict]:
    by_image: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("ok") and row.get("image_path") and row["image_path"] not in by_image:
                    by_image[row["image_path"]] = compact_existing(row, str(path))
    return by_image


def build_plan(
    root: Path,
    existing: dict[str, dict],
    per_class: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    by_label = list_dataset_images(root)
    plan: list[dict] = []
    for label, rows in by_label.items():
        existing_rows = [existing[r["image_path"]] for r in rows if r["image_path"] in existing]
        existing_rows = existing_rows[:per_class]
        need = per_class - len(existing_rows)
        fresh_candidates = [r for r in rows if r["image_path"] not in existing]
        if len(fresh_candidates) < need:
            raise RuntimeError(f"{label}: need {need} fresh images, only {len(fresh_candidates)} available")
        chosen_fresh = rng.sample(fresh_candidates, need)
        class_plan = []
        for row in existing_rows:
            class_plan.append({**row, "plan_source": "existing"})
        for row in chosen_fresh:
            class_plan.append({**row, "plan_source": "new"})
        rng.shuffle(class_plan)
        plan.extend(class_plan)
    return plan


def summarize(path: Path) -> dict:
    counts: dict[str, int] = {}
    ok = errors = imported = 0
    if not path.exists():
        return {"total": 0, "ok": 0, "errors": 0, "imported": 0, "by_label": counts}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("expected_label", "")
            counts[label] = counts.get(label, 0) + 1
            ok += bool(row.get("ok"))
            errors += not bool(row.get("ok"))
            imported += bool(row.get("imported"))
    return {"total": ok + errors, "ok": ok, "errors": errors, "imported": imported, "by_label": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unzip")
    parser.add_argument("--out-dir", default="outputs/gemini_balanced_300x4_3_1_pro")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--reuse-results",
        action="append",
        default=[
            "outputs/gemini_coarse_3_1_pro_50x4/results.jsonl",
            "outputs/gemini_concurrency_3_1_pro_2_to_10/results.jsonl",
        ],
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

    existing = load_existing_rows([Path(p) for p in args.reuse_results])
    plan = build_plan(Path(args.root), existing, args.per_class, args.seed)
    with plan_path.open("w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    completed = load_done([results_path])
    lock = threading.Lock()

    import_count = 0
    for row in plan:
        if row["image_path"] in completed:
            continue
        if row.get("plan_source") == "existing":
            append_jsonl(results_path, row, lock)
            completed.add(row["image_path"])
            import_count += 1
    print(f"imported_existing={import_count}", flush=True)

    pending = [row for row in plan if row["image_path"] not in completed]
    print(f"pending_new={len(pending)} concurrency={args.concurrency}", flush=True)
    started = time.monotonic()
    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(worker, row, api_key, args.model, args.timeout, args.max_retries): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["imported"] = False
            result["source"] = "gemini_api"
            result["plan_source"] = "new"
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
            "per_class": args.per_class,
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
