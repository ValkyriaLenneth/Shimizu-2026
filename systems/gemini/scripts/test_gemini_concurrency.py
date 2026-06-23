from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from run_gemini_coarse_test import CLASS_DIRS, IMAGE_EXTS, PROMPT, parse_json_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_dataset_images(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dirname, label in CLASS_DIRS.items():
        class_root = root / dirname
        for path in sorted(class_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "expected_dir": dirname,
                        "expected_label": label,
                        "image_path": str(path),
                        "image_rel_path": str(path.relative_to(root)),
                    }
                )
    return rows


def load_done(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("ok") and row.get("image_path"):
                    done.add(row["image_path"])
    return done


def compact_generate(api_key: str, image_path: Path, model: str, timeout: int) -> dict:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": PROMPT},
                    {"inlineData": {"mimeType": mime_type, "data": image_data}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    candidate = body["candidates"][0]
    text = candidate["content"]["parts"][0].get("text", "")
    return {
        "parsed": parse_json_text(text),
        "text": text,
        "finishReason": candidate.get("finishReason"),
        "usageMetadata": body.get("usageMetadata", {}),
        "modelVersion": body.get("modelVersion"),
        "responseId": body.get("responseId"),
    }


def worker(row: dict, api_key: str, model: str, timeout: int, max_retries: int) -> dict:
    started = time.monotonic()
    result = {
        **row,
        "model": model,
        "ok": False,
        "error": None,
        "response": None,
        "started_at": utc_now(),
        "finished_at": None,
        "latency_sec": None,
        "attempts": 0,
    }
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            result["response"] = compact_generate(api_key, Path(row["image_path"]), model, timeout)
            result["ok"] = True
            break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {message[:1000]}"
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            result["error"] = repr(exc)
        time.sleep(min(30, attempt * 3))
    result["finished_at"] = utc_now()
    result["latency_sec"] = round(time.monotonic() - started, 3)
    return result


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unzip")
    parser.add_argument("--out-dir", default="outputs/gemini_concurrency_3_1_pro_2_to_10")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--min-concurrency", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--multiplier", type=int, default=2, help="Images per level = concurrency * multiplier.")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--exclude-results",
        action="append",
        default=["outputs/gemini_coarse_3_1_pro_50x4/results.jsonl"],
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "level_summary.jsonl"
    sample_path = out_dir / "sample_plan.jsonl"
    lock = threading.Lock()

    exclude_paths = [Path(p) for p in args.exclude_results] + [results_path]
    done = load_done(exclude_paths)
    candidates = [r for r in list_dataset_images(Path(args.root)) if r["image_path"] not in done]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    cursor = 0
    plan: list[dict] = []
    for concurrency in range(args.min_concurrency, args.max_concurrency + 1):
        batch_size = concurrency * args.multiplier
        batch = candidates[cursor : cursor + batch_size]
        cursor += batch_size
        if len(batch) < batch_size:
            raise RuntimeError(f"Not enough unused images for concurrency={concurrency}")
        for row in batch:
            plan.append({**row, "concurrency_level": concurrency})
    with sample_path.open("w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for concurrency in range(args.min_concurrency, args.max_concurrency + 1):
        batch = [r for r in plan if r["concurrency_level"] == concurrency]
        print(f"level={concurrency} images={len(batch)}", flush=True)
        level_started = time.monotonic()
        level_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(worker, row, api_key, args.model, args.timeout, args.max_retries): row
                for row in batch
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                result["concurrency_level"] = concurrency
                append_jsonl(results_path, result, lock)
                level_results.append(result)
                status = "ok" if result["ok"] else "err"
                print(
                    f"  {status} {result['latency_sec']}s {result['image_rel_path']}",
                    flush=True,
                )
        elapsed = time.monotonic() - level_started
        ok_count = sum(1 for r in level_results if r["ok"])
        latencies = [r["latency_sec"] for r in level_results if r["latency_sec"] is not None]
        summary = {
            "concurrency_level": concurrency,
            "images": len(batch),
            "ok": ok_count,
            "errors": len(batch) - ok_count,
            "elapsed_sec": round(elapsed, 3),
            "throughput_img_per_min": round(len(batch) / elapsed * 60, 3) if elapsed else None,
            "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max_latency_sec": max(latencies) if latencies else None,
        }
        append_jsonl(summary_path, summary, lock)
        print("summary", json.dumps(summary, ensure_ascii=False), flush=True)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {sample_path}")


if __name__ == "__main__":
    main()
