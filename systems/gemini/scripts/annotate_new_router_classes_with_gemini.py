#!/usr/bin/env python3
"""Annotate brace/column-base router images with Gemini.

The downloaded 2026-06-30 data has directory-level labels but no boxes. This
script asks Gemini for coarse building-element boxes and writes the same JSONL
shape as the existing Gemini annotation scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

from run_gemini_coarse_test import IMAGE_EXTS, parse_json_text  # noqa: E402
from test_gemini_concurrency import append_jsonl, load_done, utc_now  # noqa: E402


CLASS_DIRS = {
    "ブレース": "ブレース",
    "柱脚": "柱脚",
}

PROMPT = """You are annotating Japanese building inspection photos.

Find the major visible building element regions. Use only these labels:
- 天井: ceiling / overhead slab or ceiling surface
- 内壁: interior wall / non-structural wall surface
- RC壁: reinforced concrete wall or shear wall
- RC柱: reinforced concrete column / pillar
- ブレース: steel brace / diagonal bracing member
- 柱脚: column base / base plate / anchor-bolt region at the foot of a column

Return JSON only, with this exact shape:
{
  "elements": [
    {
      "label": "天井|内壁|RC壁|RC柱|ブレース|柱脚",
      "bbox_2d": [ymin, xmin, ymax, xmax],
      "confidence": 0.0,
      "reason": "short reason"
    }
  ],
  "image_level_labels": ["天井|内壁|RC壁|RC柱|ブレース|柱脚"],
  "notes": "short notes if uncertain"
}

Bounding boxes must be normalized integer coordinates from 0 to 1000.
If the image mainly contains one building element, return one large region.
If multiple building elements are visible, return multiple regions.
Do not label damage level. Do not return damage boxes.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw_new_classes_20260630/extracted")
    parser.add_argument("--out-dir", default="outputs/gemini_new_router_classes_20260630")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--api-mode", choices=["interactions", "generate-content"], default="interactions")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--focus-expected",
        action="store_true",
        help="append the directory-level expected class as the primary search target",
    )
    return parser.parse_args()


def list_images(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = next((class_label for dirname, class_label in CLASS_DIRS.items() if dirname in path.parts), None)
        if not label:
            continue
        rows.append(
            {
                "expected_dir": label,
                "expected_label": label,
                "image_path": str(path),
                "image_rel_path": str(path.relative_to(root)),
            }
        )
    return rows


def extract_interaction_text(body: dict) -> str:
    for step in body.get("steps", []) or []:
        if step.get("type") != "model_output":
            continue
        for part in step.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                return str(part["text"])
    return ""


def generate_with_interactions(api_key: str, image_path: Path, model: str, timeout: int, prompt: str) -> dict:
    import base64
    import mimetypes
    import urllib.request

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": mime_type, "data": image_data},
        ],
        "response_format": {"type": "text", "mime_type": "application/json"},
    }
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    text = extract_interaction_text(body)
    return {
        "parsed": parse_json_text(text),
        "text": text,
        "finishReason": body.get("status"),
        "usageMetadata": body.get("usage", {}),
        "modelVersion": body.get("model"),
        "responseId": body.get("id"),
        "raw_keys": sorted(body.keys()),
        "api_mode": "interactions",
    }


def generate_with_generate_content(api_key: str, image_path: Path, model: str, timeout: int, prompt: str) -> dict:
    import base64
    import mimetypes
    import urllib.request

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
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
        "api_mode": "generate-content",
    }


def generate_with_prompt(
    api_key: str, image_path: Path, model: str, timeout: int, api_mode: str, prompt: str
) -> dict:
    if api_mode == "interactions":
        return generate_with_interactions(api_key, image_path, model, timeout, prompt)
    return generate_with_generate_content(api_key, image_path, model, timeout, prompt)


def worker(
    row: dict, api_key: str, model: str, timeout: int, max_retries: int, api_mode: str,
    focus_expected: bool,
) -> dict:
    import urllib.error

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
        "api_mode": api_mode,
    }
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            prompt = PROMPT
            if focus_expected:
                prompt += (
                    "\nPrimary task for this image: carefully locate every visible region of "
                    f"the expected directory class '{row['expected_label']}'. Thin, partial, "
                    "occluded, roof-plane, and background instances still count. Do not relabel "
                    "a brace as ceiling merely because it is part of a roof truss. Do not relabel "
                    "a column-base/pedestal region as only RC柱. Return no expected-class box only "
                    "when that element is genuinely not visible.\n"
                )
            result["response"] = generate_with_prompt(
                api_key, Path(row["image_path"]), model, timeout, api_mode, prompt
            )
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


def summarize(results_path: Path) -> dict:
    counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    if not results_path.exists():
        return {"total": 0, "ok": 0, "errors": 0, "detected_label_counts": {}, "expected_label_counts": {}}
    latest_by_image: dict[str, dict] = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = row.get("image_path")
            if image_path:
                latest_by_image[str(image_path)] = row
    ok = errors = 0
    for row in latest_by_image.values():
            expected_counts[row.get("expected_label", "")] += 1
            ok += bool(row.get("ok"))
            errors += not bool(row.get("ok"))
            parsed = ((row.get("response") or {}).get("parsed") or {}) if row.get("ok") else {}
            for element in parsed.get("elements", []) or []:
                if isinstance(element, dict) and element.get("label"):
                    counts[str(element["label"])] += 1
    return {
        "total": ok + errors,
        "ok": ok,
        "errors": errors,
        "detected_label_counts": dict(counts),
        "expected_label_counts": dict(expected_counts),
    }


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in the environment.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    plan_path = out_dir / "sample_plan.jsonl"
    summary_path = out_dir / "summary.json"

    rows = list_images(Path(args.root))
    if args.limit:
        rows = rows[: args.limit]
    completed = load_done([results_path])
    plan = [row for row in rows if row["image_path"] not in completed]
    with plan_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"images={len(rows)} pending={len(plan)} concurrency={args.concurrency}", flush=True)
    lock = threading.Lock()
    started = time.monotonic()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                worker, row, api_key, args.model, args.timeout, args.max_retries, args.api_mode,
                args.focus_expected,
            ): row
            for row in plan
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["source"] = "gemini_api_new_router_classes"
            result["imported"] = False
            append_jsonl(results_path, result, lock)
            done += 1
            if done % 10 == 0 or not result.get("ok"):
                elapsed = time.monotonic() - started
                rate = done / elapsed * 60 if elapsed else 0
                status = "ok" if result.get("ok") else "err"
                print(f"{done}/{len(plan)} {status} rate={rate:.2f}/min last={result['image_rel_path']}", flush=True)

    summary = summarize(results_path)
    summary.update(
        {
            "model": args.model,
            "api_mode": args.api_mode,
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
