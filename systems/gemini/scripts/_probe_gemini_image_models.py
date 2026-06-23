#!/usr/bin/env python3
"""Probe candidate Gemini image generation models with a minimal prompt.

Writes a JSON report describing which model IDs return image bytes. The
purpose is to confirm what is currently reachable from this key before we
commit to one model in the production POC script.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = (
    "A realistic Japanese building interior inspection photograph showing a"
    " reinforced concrete column next to a flat interior wall, daytime"
    " natural light, no text, no captions, no watermark."
)


def imagen_predict(api_key: str, model: str, timeout: int = 120) -> dict:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:predict"
    )
    payload = {
        "instances": [{"prompt": PROMPT}],
        "parameters": {"sampleCount": 1, "aspectRatio": "4:3"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def gemini_image_generate(api_key: str, model: str, timeout: int = 120) -> dict:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def extract_imagen_bytes(body: dict) -> bytes | None:
    for prediction in body.get("predictions", []) or []:
        b64 = prediction.get("bytesBase64Encoded")
        if b64:
            return base64.b64decode(b64)
    return None


def extract_gemini_bytes(body: dict) -> bytes | None:
    for candidate in body.get("candidates", []) or []:
        for part in candidate.get("content", {}).get("parts", []) or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    return None


def probe_one(api_key: str, model: str, endpoint: str, out_dir: Path) -> dict:
    started = time.perf_counter()
    result = {
        "model": model,
        "endpoint": endpoint,
        "ok": False,
        "saved": None,
        "error": None,
        "raw_keys": None,
        "elapsed_sec": None,
    }
    try:
        if endpoint == "predict":
            body = imagen_predict(api_key, model)
            image = extract_imagen_bytes(body)
        else:
            body = gemini_image_generate(api_key, model)
            image = extract_gemini_bytes(body)
        result["raw_keys"] = sorted(body.keys()) if isinstance(body, dict) else None
        if image:
            out_path = out_dir / f"{model.replace('/', '_')}.png"
            out_path.write_bytes(image)
            result["ok"] = True
            result["saved"] = str(out_path)
        else:
            result["error"] = "no_image_bytes"
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        result["error"] = f"HTTP {exc.code}: {msg[:600]}"
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    result["elapsed_sec"] = round(time.perf_counter() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/synthetic_router_generation_poc/_smoke",
    )
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("imagen-4.0-generate-001", "predict"),
        ("imagen-4.0-fast-generate-001", "predict"),
        ("gemini-2.5-flash-image", "generateContent"),
    ]
    rows = [probe_one(api_key, model, endpoint, out_dir) for model, endpoint in targets]

    summary = {"prompt": PROMPT, "models": rows}
    (out_dir / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
