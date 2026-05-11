from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "gemini-3.1-pro-preview"
CLASS_DIRS = {
    "1_天井": "天井",
    "2_内壁": "内壁",
    "3_RC壁": "RC壁",
    "4_RC柱": "RC柱",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


PROMPT = """You are annotating Japanese building inspection photos.

Find the major visible building element regions. Use only these labels:
- 天井: ceiling / overhead slab or ceiling surface
- 内壁: interior wall / non-structural wall surface
- RC壁: reinforced concrete wall or shear wall
- RC柱: reinforced concrete column / pillar

Return JSON only, with this exact shape:
{
  "elements": [
    {
      "label": "天井|内壁|RC壁|RC柱",
      "bbox_2d": [ymin, xmin, ymax, xmax],
      "confidence": 0.0,
      "reason": "short reason"
    }
  ],
  "image_level_labels": ["天井|内壁|RC壁|RC柱"],
  "notes": "short notes if uncertain"
}

Bounding boxes must be normalized integer coordinates from 0 to 1000.
If the image mainly contains one building element, return one large region.
If multiple building elements are visible, return multiple regions.
Do not label damage level. Do not return damage boxes.
"""


def list_images(root: Path) -> dict[str, list[Path]]:
    samples: dict[str, list[Path]] = {}
    for dirname in CLASS_DIRS:
        paths = [
            p
            for p in (root / dirname).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        samples[dirname] = sorted(paths)
    return samples


def choose_samples(root: Path, n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for dirname, paths in list_images(root).items():
        if len(paths) < n:
            raise RuntimeError(f"{dirname} has only {len(paths)} images, need {n}")
        chosen = rng.sample(paths, n)
        for path in sorted(chosen):
            rows.append(
                {
                    "expected_dir": dirname,
                    "expected_label": CLASS_DIRS[dirname],
                    "image_path": str(path),
                    "image_rel_path": str(path.relative_to(root)),
                }
            )
    return rows


def load_existing(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
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


def parse_json_text(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def generate(api_key: str, image_path: Path, model: str, timeout: int) -> dict:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": PROMPT},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": image_data,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    text = body["candidates"][0]["content"]["parts"][0].get("text", "")
    parsed = parse_json_text(text)
    return {"raw": body, "parsed": parsed, "text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unzip")
    parser.add_argument("--out-dir", default="outputs/gemini_coarse_3_1_pro_50x4")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment.")

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.jsonl"
    results_path = out_dir / "results.jsonl"

    rows = choose_samples(root, args.per_class, args.seed)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    done = load_existing(results_path)
    total = len(rows)
    with results_path.open("a", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            if row["image_path"] in done:
                print(f"[{idx}/{total}] skip {row['image_rel_path']}", flush=True)
                continue
            image_path = Path(row["image_path"])
            result = {
                **row,
                "model": args.model,
                "ok": False,
                "error": None,
                "response": None,
            }
            print(f"[{idx}/{total}] {row['expected_label']} {row['image_rel_path']}", flush=True)
            for attempt in range(1, args.max_retries + 1):
                try:
                    result["response"] = generate(api_key, image_path, args.model, args.timeout)
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
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(args.sleep)

    print(f"Wrote {manifest_path}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
