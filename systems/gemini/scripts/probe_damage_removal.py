#!/usr/bin/env python3
"""Feasibility probe for S1: can Gemini repair a damage region to sound condition?

The S1 line ("counterfactual negatives") only works if the edit is *local*: the
damage disappears and everything else - framing, lighting, material, adjacent
structure - stays put. A model that re-imagines the whole scene produces a
negative that no longer pairs with its positive, which is the entire point.

So this probe works on a padded crop around one real damage box and composites
the result back into the untouched original. Outside the crop the pixels are the
original photograph, byte for byte. Inside, only the repair.

Writes side-by-side comparisons for visual reading. No dataset is produced here.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GRADE_NAMES = {0: "B", 1: "C", 2: "D"}

ELEMENT_JA = {"brace": "鋼製ブレース（筋かい）", "column_base": "柱脚（コンクリート基礎と鉄骨柱の取合い部）"}

REPAIR_PROMPT = """This is a close-up region from a Japanese building damage survey photograph.
The element shown is {element}.

Task: repair the visible damage so the element appears in SOUND, UNDAMAGED condition.

Remove: cracks, spalling, exposed rebar, corrosion, rust staining, paint peeling,
material loss, and debris that belong to the damage.

Preserve exactly, with no change whatsoever:
- the camera viewpoint, framing, and every object boundary
- the lighting, shadows, colour temperature, and exposure
- the material identity and surface texture of the sound parts (concrete finish,
  paint colour, steel surface)
- all surrounding context: adjacent walls, floor, pipes, bolts, fixtures, dirt,
  ordinary staining and construction marks that are NOT damage
- the image sharpness, noise level, and any motion blur or compression character

Do not beautify, do not clean the scene, do not re-light, do not change the angle,
do not add or remove any object other than the damage itself. The result must look
like the same photograph taken of the same element before it was damaged, with the
same camera and the same imperfect field conditions.

Output the edited image at the same aspect ratio."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--paired-dir",
        default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724",
    )
    p.add_argument("--out-dir", default="outputs/gemini_synth/probe_damage_removal")
    p.add_argument("--model", default="gemini-3-pro-image")
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--grades", default="B,D", help="comma separated among B,C,D")
    p.add_argument("--per-grade", type=int, default=3)
    p.add_argument("--context", type=float, default=2.2, help="crop side = box side * this")
    p.add_argument("--min-crop", type=int, default=512)
    p.add_argument("--max-send", type=int, default=1024, help="downscale crop before sending")
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--timeout", type=int, default=240)
    return p.parse_args()


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            out.append((int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4])))
    return out


def crop_window(im: Image.Image, box, context: float, min_side: int) -> tuple[int, int, int, int]:
    W, H = im.size
    cx, cy, bw, bh = box
    side = max(bw * W, bh * H) * context
    side = max(side, min_side)
    side = min(side, min(W, H))
    left = min(max(cx * W - side / 2, 0), W - side)
    top = min(max(cy * H - side / 2, 0), H - side)
    return int(left), int(top), int(left + side), int(top + side)


def call_gemini(model: str, key: str, image: Image.Image, prompt: str, timeout: int):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": base64.b64encode(buf.getvalue()).decode()}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    req = urllib.request.Request(
        f"{BASE}/{model}:generateContent",
        data=json.dumps(payload).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    text_out = []
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return Image.open(io.BytesIO(base64.b64decode(inline["data"]))).convert("RGB"), None
            if part.get("text"):
                text_out.append(part["text"])
    return None, " | ".join(text_out) or json.dumps(resp)[:400]


def main() -> int:
    args = parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Set GEMINI_API_KEY")
        return 1

    rng = random.Random(args.seed)
    paired = Path(args.paired_dir) / args.category
    out = Path(args.out_dir) / args.category
    out.mkdir(parents=True, exist_ok=True)

    wanted = [g.strip() for g in args.grades.split(",") if g.strip()]
    pool: dict[str, list] = {g: [] for g in wanted}
    for lab in sorted((paired / "labels").glob("*.txt")):
        imgs = list((paired / "images").glob(lab.stem + ".*"))
        if not imgs:
            continue
        for idx, (cls, cx, cy, bw, bh) in enumerate(read_boxes(lab)):
            g = GRADE_NAMES.get(cls)
            if g in pool and 0.005 < bw * bh < 0.30:
                pool[g].append((imgs[0], idx, (cx, cy, bw, bh)))

    records = []
    for grade in wanted:
        rng.shuffle(pool[grade])
        for img_path, box_idx, box in pool[grade][: args.per_grade]:
            tag = f"{grade}_{img_path.stem}_{box_idx}"
            try:
                with Image.open(img_path) as src:
                    src = src.convert("RGB")
                    win = crop_window(src, box, args.context, args.min_crop)
                    crop = src.crop(win)
                    send = crop.copy()
                    if max(send.size) > args.max_send:
                        send.thumbnail((args.max_send, args.max_send), Image.LANCZOS)

                    prompt = REPAIR_PROMPT.format(element=ELEMENT_JA[args.category])
                    edited, err = call_gemini(args.model, key, send, prompt, args.timeout)
                    if edited is None:
                        print(f"  ! {tag}: no image returned -> {err[:150]}")
                        records.append({"tag": tag, "ok": False, "error": err[:400]})
                        continue

                    edited_full = edited.resize(crop.size, Image.LANCZOS)
                    composited = src.copy()
                    composited.paste(edited_full, (win[0], win[1]))

                    # side-by-side: original crop (box marked) | repaired crop
                    cw, ch = crop.size
                    sheet = Image.new("RGB", (cw * 2 + 12, ch), (20, 20, 20))
                    marked = crop.copy()
                    d = ImageDraw.Draw(marked)
                    W, H = src.size
                    cx, cy, bw, bh = box
                    d.rectangle(
                        [cx * W - bw * W / 2 - win[0], cy * H - bh * H / 2 - win[1],
                         cx * W + bw * W / 2 - win[0], cy * H + bh * H / 2 - win[1]],
                        outline=(0, 255, 120), width=max(2, cw // 200),
                    )
                    sheet.paste(marked, (0, 0))
                    sheet.paste(edited_full, (cw + 12, 0))
                    sheet.thumbnail((1600, 1600), Image.LANCZOS)
                    sheet.save(out / f"{tag}_compare.jpg", quality=92)
                    composited.save(out / f"{tag}_composited.jpg", quality=94)
                    print(f"  ok {tag}  crop={crop.size} win={win}")
                    records.append({"tag": tag, "ok": True, "source": img_path.name,
                                    "grade": grade, "box": list(box), "window": list(win)})
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                print(f"  ! {tag}: HTTP {e.code} {body}")
                records.append({"tag": tag, "ok": False, "error": f"HTTP {e.code} {body}"})
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {tag}: {exc}")
                records.append({"tag": tag, "ok": False, "error": str(exc)[:300]})

    (out / "probe_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    ok = sum(1 for r in records if r.get("ok"))
    print(f"\n{ok}/{len(records)} succeeded -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
