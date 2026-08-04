#!/usr/bin/env python3
"""Shared helpers for the ブレース / 柱脚 synthetic data lines (S1 / S2).

Kept separate from the older `run_gemini_*` router-era scripts because those
generate whole images from text, while this family edits real photographs
locally and composites the edit back into untouched original pixels.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

GRADE_NAMES = {0: "B", 1: "C", 2: "D"}
GRADE_IDS = {v: k for k, v in GRADE_NAMES.items()}

CATEGORY_JA = {
    "brace": "鋼製ブレース（筋かい）",
    "column_base": "柱脚（コンクリート基礎と鉄骨柱の取合い部）",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# dataset access
# --------------------------------------------------------------------------

def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO label file -> [(cls, cx, cy, w, h), ...] in normalised coords."""
    out: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            out.append((int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4])))
    return out


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(images_dir.glob(stem + ".*"))
    return hits[0] if hits else None


def load_category(paired_dir: Path, category: str) -> dict:
    """Split a category into damaged images and empty-label (negative) images."""
    cat = paired_dir / category
    img_dir, lab_dir = cat / "images", cat / "labels"
    damaged, negatives = [], []
    for lab in sorted(lab_dir.glob("*.txt")):
        img = find_image(img_dir, lab.stem)
        if img is None:
            continue
        boxes = read_boxes(lab)
        (damaged if boxes else negatives).append(
            {"stem": lab.stem, "image": img, "label": lab, "boxes": boxes}
        )
    return {"images_dir": img_dir, "labels_dir": lab_dir,
            "damaged": damaged, "negatives": negatives}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def boxes_to_pixels(boxes, size) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Normalised (cx,cy,w,h) -> (cls, (x0,y0,x1,y1)) in pixels."""
    W, H = size
    out = []
    for cls, cx, cy, bw, bh in boxes:
        out.append((cls, (
            (cx - bw / 2) * W, (cy - bh / 2) * H,
            (cx + bw / 2) * W, (cy + bh / 2) * H,
        )))
    return out


def cluster_boxes(px_boxes, size, gap_frac: float = 0.06) -> list[list[int]]:
    """Group boxes whose padded extents touch, so one edit call covers them all.

    Boxes on one photograph are usually a few damage spots on the same element.
    Editing them in one pass keeps the model's view of the scene consistent and
    costs fewer calls; editing them separately risks two incompatible repairs
    meeting at a shared boundary.
    """
    W, H = size
    gap = gap_frac * max(W, H)
    n = len(px_boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            _, (ax0, ay0, ax1, ay1) = px_boxes[i]
            _, (bx0, by0, bx1, by1) = px_boxes[j]
            if (ax0 - gap < bx1 and bx0 - gap < ax1
                    and ay0 - gap < by1 and by0 - gap < ay1):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def context_window(px_boxes, indices, size, context: float,
                   min_side: int, max_side: int) -> tuple[int, int, int, int]:
    """Square window around a box cluster, clamped to the image."""
    W, H = size
    xs0 = min(px_boxes[i][1][0] for i in indices)
    ys0 = min(px_boxes[i][1][1] for i in indices)
    xs1 = max(px_boxes[i][1][2] for i in indices)
    ys1 = max(px_boxes[i][1][3] for i in indices)
    cx, cy = (xs0 + xs1) / 2, (ys0 + ys1) / 2
    side = max(xs1 - xs0, ys1 - ys0) * context
    side = max(side, min_side)
    side = min(side, max_side, W, H)
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def build_paste_mask(crop_size, px_boxes, indices, window,
                     dilate_frac: float = 0.12, feather_frac: float = 0.06) -> np.ndarray:
    """Soft mask over the damage boxes only, in crop-local coordinates.

    This is the fix for the over-cleaning failure mode measured on 2026-08-03:
    the model sees a wide context window so it understands the scene, but only
    the damage boxes are written back. Ageing, staining, dirt and inspection
    marks outside the boxes stay at their original pixel values, so a synthetic
    negative cannot become recognisable by being unnaturally clean.
    """
    import cv2

    cw, ch = crop_size
    mask = np.zeros((ch, cw), dtype=np.float32)
    wx0, wy0 = window[0], window[1]
    for i in indices:
        x0, y0, x1, y1 = px_boxes[i][1]
        bw, bh = x1 - x0, y1 - y0
        dx, dy = bw * dilate_frac, bh * dilate_frac
        rx0 = int(round(max(x0 - dx - wx0, 0)))
        ry0 = int(round(max(y0 - dy - wy0, 0)))
        rx1 = int(round(min(x1 + dx - wx0, cw)))
        ry1 = int(round(min(y1 + dy - wy0, ch)))
        if rx1 > rx0 and ry1 > ry0:
            mask[ry0:ry1, rx0:rx1] = 1.0

    if mask.max() == 0:
        return mask
    k = max(3, int(round(min(cw, ch) * feather_frac)) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def composite(base: Image.Image, edited: Image.Image, mask: np.ndarray) -> Image.Image:
    """Alpha-blend `edited` into `base` under `mask` (both crop-sized)."""
    b = np.asarray(base.convert("RGB"), dtype=np.float32)
    e = np.asarray(edited.convert("RGB").resize(base.size, Image.LANCZOS), dtype=np.float32)
    m = mask[..., None]
    return Image.fromarray(np.clip(b * (1 - m) + e * m, 0, 255).astype(np.uint8))


def match_texture(result: Image.Image, reference: Image.Image, mask: np.ndarray,
                  strength: float = 1.0, max_gain: float = 2.0) -> Image.Image:
    """Restore high-frequency energy inside the mask to match its surroundings.

    Generated pixels come back smoother than the surrounding photograph, and the
    2026-08-03 QC pass found that a vision judge names this directly ("blurring,
    smoothing, loss of natural concrete texture where the crack was removed").
    Left alone it is a low-level cue separating edited from unedited material -
    exactly the kind of shortcut this corpus already suffers from.

    Two stages, in this order:

    1. **Amplify the structure that is already there.** Micro-contrast the model
       did produce is real detail; scaling it up is more plausible than
       manufacturing texture, so the deficit is closed this way first.
    2. **Top up the remainder with grain.** Only whatever amplification could
       not reach, capped by `max_gain`, is filled with noise matched to the
       reference ring's standard deviation.
    """
    import cv2

    if mask.max() <= 0 or strength <= 0:
        return result

    ref = np.asarray(reference.convert("RGB"), dtype=np.float32)
    res = np.asarray(result.convert("RGB"), dtype=np.float32)

    solid = (mask > 0.15).astype(np.uint8)
    ring = (cv2.dilate(solid, np.ones((41, 41), np.uint8)) - solid) > 0
    core = mask > 0.5
    if ring.sum() < 256 or core.sum() < 64:
        return result

    def hf(a):
        return a - cv2.GaussianBlur(a, (0, 0), 1.2)

    ref_hf, res_hf = hf(ref), hf(res)
    target = float(ref_hf[ring].std())
    have = float(res_hf[core].std())
    if have <= 1e-4 or target <= have:
        return result

    gain = min(target / have, max_gain)
    out = res + res_hf * (gain - 1.0) * mask[..., None] * strength

    reached = have * gain
    if reached < target:
        rng = np.random.default_rng(20260803)
        extra = np.sqrt(max(target ** 2 - reached ** 2, 0.0)) * strength
        noise = rng.normal(0.0, extra, size=res.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
        out = out + noise * mask[..., None]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def match_photometry(result: Image.Image, reference: Image.Image, mask: np.ndarray,
                     strength: float = 1.0, max_shift: float = 40.0) -> Image.Image:
    """Pull the edited region's tone back to the original region's tone.

    The model tends to lighten shadowed concrete: four of 22 images in the
    2026-08-03 batch came back 29-57 levels brighter inside the mask than the
    ring around it. That is a low-level cue distinguishing repaired regions, and
    it is correctable without another model call.

    The statistic is a per-channel MEDIAN of the original region rather than a
    mean, because the region still contains the crack being removed and a mean
    would drag the correction dark by exactly the thing that was repaired.
    """
    import cv2

    if mask.max() <= 0 or strength <= 0:
        return result
    core = mask > 0.5
    if core.sum() < 64:
        return result

    ref = np.asarray(reference.convert("RGB"), dtype=np.float32)
    res = np.asarray(result.convert("RGB"), dtype=np.float32)

    out = res.copy()
    for c in range(3):
        want = float(np.median(ref[..., c][core]))
        have = float(np.median(res[..., c][core]))
        shift = np.clip((want - have) * strength, -max_shift, max_shift)
        out[..., c] += shift * mask
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def jpeg_roundtrip(img: Image.Image, quality: int = 95, subsampling: int = 0) -> Image.Image:
    """Re-encode through JPEG so comparisons isolate real edits from codec noise."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------
# Gemini transport
# --------------------------------------------------------------------------

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def encode_image(img: Image.Image, max_side: int = 1024, quality: int = 95) -> dict:
    im = img.convert("RGB")
    if max_side and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return {"inline_data": {"mime_type": "image/jpeg",
                            "data": base64.b64encode(buf.getvalue()).decode()}}


def generate_image(model: str, parts: list[dict], api_key: str,
                   timeout: int = 240, max_retries: int = 3,
                   image_size: str | None = None, aspect_ratio: str | None = None,
                   ) -> tuple[Image.Image | None, str | None]:
    """POST parts to a Gemini image model, return the first image returned.

    `image_size` is worth setting above the default. gemini-3-pro-image returns
    1196x896 unless told otherwise, and "2K" returns 2392x1792. Generating above
    the paste-back window and downscaling supersamples the edit, which is the
    direct answer to the QC judge's repeated finding that repaired regions "lack
    the natural grain and texture" of the surrounding photograph - upscaling a
    1K generation into a 1280px window cannot produce texture that was never
    generated.
    """
    gen_cfg: dict = {"responseModalities": ["IMAGE", "TEXT"]}
    img_cfg: dict = {}
    if image_size:
        img_cfg["imageSize"] = image_size
    if aspect_ratio:
        img_cfg["aspectRatio"] = aspect_ratio
    if img_cfg:
        gen_cfg["imageConfig"] = img_cfg
    payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/{model}:generateContent",
                data=body,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            texts = []
            for cand in resp.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and inline.get("data"):
                        img = Image.open(io.BytesIO(base64.b64decode(inline["data"])))
                        return img.convert("RGB"), None
                    if part.get("text"):
                        texts.append(part["text"])
            fb = resp.get("promptFeedback", {})
            last_err = f"no image; text={' | '.join(texts)[:200]}; feedback={json.dumps(fb)[:200]}"
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (400, 403, 404):
                break
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(min(2 ** attempt * 1.5, 12))
    return None, last_err


def generate_text(model: str, parts: list[dict], api_key: str,
                  timeout: int = 180, max_retries: int = 3,
                  response_schema: dict | None = None) -> tuple[dict | str | None, str | None]:
    """POST parts to a Gemini text model; optionally force a JSON schema."""
    gen_cfg: dict = {}
    if response_schema is not None:
        gen_cfg = {"responseMimeType": "application/json",
                   "responseSchema": response_schema}
    payload = {"contents": [{"parts": parts}]}
    if gen_cfg:
        payload["generationConfig"] = gen_cfg
    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/{model}:generateContent",
                data=body,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            chunks = []
            for cand in resp.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text"):
                        chunks.append(part["text"])
            text = "".join(chunks).strip()
            if not text:
                last_err = f"empty response: {json.dumps(resp)[:200]}"
            elif response_schema is not None:
                try:
                    return json.loads(text), None
                except json.JSONDecodeError as exc:
                    last_err = f"bad json ({exc}): {text[:200]}"
            else:
                return text, None
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (400, 403, 404):
                break
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(min(2 ** attempt * 1.5, 12))
    return None, last_err


# --------------------------------------------------------------------------
# damage inventory (pre-screen)
# --------------------------------------------------------------------------

DAMAGE_INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"},
                               "minItems": 4, "maxItems": 4},
                    "kind": {"type": "string"},
                    "on_target_element": {"type": "boolean"},
                    "severity": {"type": "string", "enum": ["minor", "moderate", "severe"]},
                },
                "required": ["box_2d", "kind", "on_target_element", "severity"],
            },
        },
        "damage_is_pervasive": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["regions", "damage_is_pervasive", "notes"],
}

DAMAGE_INVENTORY_PROMPT = """This is a Japanese building damage survey photograph.
The element under inspection is {element_ja}.

List EVERY region showing structural damage anywhere in this photograph - on the
inspected element and on any other structure, pipe, bolt, plate or fixture visible.

Structural damage means: cracking, spalling, delamination, exposed or corroded
reinforcement, corroded or loose anchor bolts, corrosion with scaling or section loss,
buckling, deformation, fracture, or missing material.

The following are NOT damage; do not list them: rust staining without material loss,
water stains, efflorescence, dirt, grime, mould, faded or chalky paint, discolouration,
construction and survey markings, chalk lines, handwriting, form-tie holes, joint lines,
chamfers, casting seams, and ordinary surface irregularity.

For each region give box_2d as [ymin, xmin, ymax, xmax] normalised to 0-1000, the kind of
damage, whether it sits on the inspected element, and its severity.

Set damage_is_pervasive to true if damage covers so much of the frame that removing all
of it would amount to redrawing the photograph."""


def inventory_damage(model: str, image: Image.Image, category: str, api_key: str,
                     send_size: int = 1024, timeout: int = 180,
                     max_retries: int = 2) -> tuple[dict | None, str | None]:
    """Ask a vision model for every damage region, annotated or not.

    The 2026-08-03 QC pass showed the delivered labels cover only part of what a
    strict reader calls damage: images whose annotated crack was repaired were
    still rejected for corroded anchor bolts, scaling rust on the base plate, and
    corroded adjacent pipework, none of which carry a box. Repairing only the
    annotated boxes therefore cannot produce a clean negative.
    """
    parts = [
        encode_image(image, max_side=send_size),
        {"text": DAMAGE_INVENTORY_PROMPT.format(element_ja=CATEGORY_JA[category])},
    ]
    return generate_text(model, parts, api_key, timeout=timeout,
                         max_retries=max_retries,
                         response_schema=DAMAGE_INVENTORY_SCHEMA)


def inventory_to_boxes(inventory: dict, size: tuple[int, int],
                       min_area_frac: float = 1e-5,
                       max_area_frac: float = 0.60) -> list[tuple[int, float, float, float, float]]:
    """Convert 0-1000 [ymin,xmin,ymax,xmax] regions to normalised YOLO tuples."""
    out = []
    for r in inventory.get("regions", []):
        b = r.get("box_2d") or []
        if len(b) != 4:
            continue
        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) / 1000.0 for v in b]
        if xmax <= xmin or ymax <= ymin:
            continue
        w, h = xmax - xmin, ymax - ymin
        if not (min_area_frac <= w * h <= max_area_frac):
            continue
        out.append((0, (xmin + xmax) / 2, (ymin + ymax) / 2, w, h))
    return out


def require_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (see ~/.zshenv).")
    return key


# --------------------------------------------------------------------------
# jsonl resume
# --------------------------------------------------------------------------

_jsonl_lock = threading.Lock()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _jsonl_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done(path: Path, key: str = "job_id") -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ok") and row.get(key):
            done.add(row[key])
    return done
