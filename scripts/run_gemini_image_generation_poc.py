#!/usr/bin/env python3
"""Small-batch synthetic image generation POC for the three-class router.

Generates plausible Japanese building inspection photographs covering the
router classes 天井 / 内壁 / RC壁 / RC柱, with extra "hard case" prompts that
target the known RC柱 vs 壁类 confusion. The intent is not to flood training
data, but to validate end-to-end feasibility before deciding whether to scale
up.

Outputs land in `outputs/synthetic_router_generation_poc/` by default and are
organised so the existing Gemini coarse annotation + visualization pipeline
can be reused directly on the generated images.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLASS_DIR_NAMES: dict[str, str] = {
    "天井": "tenjo",
    "内壁": "inner_wall",
    "RC壁": "rc_wall",
    "RC柱": "rc_column",
}

EXPECTED_DIRS: dict[str, str] = {
    "天井": "1_天井",
    "内壁": "2_内壁",
    "RC壁": "3_RC壁",
    "RC柱": "4_RC柱",
}

NEGATIVE_TAIL = (
    " Composition: a quiet, completely unoccupied indoor or outdoor"
    " architectural scene. Style: an ordinary Japanese building damage"
    " survey photo, similar to a field-inspection dataset rather than a"
    " polished architectural portfolio image. Use a handheld smartphone"
    " or compact-camera look, 4:3 framing, close crop, imperfect verticals,"
    " mild lens distortion, uneven indoor exposure, slight blur or JPEG"
    " compression, and utilitarian lighting. The target building element"
    " should fill most of the frame, with only small context from adjacent"
    " walls, ceiling, floor, beams, corners, or openings. Avoid dramatic"
    " wide-angle interiors, luxury finishes, staged symmetry, glossy"
    " rendering, CGI, illustration, floor-plan style views, or clean"
    " magazine photography. The frame must contain only the architecture"
    " itself (concrete, paint, fixtures, etc.) without any human figures,"
    " hands, gloves, inspection tools, measuring tapes, rulers,"
    " markers, ladders, helmets, traffic cones, vehicles, furniture"
    " moving boxes, text overlays, captions, watermarks, diagrams,"
    " annotations, logos, or signage. Make inspection-relevant damage"
    " clearly visible on the target element: hairline cracks, diagonal"
    " cracks, small spalling, staining, paint peeling, rust bleeding,"
    " efflorescence, or surface wear. The damage should look realistic"
    " and physically plausible, not decorative, and it must stay on the"
    " building surface rather than becoming text or drawn markings."
)


PROMPTS: list[dict[str, Any]] = [
    # ---------------- 天井 ----------------
    {
        "id": "tenjo_concrete_slab_01",
        "primary_class": "天井",
        "target_classes": ["天井"],
        "scenario": "basic_concrete_slab",
        "prompt": (
            "An interior inspection photograph of an exposed reinforced"
            " concrete ceiling slab in a Japanese public building. The"
            " viewer is standing on the floor and looking up, so the ceiling"
            " fills most of the frame. Visible formwork marks, slight"
            " surface staining and small hairline cracks across the slab."
        ),
    },
    {
        "id": "tenjo_painted_ceiling_02",
        "primary_class": "天井",
        "target_classes": ["天井"],
        "scenario": "painted_ceiling_with_fixtures",
        "prompt": (
            "An interior inspection photograph of a painted flat ceiling in"
            " a Japanese office corridor. The lens is angled slightly upward"
            " so the ceiling dominates the frame, with two recessed light"
            " fixtures and one HVAC vent visible. Some yellow staining and"
            " a faint diagonal crack near the corner."
        ),
    },
    {
        "id": "tenjo_underground_parking_03",
        "primary_class": "天井",
        "target_classes": ["天井"],
        "scenario": "underground_parking_ceiling",
        "prompt": (
            "An upward inspection photograph in an underground parking"
            " structure showing the reinforced concrete ceiling. Exposed"
            " beams and ducts run across the slab. The ceiling has minor"
            " water staining and a crack pattern near the beam interface."
        ),
    },
    # ---------------- 内壁 ----------------
    {
        "id": "inner_wall_painted_corridor_01",
        "primary_class": "内壁",
        "target_classes": ["内壁"],
        "scenario": "painted_corridor_wall",
        "prompt": (
            "Inspection photograph of a painted interior corridor wall in a"
            " Japanese office building. The wall fills the entire frame and"
            " is not obviously structural, with a flat painted finish and"
            " mild scuff marks. Slight diagonal hairline cracks near the"
            " upper edge."
        ),
    },
    {
        "id": "inner_wall_wallpaper_02",
        "primary_class": "内壁",
        "target_classes": ["内壁"],
        "scenario": "wallpaper_inner_wall",
        "prompt": (
            "Inspection photograph of a wallpapered interior partition wall"
            " in a Japanese apartment hallway. The wall is shot head-on,"
            " filling the frame. Visible seams between wallpaper panels and"
            " a small crack pattern radiating from a corner."
        ),
    },
    {
        "id": "inner_wall_paint_peel_03",
        "primary_class": "内壁",
        "target_classes": ["内壁"],
        "scenario": "paint_peel_inner_wall",
        "prompt": (
            "Close inspection photograph of a non-structural interior wall"
            " with cream paint that is starting to peel and discolour."
            " The wall takes up the whole frame, with a baseboard at the"
            " bottom and a faint crack travelling vertically through the"
            " peeling area."
        ),
    },
    # ---------------- RC壁 ----------------
    {
        "id": "rc_wall_exposed_concrete_01",
        "primary_class": "RC壁",
        "target_classes": ["RC壁"],
        "scenario": "exposed_concrete_shear_wall",
        "prompt": (
            "Inspection photograph of an exposed reinforced concrete shear"
            " wall in a Japanese building stairwell. The flat concrete"
            " surface dominates the frame, with form-tie holes visible in a"
            " regular grid. Several thin diagonal cracks cross the wall."
        ),
    },
    {
        "id": "rc_wall_outdoor_facade_02",
        "primary_class": "RC壁",
        "target_classes": ["RC壁"],
        "scenario": "outdoor_rc_wall_facade",
        "prompt": (
            "Outdoor inspection photograph of a reinforced concrete external"
            " wall on a Japanese mid-rise building. The wall fills the frame"
            " and shows weathering streaks, efflorescence, and a long"
            " diagonal crack running across the middle of the panel."
        ),
    },
    {
        "id": "rc_wall_basement_rust_03",
        "primary_class": "RC壁",
        "target_classes": ["RC壁"],
        "scenario": "basement_rc_wall_with_rust",
        "prompt": (
            "Inspection photograph of a reinforced concrete basement wall in"
            " a Japanese facility. The wall surface shows rust staining"
            " bleeding from the reinforcement, plus a network of hairline"
            " cracks and one wider horizontal crack near the floor slab."
        ),
    },
    # ---------------- RC柱 ----------------
    {
        "id": "rc_column_standalone_indoor_01",
        "primary_class": "RC柱",
        "target_classes": ["RC柱"],
        "scenario": "standalone_indoor_column",
        "prompt": (
            "Inspection photograph of a free-standing rectangular reinforced"
            " concrete column inside a Japanese public hall. The column is"
            " photographed roughly head-on, occupies the centre of the"
            " frame, and shows formwork seams and minor vertical cracks"
            " along one face."
        ),
    },
    {
        "id": "rc_column_parking_garage_02",
        "primary_class": "RC柱",
        "target_classes": ["RC柱"],
        "scenario": "parking_garage_column",
        "prompt": (
            "Inspection photograph of a reinforced concrete column in a"
            " parking garage in Japan. The column is roughly centred in the"
            " frame, with concrete floor and ceiling slabs visible above"
            " and below. The column has impact scuffs and small diagonal"
            " cracks near the base."
        ),
    },
    {
        "id": "rc_column_outdoor_pilotis_03",
        "primary_class": "RC柱",
        "target_classes": ["RC柱"],
        "scenario": "outdoor_pilotis_column",
        "prompt": (
            "Inspection photograph of a reinforced concrete pilotis column"
            " under a Japanese mid-rise building. The column rises through"
            " the frame from floor to soffit, with the ground level"
            " surrounding it. There is a faint horizontal crack near"
            " mid-height."
        ),
    },
    # ---------------- Hard cases: RC柱 next to 壁类 ----------------
    {
        "id": "hard_rc_column_against_inner_wall_01",
        "primary_class": "RC柱",
        "target_classes": ["RC柱", "内壁"],
        "scenario": "hard_column_against_inner_wall",
        "prompt": (
            "Close field-inspection photograph of a Japanese building"
            " interior where a rectangular reinforced concrete column meets"
            " a painted non-structural interior wall. The column and wall"
            " share the same off-white paint and the vertical boundary is"
            " subtle, like a real router hard case. The column is cropped"
            " by the top and bottom edges and occupies about one third of"
            " the frame; the adjacent wall fills the rest. Include minor"
            " scuffs, patchy paint, shadow at the wall-column joint, and"
            " small cracks, but keep the scene plain and close-up."
        ),
    },
    {
        "id": "hard_rc_column_against_rc_wall_02",
        "primary_class": "RC柱",
        "target_classes": ["RC柱", "RC壁"],
        "scenario": "hard_column_against_rc_wall",
        "prompt": (
            "Close inspection photograph of an exposed reinforced concrete"
            " stairwell or corridor in Japan where a rectangular RC column"
            " is integrated into a continuous RC shear wall. Both surfaces"
            " share similar grey concrete color and rough texture, so only"
            " a slight vertical offset, seam, and shadow reveal the column."
            " The column is partly cropped at one side of the image and the"
            " wall fills the rest. Use flat field-survey lighting, concrete"
            " stains, form-tie marks, and small diagonal cracks."
        ),
    },
    {
        "id": "hard_rc_column_corner_window_03",
        "primary_class": "RC柱",
        "target_classes": ["RC柱", "内壁"],
        "scenario": "hard_column_next_to_window",
        "prompt": (
            "Handheld inspection photograph of a Japanese office interior"
            " where a painted reinforced concrete column stands directly"
            " next to a window frame and a painted interior wall. The column"
            " and wall have the same finish, and the window edge runs along"
            " the column, making the column-wall boundary ambiguous. Crop"
            " the column at the top or side of the frame, with uneven light"
            " from the window and small cracks around the edge."
        ),
    },
    {
        "id": "hard_partial_rc_column_04",
        "primary_class": "RC柱",
        "target_classes": ["RC柱"],
        "scenario": "hard_partial_column_in_frame",
        "prompt": (
            "Close field-inspection photograph where only part of a"
            " rectangular reinforced concrete column is visible. The column"
            " is cut off by the image edge and does not show its full"
            " height, similar to a cropped survey photo. A painted interior"
            " wall or RC wall extends beside it, and the floor or ceiling"
            " slab is barely visible. Keep the column face large in the"
            " image, with scuffs, small cracks, and soft shadow at the edge."
        ),
    },
    {
        "id": "hard_rc_column_sliver_at_edge_05",
        "primary_class": "RC柱",
        "target_classes": ["RC柱", "内壁"],
        "scenario": "hard_column_sliver_at_image_edge",
        "prompt": (
            "Handheld Japanese building inspection photo where only a"
            " narrow vertical slice of a reinforced concrete column appears"
            " at the far left edge of the frame. Most of the image is a"
            " plain painted interior wall with similar color. The column is"
            " identifiable by a slight protrusion, darker side shadow, and"
            " vertical corner line, but it is easy to confuse with the wall."
        ),
    },
    {
        "id": "hard_rc_column_flat_front_06",
        "primary_class": "RC柱",
        "target_classes": ["RC柱"],
        "scenario": "hard_flat_column_face",
        "prompt": (
            "Close-up inspection photo of the broad flat front face of a"
            " rectangular RC column in a Japanese school or public facility."
            " The column face fills most of the frame and looks almost like"
            " a wall because its side edges are barely visible. Include a"
            " faint vertical edge shadow, paint wear near the lower part,"
            " and several thin cracks."
        ),
    },
    {
        "id": "hard_rc_column_corner_low_light_07",
        "primary_class": "RC柱",
        "target_classes": ["RC柱", "RC壁"],
        "scenario": "hard_column_corner_low_light",
        "prompt": (
            "Low-light handheld inspection photo of an RC column at a"
            " concrete wall corner in a Japanese basement or stairwell. The"
            " grey column and RC wall have nearly the same texture, and the"
            " only cue is a vertical corner shadow. Use slight motion blur,"
            " uneven exposure, stains, and a close crop."
        ),
    },
    {
        "id": "hard_rc_column_with_wall_joint_08",
        "primary_class": "RC柱",
        "target_classes": ["RC柱", "RC壁"],
        "scenario": "hard_wall_column_joint",
        "prompt": (
            "Inspection photo focused on the joint between a rectangular RC"
            " column and an RC wall. The image is tightly cropped so the"
            " column is not fully visible. The wall-column boundary is a"
            " thin vertical seam with similar concrete on both sides."
            " Include formwork marks, small cracks crossing the seam, and"
            " ordinary indoor lighting."
        ),
    },
    # ---------------- Hard cases: 内壁 vs RC壁 ----------------
    {
        "id": "hard_inner_wall_looks_rc_01",
        "primary_class": "内壁",
        "target_classes": ["内壁"],
        "scenario": "hard_inner_wall_looks_rc",
        "prompt": (
            "Close inspection photograph of a non-structural interior"
            " partition wall in a Japanese commercial building. It must read"
            " as an interior wall, not as exposed concrete: include"
            " wallpaper seams or painted gypsum-board texture, a baseboard,"
            " outlet or switch plate, and a flat lightweight partition"
            " surface. The color is grey or off-white so it superficially"
            " resembles RC, but there are no form-tie holes, no exposed"
            " aggregate, and no thick structural concrete edges. Add small"
            " cracks near a corner and realistic scuffs."
        ),
    },
    {
        "id": "hard_rc_wall_looks_inner_02",
        "primary_class": "RC壁",
        "target_classes": ["RC壁"],
        "scenario": "hard_rc_wall_looks_inner",
        "prompt": (
            "Close field-inspection photograph of a reinforced concrete"
            " shear wall in a Japanese stairwell or corridor that has been"
            " painted off-white. The paint makes it superficially resemble"
            " an interior wall, but subtle formwork seams, tie-hole circles,"
            " thickness at the edge, and concrete stains are still visible."
            " The wall fills the frame with a few diagonal hairline cracks"
            " and plain uneven lighting."
        ),
    },
    {
        "id": "hard_inner_wall_gray_partition_03",
        "primary_class": "内壁",
        "target_classes": ["内壁"],
        "scenario": "hard_gray_partition_wall",
        "prompt": (
            "Handheld inspection photo of a grey painted interior partition"
            " wall in a Japanese apartment or office corridor. The wall"
            " looks slightly like concrete because of the grey paint, but"
            " visible wallpaper seams, a thin baseboard, and a small outlet"
            " make it clearly non-structural. Tight crop, mild blur, uneven"
            " exposure, scuff marks, and thin cracks."
        ),
    },
    {
        "id": "hard_inner_wall_corner_shadow_04",
        "primary_class": "内壁",
        "target_classes": ["内壁"],
        "scenario": "hard_inner_wall_corner_shadow",
        "prompt": (
            "Close Japanese inspection photograph of an interior wall corner"
            " with a strong vertical shadow that could be mistaken for a"
            " concrete column edge. The surfaces are lightweight painted"
            " interior walls with wallpaper seams and baseboard. The frame"
            " is cropped tightly around the corner, with small cracks and"
            " scuffs, but no structural concrete texture."
        ),
    },
    {
        "id": "hard_rc_wall_painted_flat_05",
        "primary_class": "RC壁",
        "target_classes": ["RC壁"],
        "scenario": "hard_painted_flat_rc_wall",
        "prompt": (
            "Close-up inspection photo of a painted RC shear wall that looks"
            " like a simple interior wall at first glance. The image should"
            " still contain subtle structural cues: faint circular tie-hole"
            " marks, a construction joint, small concrete stains bleeding"
            " through paint, and diagonal cracks. Plain Japanese building"
            " corridor, tight crop, ordinary field-photo quality."
        ),
    },
    {
        "id": "hard_wall_ceiling_boundary_06",
        "primary_class": "RC壁",
        "target_classes": ["RC壁", "天井"],
        "scenario": "hard_wall_ceiling_boundary",
        "prompt": (
            "Handheld inspection photo at the boundary between a painted RC"
            " wall and a ceiling slab in a Japanese building. The wall is"
            " the main target and fills most of the frame, while a small"
            " strip of ceiling appears at the top. The wall is off-white,"
            " with subtle formwork seams, stains, and cracks; the crop and"
            " angle make wall vs ceiling slightly ambiguous."
        ),
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: list[dict], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def imagen_predict(
    api_key: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    image_size: str,
    timeout: int,
) -> dict:
    payload: dict[str, Any] = {
        "instances": [{"prompt": prompt + NEGATIVE_TAIL}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
            "personGeneration": "dont_allow",
        },
    }
    if image_size and "fast" not in model and "ultra" not in model:
        payload["parameters"]["imageSize"] = image_size
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:predict"
    )
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


def gemini_image_generate(
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
) -> dict:
    payload: dict[str, Any] = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt + NEGATIVE_TAIL}]},
        ],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
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


def generate_one(
    api_key: str,
    model: str,
    endpoint: str,
    prompt: str,
    aspect_ratio: str,
    image_size: str,
    timeout: int,
) -> tuple[bytes | None, dict]:
    if endpoint == "predict":
        body = imagen_predict(api_key, model, prompt, aspect_ratio, image_size, timeout)
        image = extract_imagen_bytes(body)
    else:
        body = gemini_image_generate(api_key, model, prompt, timeout)
        image = extract_gemini_bytes(body)
    return image, body


def compact_response(body: dict) -> dict:
    return {
        "modelVersion": body.get("modelVersion"),
        "responseId": body.get("responseId"),
        "raw_keys": sorted(body.keys()) if isinstance(body, dict) else None,
        "usageMetadata": body.get("usageMetadata"),
    }


def determine_endpoint(model: str) -> str:
    if model.startswith("imagen-"):
        return "predict"
    return "generateContent"


def worker(
    job: dict,
    api_key: str,
    model: str,
    endpoint: str,
    aspect_ratio: str,
    image_size: str,
    timeout: int,
    max_retries: int,
) -> dict:
    started = time.perf_counter()
    result = {
        **job,
        "model": model,
        "endpoint": endpoint,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "ok": False,
        "error": None,
        "started_at": utc_now(),
        "finished_at": None,
        "latency_sec": None,
        "attempts": 0,
        "response_compact": None,
        "image_path": None,
        "image_rel_path": None,
    }
    out_path = Path(job["image_path_planned"])
    if out_path.exists():
        result["ok"] = True
        result["image_path"] = str(out_path)
        result["image_rel_path"] = job["image_rel_path_planned"]
        result["error"] = "reused_existing_file"
        result["finished_at"] = utc_now()
        result["latency_sec"] = 0.0
        return result
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            image, body = generate_one(
                api_key, model, endpoint, job["prompt"], aspect_ratio, image_size, timeout,
            )
            result["response_compact"] = compact_response(body)
            if not image:
                result["error"] = "no_image_bytes"
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(image)
            result["ok"] = True
            result["image_path"] = str(out_path)
            result["image_rel_path"] = job["image_rel_path_planned"]
            result["error"] = None
            break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {message[:600]}"
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
        time.sleep(min(20, attempt * 3))
    result["finished_at"] = utc_now()
    result["latency_sec"] = round(time.perf_counter() - started, 3)
    return result


def build_jobs(
    out_dir: Path,
    samples_per_prompt: int,
    prompt_filter: list[str] | None,
    class_filter: list[str] | None,
) -> list[dict]:
    images_root = out_dir / "images"
    jobs: list[dict] = []
    for prompt_entry in PROMPTS:
        if prompt_filter and prompt_entry["id"] not in prompt_filter:
            continue
        if class_filter and prompt_entry["primary_class"] not in class_filter:
            continue
        class_dir = CLASS_DIR_NAMES[prompt_entry["primary_class"]]
        expected_dir = EXPECTED_DIRS[prompt_entry["primary_class"]]
        for sample_idx in range(samples_per_prompt):
            stem = f"{class_dir}__{prompt_entry['id']}__s{sample_idx:02d}"
            rel_path = f"images/{class_dir}/{stem}.png"
            jobs.append(
                {
                    "prompt_id": prompt_entry["id"],
                    "primary_class": prompt_entry["primary_class"],
                    "target_classes": prompt_entry["target_classes"],
                    "scenario": prompt_entry["scenario"],
                    "expected_label": prompt_entry["primary_class"],
                    "expected_dir": expected_dir,
                    "sample_index": sample_idx,
                    "prompt": prompt_entry["prompt"],
                    "image_path_planned": str(out_dir / rel_path),
                    "image_rel_path_planned": rel_path,
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/synthetic_router_generation_poc",
        help="Where to drop generated images and metadata.",
    )
    parser.add_argument(
        "--model",
        default="imagen-4.0-fast-generate-001",
        help=(
            "Image generation model. Confirmed candidates:"
            " imagen-4.0-fast-generate-001 (default, cheapest),"
            " imagen-4.0-generate-001 (higher quality),"
            " gemini-2.5-flash-image (Nano Banana, conversational)."
        ),
    )
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--aspect-ratio", default="4:3")
    parser.add_argument("--image-size", default="1K", help="Standard models only; fast ignores this.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--only-prompt-ids",
        nargs="*",
        default=None,
        help="Restrict to a subset of prompt IDs (defined in this script).",
    )
    parser.add_argument(
        "--only-classes",
        nargs="*",
        default=None,
        help="Restrict to one or more primary classes, e.g. RC柱 内壁.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only enumerate planned jobs and write the manifest, do not call the API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("Set GEMINI_API_KEY before running (or pass --dry-run).", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        out_dir,
        args.samples_per_prompt,
        args.only_prompt_ids,
        args.only_classes,
    )
    manifest_path = out_dir / "generation_plan.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(
        f"plan={len(jobs)} prompts={len(PROMPTS)} samples_per_prompt={args.samples_per_prompt}"
        f" model={args.model} out_dir={out_dir}",
        flush=True,
    )
    if args.dry_run:
        return 0

    endpoint = determine_endpoint(args.model)
    results_path = out_dir / "generation_results.jsonl"
    lock = threading.Lock()
    summary_counts: dict[str, int] = {"ok": 0, "errors": 0, "reused": 0}
    by_class: dict[str, int] = {}
    started = time.monotonic()
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                worker,
                job,
                api_key,
                args.model,
                endpoint,
                args.aspect_ratio,
                args.image_size,
                args.timeout,
                args.max_retries,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            write_jsonl(results_path, [result], lock)
            done += 1
            status = "ok" if result.get("ok") else "err"
            if status == "ok":
                if result.get("error") == "reused_existing_file":
                    summary_counts["reused"] += 1
                else:
                    summary_counts["ok"] += 1
                by_class[result["primary_class"]] = by_class.get(result["primary_class"], 0) + 1
            else:
                summary_counts["errors"] += 1
            elapsed = time.monotonic() - started
            rate = done / elapsed * 60 if elapsed else 0
            tag = result.get("prompt_id")
            print(
                f"{done}/{len(jobs)} {status} rate={rate:.2f}/min prompt={tag}"
                f" sample={result.get('sample_index')} latency={result.get('latency_sec')}s",
                flush=True,
            )

    summary = {
        "model": args.model,
        "endpoint": endpoint,
        "aspect_ratio": args.aspect_ratio,
        "image_size": args.image_size if endpoint == "predict" else None,
        "concurrency": args.concurrency,
        "samples_per_prompt": args.samples_per_prompt,
        "finished_at": utc_now(),
        "planned": len(jobs),
        "ok": summary_counts["ok"],
        "reused": summary_counts["reused"],
        "errors": summary_counts["errors"],
        "by_class": by_class,
        "results": str(results_path),
        "plan": str(manifest_path),
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
