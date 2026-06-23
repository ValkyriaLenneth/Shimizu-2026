#!/usr/bin/env python3
"""Generate synthetic router images and immediately annotate each image.

This is the scaled version of the small POC flow:

1. Generate a synthetic inspection photo with a Gemini image model.
2. If generation succeeds, annotate the image with the existing Gemini coarse
   building-element annotation prompt.
3. Write generation and annotation JSONL rows incrementally so the run can be
   resumed after interruption.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gemini_coarse_test import generate as annotate_image, parse_json_text  # noqa: E402
from run_gemini_image_generation_poc import (  # noqa: E402
    CLASS_DIR_NAMES,
    EXPECTED_DIRS,
    NEGATIVE_TAIL,
    PROMPTS,
    compact_response,
    determine_endpoint,
    extract_gemini_bytes,
    generate_one,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_REFERENCE_ROOT = (
    "handoff_20260519/shimizu_20260519_minimal_repro_package/data/unzip"
)

REFERENCE_DIRS = {
    "天井": "1_天井",
    "内壁": "2_内壁",
    "RC壁": "3_RC壁",
    "RC柱": "4_RC柱",
}

BUILDING_TYPES = [
    "school corridor",
    "municipal office building",
    "apartment common hallway",
    "hospital service corridor",
    "stairwell landing",
    "underground parking area",
    "public hall back corridor",
    "older reinforced concrete facility",
    "warehouse-like service space",
    "small inspection room",
]

CAMERA_ANGLES = [
    "head-on but slightly tilted",
    "oblique from the lower left",
    "oblique from the lower right",
    "looking upward at a shallow angle",
    "looking downward slightly",
    "very close crop with imperfect verticals",
    "partial side view with a cropped edge",
    "off-centre framing like a quick field photo",
]

DISTANCES = [
    "very close, target surface fills almost the whole frame",
    "close, only a narrow strip of surrounding context is visible",
    "medium-close, one adjacent boundary is visible",
    "cropped so one side of the target is outside the frame",
    "cropped at top or bottom like a phone snapshot",
]

LIGHTING = [
    "dim fluorescent indoor light",
    "uneven daylight from one side",
    "low light with mild motion blur",
    "overexposed patch near one edge",
    "yellowish corridor lighting",
    "flat cloudy daylight",
    "harsh shadow along one joint",
]

DAMAGE_PATTERNS = [
    "several thin diagonal cracks crossing the target surface",
    "one clear vertical crack plus small branching hairline cracks",
    "paint peeling around a crack and small scuff marks",
    "water staining with hairline cracks near the stain",
    "rust bleeding from a small spalled area",
    "fine map cracking across a painted surface",
    "one wider crack near a boundary or corner",
    "minor concrete spalling with aggregate visible",
    "efflorescence streaks and surface wear",
    "impact scuffs with short diagonal cracks",
]

SURFACE_VARIANTS = [
    "off-white painted surface",
    "grey painted surface",
    "bare concrete with formwork marks",
    "aged concrete with stains",
    "wallpapered surface with seams",
    "painted gypsum-board-like partition finish",
    "rough concrete with tie-hole marks",
    "slightly glossy old paint",
]

CLASS_SURFACE_VARIANTS = {
    "天井": [
        "flat painted ceiling surface",
        "suspended acoustic ceiling board surface",
        "gypsum-board ceiling with panel seams",
        "aged off-white ceiling paint with stains",
        "ceiling surface around a recessed light fixture",
        "ceiling surface around an HVAC vent",
        "plain corridor ceiling with inspection stains",
    ],
    "内壁": [
        "off-white painted interior partition",
        "grey painted gypsum-board-like partition",
        "wallpapered surface with visible seams",
        "cream painted interior wall",
        "lightweight partition finish with baseboard",
        "slightly glossy old paint on an interior wall",
        "painted surface with outlet or switch plate nearby",
    ],
    "RC壁": [
        "bare concrete wall with formwork marks",
        "painted RC shear wall with subtle tie-hole marks",
        "aged concrete wall with stains",
        "rough concrete surface with construction joints",
        "off-white painted structural concrete wall",
        "weathered exterior RC wall surface",
        "basement RC wall with rust or efflorescence",
    ],
    "RC柱": [
        "painted rectangular RC column face",
        "bare concrete column with formwork marks",
        "aged concrete column surface",
        "off-white column surface matching adjacent wall",
        "grey concrete column with edge shadow",
        "parking-garage RC column surface",
        "column face with scuffs near the lower part",
    ],
}

CONTEXT_CUES = [
    "a small strip of ceiling appears at the top",
    "a baseboard is visible near the bottom",
    "a wall-column joint is visible at one side",
    "a narrow window or opening edge appears near the target",
    "a beam or slab edge appears only partially",
    "a corner shadow cuts across the target",
    "a pipe or conduit is barely visible near the boundary",
    "only the target surface is visible, with almost no context",
]

CLASS_CONTEXT_CUES = {
    "天井": [
        "a small strip of wall appears at the edge",
        "a light fixture or vent appears near one side",
        "a sprinkler head or small ceiling fixture is visible",
        "a ceiling access panel is cropped near one edge",
        "only the ceiling surface is visible, with almost no context",
        "a corner where ceiling meets wall is cropped tightly",
    ],
    "内壁": [
        "a baseboard is visible near the bottom",
        "an outlet or switch plate appears near the edge",
        "a door or window frame edge is cropped at one side",
        "a vertical corner shadow cuts across the wall",
        "a small strip of ceiling appears at the top",
        "only the interior wall surface is visible",
    ],
    "RC壁": [
        "a small strip of ceiling appears at the top",
        "a construction joint or tie-hole row is visible",
        "a wall-column joint is visible at one side",
        "a slab or beam edge appears only partially",
        "a corner shadow cuts across the concrete wall",
        "only the RC wall surface is visible, with almost no context",
    ],
    "RC柱": [
        "a wall-column joint is visible at one side",
        "one column edge is cropped by the image border",
        "a ceiling or floor slab is barely visible",
        "an adjacent wall with similar paint appears beside the column",
        "a vertical edge shadow reveals the column corner",
        "only the column face is visible, making it wall-like",
    ],
}

CLASS_POSITIVE_CONSTRAINTS = {
    "天井": (
        "Ceiling-specific requirement: the generated image must clearly be"
        " a building ceiling / overhead interior surface. The ceiling plane"
        " should occupy at least 70 percent of the image. The camera should"
        " look upward or nearly upward. It is acceptable to show a thin wall"
        " edge, light fixture, vent, sprinkler, access panel, or ceiling"
        " corner, but no vertical column, pillar, large wall, stairwell frame,"
        " or structural column-beam scene may dominate the image."
    ),
    "内壁": (
        "Interior-wall-specific requirement: the generated image must clearly"
        " be an interior wall or partition surface. The wall should occupy"
        " most of the image. Ceiling, floor, doors, or fixtures may appear"
        " only as minor context."
    ),
    "RC壁": (
        "RC-wall-specific requirement: the generated image must clearly be a"
        " reinforced concrete wall or shear wall surface. It may be bare or"
        " painted, but the wall surface must dominate the image. Columns may"
        " appear only as minor edge context."
    ),
    "RC柱": (
        "RC-column-specific requirement: the generated image must clearly be"
        " a reinforced concrete column or pillar. The column should be the"
        " main object even when it is partially cropped or wall-like."
    ),
}

CLASS_NEGATIVE_CONSTRAINTS = {
    "天井": (
        "Do not generate a free-standing column, pillar, vertical wall, large"
        " concrete frame, stairwell, parking-garage column row, or beam-column"
        " structural composition as the main subject. If columns are visible,"
        " they must be tiny peripheral context, not the target."
    ),
    "内壁": (
        "Do not make the ceiling or an RC column the main subject. Do not add"
        " strong formwork/tie-hole patterns unless the prompt explicitly asks"
        " for an inner wall that only superficially resembles RC."
    ),
    "RC壁": (
        "Do not make a standalone column the main subject. Do not make the"
        " image look like a lightweight wallpapered partition unless the"
        " prompt asks for painted RC wall ambiguity."
    ),
    "RC柱": (
        "Do not make the adjacent wall, ceiling, or floor dominate the image."
        " The column must remain visually identifiable as the target."
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done(path: Path) -> set[str]:
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


def prompt_pool(classes: list[str]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for cls in classes:
        rows = [p for p in PROMPTS if p["primary_class"] == cls]
        if not rows:
            raise RuntimeError(f"No prompts found for class {cls}")
        pools[cls] = rows
    return pools


def load_reference_pools(reference_root: Path, classes: list[str], seed: int) -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = {}
    for cls in classes:
        dirname = REFERENCE_DIRS.get(cls)
        if not dirname:
            pools[cls] = []
            continue
        root = reference_root / dirname
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng = random.Random(f"{seed}:reference:{cls}")
        rng.shuffle(paths)
        pools[cls] = paths
    return pools


def pick_variation(cls: str, idx: int, seed: int) -> dict[str, str]:
    rng = random.Random(f"{seed}:variation:{cls}:{idx}")
    surface_pool = CLASS_SURFACE_VARIANTS.get(cls, SURFACE_VARIANTS)
    context_pool = CLASS_CONTEXT_CUES.get(cls, CONTEXT_CUES)
    return {
        "building_type": rng.choice(BUILDING_TYPES),
        "camera_angle": rng.choice(CAMERA_ANGLES),
        "distance": rng.choice(DISTANCES),
        "lighting": rng.choice(LIGHTING),
        "damage": rng.choice(DAMAGE_PATTERNS),
        "surface": rng.choice(surface_pool),
        "context": rng.choice(context_pool),
    }


def compose_prompt(base_prompt: str, cls: str, idx: int, seed: int, has_reference: bool) -> tuple[str, dict[str, str]]:
    variation = pick_variation(cls, idx, seed)
    reference_text = ""
    if has_reference:
        reference_text = (
            "\n\nA real inspection photo is attached as a visual reference."
            " Use it only for dataset style, framing, camera quality, and"
            " realism. Do not copy the exact scene, layout, damage pattern,"
            " or any identifiable details. Generate a new synthetic photo"
            " with the same rough field-survey feeling."
        )
    variation_text = (
        "\n\nFor this specific synthetic sample, force these variations:"
        f"\n- building type: {variation['building_type']}"
        f"\n- camera angle/framing: {variation['camera_angle']}"
        f"\n- distance/crop: {variation['distance']}"
        f"\n- lighting: {variation['lighting']}"
        f"\n- target surface finish: {variation['surface']}"
        f"\n- visible damage: {variation['damage']}"
        f"\n- context cue: {variation['context']}"
        "\nThe image must still primarily depict the requested class and"
        " should not become a generic room scene."
    )
    return base_prompt + reference_text + variation_text, variation


def pick_references(
    reference_pools: dict[str, list[Path]],
    cls: str,
    idx: int,
    references_per_job: int,
) -> list[str]:
    paths = reference_pools.get(cls, [])
    if not paths or references_per_job <= 0:
        return []
    refs = []
    for offset in range(references_per_job):
        refs.append(str(paths[(idx * references_per_job + offset) % len(paths)]))
    return refs


def build_jobs(
    out_dir: Path,
    per_class: int,
    classes: list[str],
    seed: int = 20260526,
    reference_root: Path | None = None,
    references_per_job: int = 1,
) -> list[dict[str, Any]]:
    pools = prompt_pool(classes)
    reference_pools = (
        load_reference_pools(reference_root, classes, seed)
        if reference_root and references_per_job > 0
        else {cls: [] for cls in classes}
    )
    jobs: list[dict[str, Any]] = []
    for cls in classes:
        prompts = pools[cls]
        class_dir = CLASS_DIR_NAMES[cls]
        expected_dir = EXPECTED_DIRS[cls]
        for idx in range(per_class):
            prompt = prompts[idx % len(prompts)]
            cycle = idx // len(prompts)
            reference_paths = pick_references(reference_pools, cls, idx, references_per_job)
            composed_prompt, variation = compose_prompt(
                prompt["prompt"],
                cls,
                idx,
                seed,
                bool(reference_paths),
            )
            stem = f"{class_dir}__{prompt['id']}__n{idx:03d}_c{cycle:02d}_v2"
            rel_path = f"images/{class_dir}/{stem}.png"
            jobs.append(
                {
                    "job_id": f"{cls}_{idx:03d}",
                    "prompt_id": prompt["id"],
                    "primary_class": cls,
                    "target_classes": prompt["target_classes"],
                    "scenario": prompt["scenario"],
                    "expected_label": cls,
                    "expected_dir": expected_dir,
                    "sample_index": idx,
                    "prompt_cycle": cycle,
                    "prompt": composed_prompt,
                    "base_prompt": prompt["prompt"],
                    "variation": variation,
                    "reference_image_paths": reference_paths,
                    "image_path_planned": str(out_dir / rel_path),
                    "image_rel_path_planned": rel_path,
                }
            )
    return jobs


def generate_with_retries(
    job: dict[str, Any],
    api_key: str,
    model: str,
    endpoint: str,
    aspect_ratio: str,
    image_size: str,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = Path(job["image_path_planned"])
    result = {
        **job,
        "model": model,
        "endpoint": endpoint,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size if endpoint == "predict" else None,
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
    if out_path.exists():
        result.update(
            {
                "ok": True,
                "error": "reused_existing_file",
                "image_path": str(out_path),
                "image_rel_path": job["image_rel_path_planned"],
                "finished_at": utc_now(),
                "latency_sec": 0.0,
            }
        )
        return result
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            image, body = generate_one_pipeline(
                api_key,
                model,
                endpoint,
                job,
                aspect_ratio,
                image_size,
                timeout,
            )
            result["response_compact"] = compact_response(body)
            if not image:
                result["error"] = "no_image_bytes"
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(image)
            result.update(
                {
                    "ok": True,
                    "error": None,
                    "image_path": str(out_path),
                    "image_rel_path": job["image_rel_path_planned"],
                }
            )
            break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {message[:800]}"
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
        time.sleep(min(30, attempt * 4))
    result["finished_at"] = utc_now()
    result["latency_sec"] = round(time.perf_counter() - started, 3)
    return result


def gemini_image_generate_with_references(
    api_key: str,
    model: str,
    prompt: str,
    reference_image_paths: list[str],
    timeout: int,
) -> dict:
    parts: list[dict[str, Any]] = [{"text": prompt + NEGATIVE_TAIL}]
    for image_path in reference_image_paths:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            }
        )
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
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


def generate_one_pipeline(
    api_key: str,
    model: str,
    endpoint: str,
    job: dict[str, Any],
    aspect_ratio: str,
    image_size: str,
    timeout: int,
) -> tuple[bytes | None, dict]:
    reference_paths = job.get("reference_image_paths") or []
    if endpoint == "generateContent" and reference_paths:
        body = gemini_image_generate_with_references(
            api_key,
            model,
            job["prompt"],
            reference_paths,
            timeout,
        )
        return extract_gemini_bytes(body), body
    return generate_one(
        api_key,
        model,
        endpoint,
        job["prompt"],
        aspect_ratio,
        image_size,
        timeout,
    )


def prompt_generation_instruction(job: dict[str, Any]) -> str:
    cls = job["primary_class"]
    variation = job.get("variation") or {}
    return (
        "You are writing one image-generation prompt for a synthetic training"
        " image of Japanese building inspection photos.\n\n"
        "Use the attached real photo only as a reference for dataset style,"
        " framing, camera quality, and field-survey realism. Do not copy the"
        " exact scene, geometry, damage, or identifiable details.\n\n"
        f"Target class: {cls}\n"
        f"Base scenario: {job.get('base_prompt') or job.get('prompt')}\n\n"
        "Required variation for this sample:\n"
        f"- building type: {variation.get('building_type')}\n"
        f"- camera angle/framing: {variation.get('camera_angle')}\n"
        f"- distance/crop: {variation.get('distance')}\n"
        f"- lighting: {variation.get('lighting')}\n"
        f"- target surface finish: {variation.get('surface')}\n"
        f"- visible damage: {variation.get('damage')}\n"
        f"- context cue: {variation.get('context')}\n\n"
        f"Positive class constraint: {CLASS_POSITIVE_CONSTRAINTS.get(cls, '')}\n"
        f"Negative class constraint: {CLASS_NEGATIVE_CONSTRAINTS.get(cls, '')}\n\n"
        "Write a concrete, visually specific generation prompt in English."
        " Make the output diverse and faithful to the reference-photo style,"
        " but keep the target class unambiguous. Damage must be visible and"
        " physically plausible.\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "generation_prompt": "prompt text",\n'
        '  "reference_style_summary": "short summary",\n'
        '  "diversity_notes": "what changed from the reference"\n'
        "}"
    )


def generate_prompt_once(
    api_key: str,
    model: str,
    job: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt_generation_instruction(job)}]
    for image_path in job.get("reference_image_paths") or []:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            }
        )
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.85,
            "responseMimeType": "application/json",
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
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
        body = json.load(resp)
    text = body["candidates"][0]["content"]["parts"][0].get("text", "")
    parsed = parse_json_text(text)
    return {"raw": body, "parsed": parsed, "text": text}


def prompt_with_retries(
    job: dict[str, Any],
    api_key: str,
    prompt_model: str,
    timeout: int,
    max_retries: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = {
        **job,
        "model": prompt_model,
        "ok": False,
        "error": None,
        "response": None,
        "final_prompt": job["prompt"],
        "source": "template_fallback",
    }
    if not enabled:
        result["ok"] = True
        result["source"] = "template_disabled"
        return result, {**job, "prompt": job["prompt"], "prompt_source": result["source"]}
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            response = generate_prompt_once(api_key, prompt_model, job, timeout)
            parsed = response.get("parsed") or {}
            final_prompt = str(parsed.get("generation_prompt") or "").strip()
            if not final_prompt:
                result["error"] = "empty_generation_prompt"
                continue
            result.update(
                {
                    "ok": True,
                    "error": None,
                    "response": {
                        "parsed": parsed,
                        "text": response.get("text"),
                        "raw_compact": compact_response(response.get("raw") or {}),
                    },
                    "final_prompt": final_prompt,
                    "source": "gemini_prompt_generation",
                }
            )
            generated_job = {
                **job,
                "prompt": final_prompt,
                "template_prompt": job["prompt"],
                "prompt_source": result["source"],
                "prompt_generation_model": prompt_model,
                "prompt_generation_response": parsed,
            }
            return result, generated_job
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {message[:800]}"
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
        time.sleep(min(30, attempt * 4))
    return result, {**job, "prompt": job["prompt"], "prompt_source": "template_fallback_after_error"}


def annotate_with_retries(
    generation_row: dict[str, Any],
    api_key: str,
    annotation_model: str,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    row = {
        "expected_dir": generation_row.get("expected_dir"),
        "expected_label": generation_row.get("expected_label"),
        "image_path": generation_row["image_path"],
        "image_rel_path": generation_row.get("image_rel_path") or generation_row["image_path"],
        "prompt_id": generation_row.get("prompt_id"),
        "primary_class": generation_row.get("primary_class"),
        "target_classes": generation_row.get("target_classes"),
        "scenario": generation_row.get("scenario"),
        "sample_index": generation_row.get("sample_index"),
        "variation": generation_row.get("variation"),
        "reference_image_paths": generation_row.get("reference_image_paths"),
        "prompt_source": generation_row.get("prompt_source"),
        "prompt_generation_response": generation_row.get("prompt_generation_response"),
        "generation_model": generation_row.get("model"),
    }
    result = {
        **row,
        "model": annotation_model,
        "ok": False,
        "error": None,
        "response": None,
        "source": "gemini_api_synthetic_pipeline",
        "imported": False,
    }
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            result["response"] = annotate_image(
                api_key,
                Path(generation_row["image_path"]),
                annotation_model,
                timeout,
            )
            result["ok"] = True
            result["error"] = None
            break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {exc.code}: {message[:800]}"
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
        time.sleep(min(30, attempt * 4))
    return result


def pipeline_worker(
    job: dict[str, Any],
    api_key: str,
    prompt_model: str,
    generation_model: str,
    annotation_model: str,
    endpoint: str,
    aspect_ratio: str,
    image_size: str,
    prompt_timeout: int,
    generation_timeout: int,
    annotation_timeout: int,
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    prompt_result, prompted_job = prompt_with_retries(
        job,
        api_key,
        prompt_model,
        prompt_timeout,
        max_retries,
        enabled=True,
    )
    gen = generate_with_retries(
        prompted_job,
        api_key,
        generation_model,
        endpoint,
        aspect_ratio,
        image_size,
        generation_timeout,
        max_retries,
    )
    if not gen.get("ok"):
        return prompt_result, gen, None
    ann = annotate_with_retries(
        gen,
        api_key,
        annotation_model,
        annotation_timeout,
        max_retries,
    )
    return prompt_result, gen, ann


def summarize_annotations(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    coverage: dict[str, dict[str, int]] = {}
    detected_counts: dict[str, int] = {}
    ok = 0
    for row in rows:
        label = row.get("expected_label") or ""
        coverage.setdefault(label, {"annotated": 0, "matched_primary_class": 0})
        coverage[label]["annotated"] += 1
        if not row.get("ok"):
            continue
        ok += 1
        parsed = (row.get("response") or {}).get("parsed") or {}
        detected = {
            e.get("label")
            for e in parsed.get("elements", []) or []
            if isinstance(e, dict) and e.get("label")
        }
        for label_name in detected:
            detected_counts[label_name] = detected_counts.get(label_name, 0) + 1
        if label in detected:
            coverage[label]["matched_primary_class"] += 1
    return {
        "total": len(rows),
        "ok": ok,
        "errors": len(rows) - ok,
        "detected_label_counts": detected_counts,
        "primary_class_coverage": coverage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/synthetic_router_pipeline_nb2_100x4")
    parser.add_argument("--prompt-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--generation-model", default="gemini-3.1-flash-image-preview")
    parser.add_argument("--annotation-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--classes", nargs="+", default=["天井", "内壁", "RC壁", "RC柱"])
    parser.add_argument("--aspect-ratio", default="4:3")
    parser.add_argument("--image-size", default="1K")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-timeout", type=int, default=180)
    parser.add_argument("--generation-timeout", type=int, default=240)
    parser.add_argument("--annotation-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--references-per-job", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("Set GEMINI_API_KEY before running.", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_root = Path(args.reference_root) if args.reference_root else None
    jobs = build_jobs(
        out_dir,
        args.per_class,
        args.classes,
        seed=args.seed,
        reference_root=reference_root,
        references_per_job=args.references_per_job,
    )
    if args.limit:
        jobs = jobs[: args.limit]
    plan_path = out_dir / "pipeline_plan.jsonl"
    prompt_results = out_dir / "prompt_results.jsonl"
    generation_results = out_dir / "generation_results.jsonl"
    annotation_results = out_dir / "annotation_results.jsonl"
    with plan_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    annotated_done = load_done(annotation_results)
    jobs = [job for job in jobs if job["image_path_planned"] not in annotated_done]
    endpoint = determine_endpoint(args.generation_model)
    print(
        f"pipeline_jobs={len(jobs)} per_class={args.per_class} classes={args.classes} "
        f"prompt_model={args.prompt_model} generation_model={args.generation_model} "
        f"annotation_model={args.annotation_model} "
        f"endpoint={endpoint} concurrency={args.concurrency} seed={args.seed} "
        f"references_per_job={args.references_per_job} out_dir={out_dir}",
        flush=True,
    )
    if args.dry_run:
        return 0

    lock = threading.Lock()
    started = time.monotonic()
    counts = {"prompt_ok": 0, "prompt_err": 0, "gen_ok": 0, "gen_err": 0, "ann_ok": 0, "ann_err": 0}
    by_class: dict[str, int] = {}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                pipeline_worker,
                job,
                api_key,
                args.prompt_model,
                args.generation_model,
                args.annotation_model,
                endpoint,
                args.aspect_ratio,
                args.image_size,
                args.prompt_timeout,
                args.generation_timeout,
                args.annotation_timeout,
                args.max_retries,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            prompt_result, gen, ann = future.result()
            append_jsonl(prompt_results, prompt_result, lock)
            if prompt_result.get("ok"):
                counts["prompt_ok"] += 1
            else:
                counts["prompt_err"] += 1
            append_jsonl(generation_results, gen, lock)
            if gen.get("ok"):
                counts["gen_ok"] += 1
                by_class[gen["primary_class"]] = by_class.get(gen["primary_class"], 0) + 1
            else:
                counts["gen_err"] += 1
            if ann is not None:
                append_jsonl(annotation_results, ann, lock)
                if ann.get("ok"):
                    counts["ann_ok"] += 1
                else:
                    counts["ann_err"] += 1
            done += 1
            elapsed = max(1e-6, time.monotonic() - started)
            rate = done / elapsed * 60
            status = "ok" if gen.get("ok") and ann and ann.get("ok") else "err"
            print(
                f"{done}/{len(jobs)} {status} rate={rate:.2f}/min "
                f"class={gen.get('primary_class')} prompt={gen.get('prompt_id')} "
                f"prompt_gen={prompt_result.get('ok')} gen={gen.get('ok')} ann={ann.get('ok') if ann else None}",
                flush=True,
            )

    ann_summary = summarize_annotations(annotation_results)
    summary = {
        "generation_model": args.generation_model,
        "prompt_model": args.prompt_model,
        "annotation_model": args.annotation_model,
        "endpoint": endpoint,
        "classes": args.classes,
        "per_class": args.per_class,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "reference_root": args.reference_root,
        "references_per_job": args.references_per_job,
        "finished_at": utc_now(),
        "planned_total": len(
            build_jobs(
                out_dir,
                args.per_class,
                args.classes,
                seed=args.seed,
                reference_root=reference_root,
                references_per_job=args.references_per_job,
            )
        ),
        "processed_this_run": done,
        "counts": counts,
        "generated_by_class_this_run": by_class,
        "annotation_summary": ann_summary,
        "plan": str(plan_path),
        "prompt_results": str(prompt_results),
        "generation_results": str(generation_results),
        "annotation_results": str(annotation_results),
        "out_dir": str(out_dir),
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
