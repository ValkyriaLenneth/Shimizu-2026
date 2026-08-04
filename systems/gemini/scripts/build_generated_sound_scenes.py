#!/usr/bin/env python3
"""S5 - generate SOUND element scenes, so negatives stop being the scarce side.

The 2026-08-04 false-alarm measurement is the reason this exists:

    real negatives, already trained on : 0.0% fire rate, peak score median 0.000
    counterfactual negatives, unseen   : 95.7%, median 0.622

The model has not learnt "no damage here"; it memorised the 141 zero-box images.
Repetition of those saturates at 31-35% and turns harmful by 48%, so the binding
shortage is not negative *volume* but negative *diversity*, and diversity cannot
come from re-using the same 141 scenes.

Why generate sound scenes rather than repair damaged ones (S1)
--------------------------------------------------------------
S1 asked the model to remove damage, which on steel it does by rebuilding the
member: 45% of column_base and over half of brace outputs had a component added,
removed or reshaped, and the QC gate passed them because a rebuilt member scores
full marks on "is the damage gone". Generating a sound element from scratch has
no repair step, so that failure mode does not exist. The label is also certain by
construction - we asked for an undamaged element, so it is a negative - which
means no QC judge and no pseudo-labels.

Pairing with copy-paste keeps both arms on one path
---------------------------------------------------
S1 also died of round-trip mismatch: only the synthetic side passed through the
model's re-encoding, so "generated texture" was separable from real positives by
low-level statistics alone, and the 80-epoch dual-arm run measured the cost (the
S1 arm reached no P>=0.60 operating point at all). Here the same generated images
serve both roles: left alone they are negatives, and with a real damage crop
Poisson-blended in they are positives. "Was this generated" then carries no label
information, because every image in the set was.

Diversity comes from sampling different real negatives as visual references on
each call, plus a rotating scene brief, so the batch does not collapse onto one
imagined site.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path

from PIL import Image

import synth_common as sc

CATEGORY_EN = {
    "brace": "steel brace / bracing member of a building frame",
    "column_base": "steel column base (柱脚) seated on a concrete pedestal",
}

# Rotating briefs. Element and lighting vary so a batch does not collapse onto a
# single imagined site; the "sound" requirement is repeated in each.
SCENES = [
    "an indoor factory or warehouse floor, overhead lighting, dusty concrete floor",
    "an outdoor walkway beside a building, overcast daylight, wet ground",
    "a rooftop or exposed machine deck, bright daylight with hard shadows",
    "a narrow service corridor with pipes and cable trays, dim mixed lighting",
    "a parking structure or open ground floor, diffuse shade, gravel and debris",
    "an equipment yard beside a plant building, late afternoon light",
]

PROMPT = """These reference photographs come from a Japanese building condition survey and
show {element_en} in SOUND condition.

Generate ONE new photograph of a DIFFERENT {element_en}, in this setting:
{scene}

Requirements:
- documentary style: handheld by a surveyor with an ordinary camera, not a render,
  not a studio shot, no people, no text overlay
- the element occupies a similar share of the frame as in the references
- ordinary ageing is expected and wanted: dust, dirt, water stains, efflorescence,
  faded or chalky paint, scuff marks, moss, surrounding pipes, cables, floor debris
- but the element itself is UNDAMAGED: no cracks, no spalling, no exposed rebar,
  no rust scaling with section loss, no deformation, no gap under the base plate,
  no fractured or buckled member
Return only the image."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="column_base", choices=sorted(CATEGORY_EN))
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--out-dir", default="outputs/gemini_synth/s5_generated_sound")
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--model", default="gemini-3-pro-image")
    p.add_argument("--image-size", default="2K", choices=["1K", "2K", "4K"])
    p.add_argument("--references", type=int, default=3)
    p.add_argument("--ref-size", type=int, default=768)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260804)
    return p.parse_args()


def sound_images(paired: Path, cat: str) -> list[Path]:
    out = []
    for lab in sorted((paired / cat / "labels").glob("*.txt")):
        if lab.read_text().strip():
            continue
        img = sc.find_image(paired / cat / "images", lab.stem)
        if img is not None:
            out.append(img)
    return out


def one(idx: int, refs: list[Path], scene: str, args, key: str, out_dir: Path) -> dict:
    name = f"gen_{args.category}_{idx:04d}"
    dst = out_dir / "images" / f"{name}.jpg"
    if dst.exists():
        return {"stem": name, "skipped": True}
    try:
        parts = []
        for p in refs:
            with Image.open(p) as h:
                parts.append(sc.encode_image(h.convert("RGB"), max_side=args.ref_size))
        parts.append({"text": PROMPT.format(element_en=CATEGORY_EN[args.category], scene=scene)})
    except Exception as exc:
        return {"stem": name, "error": f"refs_failed: {exc}"}

    img, err = sc.generate_image(args.model, parts, key, timeout=args.timeout,
                                 max_retries=args.max_retries, image_size=args.image_size)
    if img is None:
        return {"stem": name, "error": err or "generate_failed"}
    img.save(dst, quality=95, subsampling=0)
    # Empty label: the image is a negative by construction.
    (out_dir / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
    return {"stem": name, "size": list(img.size), "scene": scene,
            "references": [p.stem for p in refs]}


def main() -> int:
    args = parse_args()
    key = sc.require_api_key()
    rng = random.Random(args.seed)
    paired = Path(args.paired_dir)
    pool = sound_images(paired, args.category)
    if len(pool) < args.references:
        print(f"only {len(pool)} sound reference images; need {args.references}")
        return 1

    out_dir = Path(args.out_dir) / args.category
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    print(f"category={args.category}  references available={len(pool)}  target={args.count}")

    jobs = []
    for i in range(args.count):
        refs = rng.sample(pool, args.references)
        jobs.append((i, refs, SCENES[i % len(SCENES)]))

    results, ok, skipped = [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool_ex:
        futs = {pool_ex.submit(one, i, r, s, args, key, out_dir): i for i, r, s in jobs}
        for n, f in enumerate(concurrent.futures.as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if r.get("skipped"):
                skipped += 1
            elif "error" in r:
                print(f"[{n}/{len(jobs)}] ERR {r['stem']}: {r['error'][:70]}")
            else:
                ok += 1
                if ok % 10 == 0:
                    print(f"[{n}/{len(jobs)}] {ok} generated")

    (out_dir / "manifest.json").write_text(json.dumps({
        "category": args.category, "model": args.model, "image_size": args.image_size,
        "generated": ok, "skipped": skipped, "seed": args.seed,
        "label_semantics": "empty by construction - the prompt asks for an undamaged element",
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n  生成 {ok} 张（跳过已存在 {skipped}）-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
