#!/usr/bin/env python3
"""S1 - counterfactual negatives by damage removal.

Takes a real damaged photograph, repairs every annotated damage box back to a
sound state, and emits the result with an empty label file. The product is a
negative that shares its scene, element, framing, lighting and camera with a
real positive, which is the most direct available attack on the element/damage
correlation documented in `2026-07-26-new-classes-shortcut-learning-finding.md`.

Three properties are enforced by construction, each answering a measured
failure:

1. **Only the damage boxes are written back.** The model sees a wide context
   window, but compositing is masked to the boxes. Everything else keeps its
   original pixels, so the over-cleaning measured by `probe_damage_removal.py`
   on 2026-08-03 - which erased rust staining, ageing and even the surveyor's
   red inspection circles - cannot reach outside the box.
2. **Real negatives are supplied as visual references.** Text alone did not
   hold the model to "sound but aged"; it produced pristine new-looking
   elements. Reference photographs from the same category's empty-label pool
   define the target condition instead.
3. **Every box on the image is repaired.** Repairing one box while others
   remain would emit an image labelled negative that still contains damage.
   Clusters are processed in sequence over the accumulating result.

Grain is re-matched inside the mask so the edited region does not read as
smoother than the surrounding photograph.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import synth_common as sc

REPAIR_INSTRUCTION = """You are editing one region of a real Japanese building damage survey photograph.
The structural element is {element_ja} ({element_en}).

The FIRST image is the region to edit. The images after it are reference photographs of
UNDAMAGED examples of the same element type, taken in the same kind of survey. Study them:
they show what "sound condition" looks like in this dataset - aged, dirty, stained,
imperfectly lit, ordinary. That is your target, NOT a new or renovated element.

TASK
Repair only the structural damage in the edit region so the element is in sound condition:
{damage_hint}

MUST PRESERVE - these are not damage:
- rust staining, water stains, efflorescence, dirt, grime, mould, discolouration
- paint that is merely old, faded, chalky or unevenly toned
- construction marks, chalk lines, crayon or paint survey markings, RED INSPECTION
  CIRCLES and any handwriting drawn by the surveyor
- form-tie holes, joint lines, chamfers, casting seams, minor surface irregularity
- every bolt, plate, pipe, conduit, cable, bracket and fixture, in its exact position
- the camera viewpoint, framing, perspective, and every object boundary
- the lighting, shadow direction, colour temperature, exposure and white balance
- image sharpness, sensor noise, motion blur and JPEG character

MUST NOT
- do not beautify, tidy, re-light, re-paint or modernise the scene
- do not make surfaces uniform, smooth or freshly finished
- do not remove ageing; an element can be very dirty and still be undamaged
- do not add, move or delete any object other than the damage itself
- do not change the aspect ratio

The result must look like the same photograph of the same element, taken by the same
camera under the same conditions, before the damage occurred."""

DAMAGE_HINT = {
    "brace": (
        "remove corrosion pitting, section loss, paint blistering and peeling caused by "
        "corrosion, cracks at connections, and any buckling or deformation of the member; "
        "the brace and its gusset plates and bolts must read as straight, intact and sound"
    ),
    "column_base": (
        "remove concrete cracking, spalling, delamination, exposed or corroded reinforcement, "
        "corroded anchor bolts and base plates, loss of concrete section, and loose debris; "
        "the concrete pedestal must read as continuous and intact, and steel as sound"
    ),
}

# Corrosion is the hard case. On steel the model re-renders rust rather than
# removing it - the 2026-08-03 batch scored 0/3 passes on corrosion-only images
# against 5/17 on concrete, and one repair added a pool of orange rust that was
# not in the original. Naming the finished surface concretely works better than
# asking for corrosion to be "removed".
STEEL_HINT = (
    "the steel must end up with a CLEAN, INTACT protective surface: an even mill or "
    "painted finish in its original colour, continuous over the whole member. There must "
    "be NO orange, brown, red or ochre corrosion product anywhere, NO flaking or scaling, "
    "NO pitting, NO rust bleed or run-off staining onto adjacent concrete, and NO section "
    "loss - bolt threads, nuts, washers, plate edges and welds must read as complete and "
    "sharply defined. Do not redraw the rust in a tidier form; the corroded material is "
    "gone and clean metal is underneath. Keep the exact geometry, position and lighting of "
    "every bolt, nut, washer, plate and weld"
)

ELEMENT_EN = {"brace": "steel brace / bracing member",
              "column_base": "column base / steel column footing on a concrete pedestal"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--out-dir", default="outputs/gemini_synth/s1_counterfactual_negatives")
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--model", default="gemini-3-pro-image")
    p.add_argument("--limit", type=int, default=0, help="0 = all damaged images")
    p.add_argument("--stems", default="", help="comma separated stems, overrides --limit")
    p.add_argument("--max-boxes", type=int, default=6,
                   help="skip images with more boxes than this; they are usually wide scenes")
    p.add_argument("--cluster-gap", type=float, default=0.10,
                   help="merge boxes within this fraction of the long side")
    p.add_argument("--refine-min-severity", default="moderate",
                   choices=["minor", "moderate", "severe"],
                   help="refine rounds chase only regions at least this severe")
    p.add_argument("--refine-min-area", type=float, default=2e-4,
                   help="ignore refine regions smaller than this area fraction")
    p.add_argument("--max-total-boxes", type=int, default=8,
                   help="stop refining once this many regions have been repaired")
    p.add_argument("--context", type=float, default=2.4,
                   help="context window side = cluster side * this")
    p.add_argument("--min-window", type=int, default=768,
                   help="a tight window starves the model of context and it "
                        "returns a flat grey patch instead of a repair")
    p.add_argument("--max-window", type=int, default=1280,
                   help="a window much larger than the model's output is upscaled on "
                        "paste-back, which is what the QC judge reads as blurring")
    p.add_argument("--send-size", type=int, default=1024, help="longest side sent to the model")
    p.add_argument("--image-size", default="2K", choices=["1K", "2K", "4K"],
                   help="generate above the paste-back window and downscale")
    p.add_argument("--prescreen", action="store_true",
                   help="inventory unannotated damage with a vision model and repair it too")
    p.add_argument("--prescreen-model", default="gemini-3.1-pro-preview")
    p.add_argument("--prescreen-off-element", action="store_true",
                   help="also repair damage the inventory places off the inspected element")
    p.add_argument("--skip-pervasive", action="store_true", default=True,
                   help="skip images the inventory calls pervasively damaged")
    p.add_argument("--refine-rounds", type=int, default=1,
                   help="re-inventory the repaired image and repair what is left")
    p.add_argument("--candidates", type=int, default=3,
                   help="candidate edits sampled per cluster; the least invasive workable one wins")
    p.add_argument("--min-candidates", type=int, default=2,
                   help="keep sampling until this many, even once one is acceptable")
    p.add_argument("--min-core-diff", type=float, default=2.0,
                   help="below this the model returned the region unchanged")
    p.add_argument("--max-core-diff", type=float, default=55.0,
                   help="above this the region was re-imagined rather than repaired")
    p.add_argument("--references", type=int, default=3, help="real negative reference images")
    p.add_argument("--ref-size", type=int, default=768)
    p.add_argument("--dilate", type=float, default=0.12, help="box dilation before feathering")
    p.add_argument("--feather", type=float, default=0.06, help="feather width, fraction of crop")
    p.add_argument("--grain", type=float, default=1.0, help="0 disables grain matching")
    p.add_argument("--photometry", type=float, default=1.0,
                   help="pull edited-region tone back to the original; 0 disables")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--no-compare", action="store_true", help="skip side-by-side artifacts")
    return p.parse_args()


def pick_references(negatives: list[dict], stem: str, count: int,
                    rng_seed: int) -> list[Path]:
    """Deterministic per-stem reference choice, so a rerun reproduces the input."""
    if not negatives:
        return []
    rng = random.Random(f"{rng_seed}:{stem}")
    pool = [n["image"] for n in negatives]
    return rng.sample(pool, min(count, len(pool)))


def repair_image(job: dict, args: argparse.Namespace, api_key: str,
                 ref_paths: list[Path]) -> dict:
    stem = job["stem"]
    src_path: Path = job["image"]
    record: dict = {"job_id": stem, "stem": stem, "category": args.category,
                    "source": src_path.name, "ok": False, "clusters": [],
                    "started_at": sc.utc_now()}

    with Image.open(src_path) as im:
        original = im.convert("RGB")
    size = original.size

    def inventory_boxes(img: Image.Image, min_severity: str | None = None,
                        min_area: float = 0.0) -> tuple[list, dict | None, str | None]:
        inv, err = sc.inventory_damage(args.prescreen_model, img, args.category,
                                       api_key, send_size=args.send_size,
                                       timeout=args.timeout, max_retries=args.max_retries)
        if inv is None:
            return [], None, err
        found = sc.inventory_to_boxes(inv, size)
        rank = {"minor": 0, "moderate": 1, "severe": 2}
        floor = rank[min_severity] if min_severity else 0
        keep = []
        for b, region in zip(found, inv.get("regions", [])):
            if not (region.get("on_target_element") or args.prescreen_off_element):
                continue
            if rank.get(region.get("severity", "moderate"), 1) < floor:
                continue
            if b[3] * b[4] < min_area:
                continue
            keep.append(b)
        return keep, inv, None

    boxes = list(job["boxes"])
    record["n_annotated_boxes"] = len(boxes)

    if args.prescreen:
        keep, inv, inv_err = inventory_boxes(original)
        record["inventory_error"] = inv_err
        if inv is not None:
            record["inventory"] = inv
            if inv.get("damage_is_pervasive") and args.skip_pervasive:
                record.update({"skipped": "pervasive_damage", "finished_at": sc.utc_now()})
                return record
            record["n_inventory_boxes"] = len(keep)
            boxes = boxes + keep

    record["image_size"] = list(size)

    ref_parts = []
    for rp in ref_paths:
        try:
            with Image.open(rp) as rim:
                ref_parts.append(sc.encode_image(rim, max_side=args.ref_size))
        except Exception:  # noqa: BLE001
            continue
    record["references"] = [p.name for p in ref_paths]

    kinds_text = " ".join(
        r.get("kind", "") for r in (record.get("inventory") or {}).get("regions", []))
    steel_words = ("corro", "rust", "bolt", "plate", "steel", "weld", "anchor", "flange")
    conc_words = ("crack", "spall", "delamin", "concrete", "rebar", "honeycomb")
    lower = kinds_text.lower()
    is_steel = any(w in lower for w in steel_words)
    is_conc = any(w in lower for w in conc_words)
    if is_steel and not is_conc:
        hint = STEEL_HINT
    elif is_steel:
        hint = DAMAGE_HINT[args.category] + ". Additionally, " + STEEL_HINT
    else:
        hint = DAMAGE_HINT[args.category]
    record["damage_profile"] = ("steel" if is_steel and not is_conc
                                else "mixed" if is_steel else "concrete")

    instruction = REPAIR_INSTRUCTION.format(
        element_ja=sc.CATEGORY_JA[args.category],
        element_en=ELEMENT_EN[args.category],
        damage_hint=hint,
    )

    def repair_cluster(working: Image.Image, px_boxes, indices) -> tuple[Image.Image | None,
                                                                        dict]:
        """Repair one box cluster, choosing the least invasive workable candidate.

        Sampling more than one candidate is what keeps hallucination out: the
        2026-08-03 QC pass caught a single-sample edit that replaced a band of
        concrete with pipes and a blue tarpaulin. That failure is loud in
        `core_diff`, so scoring candidates on how little they changed - subject
        to changing enough to actually remove the damage - selects it away
        without another model call to adjudicate.
        """
        window = sc.context_window(px_boxes, indices, size, args.context,
                                   args.min_window, args.max_window)
        crop = working.crop(window)
        mask = sc.build_paste_mask(crop.size, px_boxes, indices, window,
                                   dilate_frac=args.dilate, feather_frac=args.feather)
        core = mask > 0.5
        crop_arr = np.asarray(crop, dtype=np.float32)

        best = None
        best_score = None
        last_err = None
        tried = []

        for attempt in range(args.candidates):
            parts = [sc.encode_image(crop, max_side=args.send_size)]
            parts.extend(ref_parts)
            text = instruction
            if attempt and tried and tried[-1] < args.min_core_diff:
                text += ("\n\nThe previous attempt returned the region unchanged. "
                         "The damage IS present and MUST be repaired this time.")
            parts.append({"text": text})

            edited, last_err = sc.generate_image(args.model, parts, api_key,
                                                 timeout=args.timeout, max_retries=1,
                                                 image_size=args.image_size,
                                                 aspect_ratio="1:1")
            if edited is None:
                continue
            candidate = sc.composite(crop, edited, mask)
            diff = 0.0
            if core.sum():
                diff = float(np.abs(crop_arr[core]
                                    - np.asarray(candidate, dtype=np.float32)[core]).mean())
            tried.append(diff)

            if diff < args.min_core_diff:
                score = 1e6 + (args.min_core_diff - diff)      # did nothing
            elif diff > args.max_core_diff:
                score = 1e5 + (diff - args.max_core_diff)      # re-imagined the region
            else:
                score = diff                                   # prefer the smallest real edit
            if best_score is None or score < best_score:
                best, best_score = candidate, score
            if best_score < 1e5 and attempt + 1 >= args.min_candidates:
                break

        if best is None:
            return None, {"ok": False, "error": last_err, "window": list(window)}

        if args.photometry > 0:
            best = sc.match_photometry(best, crop, mask, strength=args.photometry)
        if args.grain > 0:
            best = sc.match_texture(best, crop, mask, strength=args.grain)
        working.paste(best, (window[0], window[1]))
        return working, {"ok": True, "window": list(window),
                         "boxes": [int(i) for i in indices],
                         "core_diff": tried[-1] if tried else 0.0,
                         "candidate_diffs": tried,
                         "accepted": bool(best_score is not None and best_score < 1e5),
                         "mask_coverage": float(core.mean())}

    working = original.copy()
    all_boxes: list = []
    rounds: list[dict] = []

    for rnd in range(args.refine_rounds + 1):
        if not boxes:
            break
        offset = len(all_boxes)
        all_boxes.extend(boxes)
        px_boxes = sc.boxes_to_pixels(all_boxes, size)
        idx_this_round = list(range(offset, len(all_boxes)))
        clusters = [c for c in sc.cluster_boxes(px_boxes, size, gap_frac=args.cluster_gap)
                    if any(i in idx_this_round for i in c)]

        round_rec = {"round": rnd, "n_boxes": len(boxes), "n_clusters": len(clusters),
                     "clusters": []}
        for indices in clusters:
            working, cl = repair_cluster(working, px_boxes, indices)
            round_rec["clusters"].append(cl)
            record["clusters"].append({**cl, "round": rnd})
            if working is None:
                record["error"] = cl.get("error")
                record["rounds"] = rounds + [round_rec]
                record["finished_at"] = sc.utc_now()
                return record
        rounds.append(round_rec)

        # Re-inventory the repaired image. Residual damage was the largest single
        # QC rejection reason, and it is mostly fine cracks the first inventory
        # missed rather than boxes the repair failed on.
        if rnd < args.refine_rounds:
            if len(all_boxes) >= args.max_total_boxes:
                boxes = []
            else:
                boxes, _, _ = inventory_boxes(
                    working, min_severity=args.refine_min_severity,
                    min_area=args.refine_min_area)
            round_rec["residual_found"] = len(boxes)
        else:
            boxes = []

    record["rounds"] = rounds
    record["n_boxes"] = len(all_boxes)
    record["n_clusters"] = sum(r["n_clusters"] for r in rounds)
    # cluster entries index into this list, not into the delivered labels, so QC
    # must rebuild the edit mask from exactly what was repaired
    record["boxes_used"] = [[float(v) for v in b] for b in all_boxes]

    out_root = Path(args.out_dir) / args.category
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    out_img = out_root / "images" / f"{stem}_s1neg.jpg"
    working.save(out_img, quality=95, subsampling=0)
    (out_root / "labels" / f"{stem}_s1neg.txt").write_text("")

    if not args.no_compare:
        (out_root / "compare").mkdir(parents=True, exist_ok=True)
        marked = original.copy()
        d = ImageDraw.Draw(marked)
        lw = max(2, min(size) // 250)
        for _, (x0, y0, x1, y1) in px_boxes:
            d.rectangle([x0, y0, x1, y1], outline=(0, 255, 120), width=lw)
        W, H = size
        sheet = Image.new("RGB", (W * 2 + 16, H), (18, 18, 18))
        sheet.paste(marked, (0, 0))
        sheet.paste(working, (W + 16, 0))
        sheet.thumbnail((2000, 2000), Image.LANCZOS)
        sheet.save(out_root / "compare" / f"{stem}_compare.jpg", quality=90)

    record.update({"ok": True, "output_image": str(out_img),
                   "output_label": str(out_root / "labels" / f"{stem}_s1neg.txt"),
                   "finished_at": sc.utc_now()})
    return record


def main() -> int:
    args = parse_args()
    api_key = sc.require_api_key()

    data = sc.load_category(Path(args.paired_dir), args.category)
    damaged, negatives = data["damaged"], data["negatives"]

    if args.stems:
        wanted = {s.strip() for s in args.stems.split(",") if s.strip()}
        jobs = [j for j in damaged if j["stem"] in wanted]
    else:
        jobs = [j for j in damaged if len(j["boxes"]) <= args.max_boxes]
        if args.limit:
            rng = random.Random(args.seed)
            jobs = rng.sample(jobs, min(args.limit, len(jobs)))

    out_root = Path(args.out_dir) / args.category
    results_path = out_root / "generation_results.jsonl"
    done = sc.load_done(results_path)
    jobs = [j for j in jobs if j["stem"] not in done]

    sc.log(f"category={args.category} model={args.model}")
    sc.log(f"damaged={len(damaged)} negatives={len(negatives)} "
           f"queued={len(jobs)} already_done={len(done)}")
    if not jobs:
        sc.log("nothing to do")
        return 0

    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(repair_image, job, args, api_key,
                      pick_references(negatives, job["stem"], args.references, args.seed)): job
            for job in jobs
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            job = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"job_id": job["stem"], "ok": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            sc.append_jsonl(results_path, rec)
            if rec.get("skipped"):
                sc.log(f"[{i}/{len(jobs)}] skip {rec.get('job_id')}: {rec['skipped']}")
                continue
            if rec.get("ok"):
                ok += 1
                sc.log(f"[{i}/{len(jobs)}] ok   {rec['stem']} "
                       f"boxes={rec.get('n_boxes')} clusters={rec.get('n_clusters')}")
            else:
                sc.log(f"[{i}/{len(jobs)}] FAIL {rec.get('job_id')}: "
                       f"{str(rec.get('error'))[:160]}")

    sc.log(f"\n{ok}/{len(jobs)} ok -> {out_root}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
