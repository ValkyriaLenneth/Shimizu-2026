#!/usr/bin/env python3
"""Quality gate for S1 counterfactual negatives.

A synthetic negative is only useful if two things hold, and they fail in
opposite directions:

* **Residual damage.** If damage survives the repair, the image is labelled
  empty but still contains damage, and training on it teaches suppression of
  genuine damage - the same poison identified in the 141 untriaged empty-label
  images.
* **Detectable editing.** If the repaired region reads as cleaner, smoother or
  flatter than the rest of the photograph, the model can separate synthetic
  negatives from real positives on that cue alone. That replaces the element
  shortcut with an edit-artefact shortcut and is strictly worse, because it
  would look like a large gain on any test set built the same way.

So the gate runs two independent passes.

1. **Local pixel metrics**, no API. Confirms the untouched region really is
   untouched, and compares the edited region against a ring of surrounding
   original pixels on brightness, saturation, colour and high-frequency
   energy. Over-cleaning shows up as the edited region being brighter, less
   saturated and markedly smoother than its own surroundings.

2. **Vision judge**, Gemini. Two questions per image: is any structural damage
   visible in the repaired image, and - shown the original beside it - was the
   only change the removal of damage.

Verdicts: `pass`, `review`, `reject`. Only `pass` images should be built into a
training view.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import synth_common as sc

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "damage_removed": {"type": "boolean"},
        "residual_at_repair_site": {"type": "string",
                                    "enum": ["none", "minor", "moderate", "severe"]},
        "residual_description": {"type": "string"},
        # Added 2026-08-04. The two questions above are both satisfied by an edit
        # that REBUILDS the member instead of repairing it, which is the dominant
        # failure mode on steel: e-00085 had a severed brace replaced by a clean
        # bolted connection and the judge passed it, describing the replacement
        # approvingly. A rebuilt member is worse than a failed repair, because the
        # image enters training labelled "no damage" while showing a component that
        # does not exist in the real corpus.
        "structure_preserved": {"type": "boolean"},
        "structure_change_description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["damage_removed", "residual_at_repair_site",
                 "residual_description", "structure_preserved",
                 "structure_change_description", "confidence"],
}

# Asked WITHOUT the original alongside. Shown a pair, the judge reports "looks
# edited" from the mere existence of a difference: on 2026-08-03 a reversed-pair
# control fired at 5/6 in BOTH directions, so it flagged the genuine photograph
# as edited exactly as often as the synthetic one. Edit detection therefore has
# to be a single-image question.
AUTH_SCHEMA = {
    "type": "object",
    "properties": {
        "looks_edited": {"type": "boolean"},
        "edit_evidence": {"type": "string"},
        "authentic_survey_photo": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["looks_edited", "edit_evidence",
                 "authentic_survey_photo", "confidence"],
}

AUTH_PROMPT = """This is a region cut from a Japanese building damage survey photograph of
{element_ja}, taken hand-held in the field.

Judge this single image on its own. You have no reference to compare against.

1. looks_edited: does any region betray digital editing or generated content? Look for a
   patch that is smoother, flatter, blurrier or more uniform than its surroundings; a
   visible seam or boundary; repeated or implausible texture; an object cut in half or
   partially erased; geometry that could not exist; or content that does not belong in
   the scene.
   These are NORMAL in this kind of photograph and are NOT editing: motion blur, soft
   focus, harsh flash, blown highlights, deep shadow, JPEG blocking, low resolution,
   sensor noise, dirt, stains, rust, paint marks and clutter.

2. authentic_survey_photo: would this pass as a genuine hand-held field inspection
   photograph?

confidence: 0.0-1.0."""

# Calibrated on 2026-08-03 against 16 real negatives and 16 real positives, all
# unedited and passed as both images:
#   looks_edited            0% false positive   -> trusted
#   authentic_survey_photo  0% false positive   -> trusted
#   only_change_is_...     69-100% false pos    -> question removed
#   damage_visible         69% of client-clean negatives flagged -> unusable as an
#                          absolute gate, since the real negatives this corpus
#                          already trains on would not pass it either
# So the damage question is asked comparatively about the repaired sites only.
JUDGE_PROMPT = """You are auditing synthetic training data for a building damage detection model.

IMAGE 1 is a region cut from an original Japanese building damage survey photograph of
{element_ja}. It contained structural damage.
IMAGE 2 is the same region after an automated repair intended to remove that damage while
leaving everything else untouched. Both are shown at native resolution, so fine cracks
and surface texture are visible.

1. damage_removed: comparing the two images, has the structural damage that was visible
   in IMAGE 1 been removed in IMAGE 2? Judge ONLY the places that changed between the two
   images. Answer false if IMAGE 2 is identical to IMAGE 1.

2. residual_at_repair_site: how much damage is still visible AT THOSE REPAIRED PLACES in
   IMAGE 2 - "none", "minor", "moderate" or "severe"?
   Judge only the repaired places. Do NOT count damage elsewhere in the frame, damage on
   background pipes, walls, floors or neighbouring structures, or anything that is
   unchanged between the two images.
   Ageing is NOT damage: rust staining without material loss, water stains, efflorescence,
   dirt, grime, mould, faded or chalky paint, discolouration, survey markings, chalk lines,
   handwriting, form-tie holes, joint lines, chamfers and casting seams.

3. structure_preserved: is IMAGE 2 the SAME PHYSICAL OBJECT as IMAGE 1, only in a
   sound condition? A valid repair changes surface state only - a crack is filled,
   rust is cleaned off, a spall is made good. Answer FALSE if the member itself was
   rebuilt or substituted in any way, for example:
   - a bolt, nut, gusset plate, splice plate, weld or bracket appears that was not
     in IMAGE 1, or one present in IMAGE 1 disappears
   - the outline, width, cross-section or edge profile of the member changes
   - a damaged or severed member is shown as a new, differently-shaped component
   - a rod, pipe or bar changes into an object of another shape
   Answer TRUE only if every structural element keeps its original shape, position
   and count, and the only difference is the condition of the surface.
   Be strict: when the geometry of a member looks different, answer false even if
   the result looks like a plausible undamaged structure. A convincingly rebuilt
   member is a WORSE outcome than a visibly failed repair.

structure_change_description: if structure_preserved is false, name the component that
   was added, removed or reshaped. Empty string otherwise.

confidence: 0.0-1.0 in your overall assessment."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synth-dir", default="outputs/gemini_synth/s1_counterfactual_negatives")
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--send-size", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-judge", action="store_true", help="local metrics only")
    p.add_argument("--max-judge-windows", type=int, default=4,
                   help="repaired windows judged per image")
    # thresholds
    p.add_argument("--max-outside-diff", type=float, default=0.6,
                   help="mean abs pixel diff allowed outside the edit mask")
    p.add_argument("--max-smoothness-ratio", type=float, default=0.55,
                   help="reject if edited HF energy / ring HF energy falls below this")
    p.add_argument("--min-smoothness-ratio", type=float, default=1.9,
                   help="reject above this: invented texture, not a repair")
    p.add_argument("--max-core-diff", type=float, default=55.0,
                   help="reject above this: the region was re-imagined, not repaired")
    p.add_argument("--max-brightness-shift", type=float, default=26.0)
    p.add_argument("--max-saturation-drop", type=float, default=22.0)
    return p.parse_args()


# --------------------------------------------------------------------------
# local metrics
# --------------------------------------------------------------------------

def rebuild_mask(record: dict, size: tuple[int, int], paired_boxes,
                 dilate: float = 0.12, feather: float = 0.06) -> np.ndarray:
    """Reconstruct the union edit mask in full-image coordinates.

    Prefer `boxes_used` from the generation record: with `--prescreen` the boxes
    actually repaired include unannotated damage found by the vision inventory,
    and cluster entries index into that combined list rather than the delivered
    labels.
    """
    W, H = size
    full = np.zeros((H, W), dtype=np.float32)
    boxes = record.get("boxes_used")
    if boxes:
        boxes = [(int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes]
    else:
        boxes = paired_boxes
    px_boxes = sc.boxes_to_pixels(boxes, size)
    for cl in record.get("clusters", []):
        if not cl.get("ok"):
            continue
        win = cl["window"]
        crop_size = (win[2] - win[0], win[3] - win[1])
        m = sc.build_paste_mask(crop_size, px_boxes, cl["boxes"], win,
                                dilate_frac=dilate, feather_frac=feather)
        full[win[1]:win[3], win[0]:win[2]] = np.maximum(
            full[win[1]:win[3], win[0]:win[2]], m)
    return full


def local_metrics(original: Image.Image, repaired: Image.Image,
                  mask: np.ndarray) -> dict:
    # The repaired file is JPEG; comparing it against the original file measures
    # codec noise as if it were an edit. Put the original through the same
    # encoder first so the difference isolates what the model actually changed.
    o = np.asarray(sc.jpeg_roundtrip(original), dtype=np.float32)
    r = np.asarray(repaired.convert("RGB"), dtype=np.float32)
    if o.shape != r.shape:
        return {"error": f"shape mismatch {o.shape} vs {r.shape}"}

    core = mask > 0.5
    outside = mask <= 0.01
    ring = (cv2.dilate((mask > 0.15).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & outside

    out: dict = {"edited_frac": float(core.mean())}
    if outside.sum() > 0:
        out["outside_mean_abs_diff"] = float(np.abs(o[outside] - r[outside]).mean())
    if core.sum() < 64 or ring.sum() < 256:
        out["insufficient_area"] = True
        return out

    def hf_energy(a, sel):
        g = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = g - cv2.GaussianBlur(g, (0, 0), 1.2)
        return float(h[sel].std())

    r_hsv = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    hf_core, hf_ring = hf_energy(r, core), hf_energy(r, ring)
    out["hf_core"] = hf_core
    out["hf_ring"] = hf_ring
    out["smoothness_ratio"] = hf_core / hf_ring if hf_ring > 1e-6 else 0.0

    out["brightness_core"] = float(r_hsv[..., 2][core].mean())
    out["brightness_ring"] = float(r_hsv[..., 2][ring].mean())
    out["brightness_shift"] = out["brightness_core"] - out["brightness_ring"]

    out["saturation_core"] = float(r_hsv[..., 1][core].mean())
    out["saturation_ring"] = float(r_hsv[..., 1][ring].mean())
    out["saturation_shift"] = out["saturation_core"] - out["saturation_ring"]

    # how much actually changed inside the mask - a near-zero edit means the
    # model returned the region unchanged and no repair happened
    out["core_mean_abs_diff"] = float(np.abs(o[core] - r[core]).mean())
    return out


def grade_local(m: dict, args) -> tuple[str, list[str]]:
    flags: list[str] = []
    if m.get("error"):
        return "reject", [m["error"]]
    if m.get("outside_mean_abs_diff", 0.0) > args.max_outside_diff:
        flags.append(f"outside_changed({m['outside_mean_abs_diff']:.2f})")
    if m.get("insufficient_area"):
        flags.append("insufficient_area")
        return "review", flags
    if m.get("core_mean_abs_diff", 99) < 2.0:
        flags.append(f"no_effective_edit({m['core_mean_abs_diff']:.2f})")
    # An edit that rewrites most of the region is a hallucination, not a repair:
    # f-00048 on 2026-08-03 replaced a band of concrete with pipes and a blue
    # tarpaulin at core_diff 81 and smoothness 2.14.
    if m.get("core_mean_abs_diff", 0) > args.max_core_diff:
        flags.append(f"hallucinated({m['core_mean_abs_diff']:.1f})")
    if m.get("smoothness_ratio", 1.0) > args.min_smoothness_ratio:
        flags.append(f"invented_texture({m['smoothness_ratio']:.2f})")
    if m.get("smoothness_ratio", 1.0) < args.max_smoothness_ratio:
        flags.append(f"over_smooth({m['smoothness_ratio']:.2f})")
    if m.get("brightness_shift", 0.0) > args.max_brightness_shift:
        flags.append(f"brighter({m['brightness_shift']:.1f})")
    if -m.get("saturation_shift", 0.0) > args.max_saturation_drop:
        flags.append(f"desaturated({m['saturation_shift']:.1f})")

    hard = [f for f in flags if f.startswith(
        ("outside_changed", "no_effective_edit", "hallucinated", "invented_texture"))]
    if hard:
        return "reject", flags
    return ("review" if flags else "pass"), flags


# --------------------------------------------------------------------------
# vision judge
# --------------------------------------------------------------------------

SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def judge(record: dict, args, api_key: str, original: Image.Image,
          repaired: Image.Image) -> tuple[dict | None, str | None]:
    """Judge each repaired window as a before/after crop pair, then aggregate.

    Judging the whole frame downscaled to 1024 made the instrument blind: on
    2026-08-03 two images whose repairs were real (core_diff 4.2 and 14.2) came
    back as "identical to the original", because a hairline crack in a 2816px
    photograph does not survive the downscale in EITHER image. Cropping to the
    repaired window keeps the evidence at native resolution.
    """
    windows = [cl["window"] for cl in record.get("clusters", []) if cl.get("ok")]
    if not windows:
        return None, "no repaired windows in record"

    per_window, errors = [], []
    for win in windows[: args.max_judge_windows]:
        parts = [
            sc.encode_image(original.crop(tuple(win)), max_side=args.send_size),
            sc.encode_image(repaired.crop(tuple(win)), max_side=args.send_size),
            {"text": JUDGE_PROMPT.format(element_ja=sc.CATEGORY_JA[args.category])},
        ]
        j, err = sc.generate_text(args.model, parts, api_key, timeout=args.timeout,
                                  max_retries=args.max_retries,
                                  response_schema=JUDGE_SCHEMA)
        if j is None:
            errors.append(err)
            continue
        a, aerr = sc.generate_text(
            args.model,
            [sc.encode_image(repaired.crop(tuple(win)), max_side=args.send_size),
             {"text": AUTH_PROMPT.format(element_ja=sc.CATEGORY_JA[args.category])}],
            api_key, timeout=args.timeout, max_retries=args.max_retries,
            response_schema=AUTH_SCHEMA)
        if aerr:
            errors.append(aerr)
        per_window.append({**j, **(a or {}), "window": list(win)})

    if not per_window:
        return None, "; ".join(str(e) for e in errors)[:300]

    worst = max(per_window,
                key=lambda j: SEVERITY_RANK.get(j.get("residual_at_repair_site", "none"), 0))
    agg = {
        "damage_removed": all(j.get("damage_removed") for j in per_window),
        "residual_at_repair_site": worst.get("residual_at_repair_site", "none"),
        "residual_description": worst.get("residual_description", ""),
        "looks_edited": any(j.get("looks_edited") for j in per_window),
        "edit_evidence": next((j.get("edit_evidence") for j in per_window
                               if j.get("looks_edited")), ""),
        "authentic_survey_photo": all(j.get("authentic_survey_photo", True)
                                      for j in per_window),
        "confidence": min(j.get("confidence", 0.0) for j in per_window),
        "n_windows": len(per_window),
        "per_window": per_window,
    }
    return agg, ("; ".join(str(e) for e in errors)[:200] if errors else None)


def combine(local_verdict: str, flags: list[str], j: dict | None) -> tuple[str, list[str]]:
    verdict, reasons = local_verdict, list(flags)
    if j is None:
        return ("review" if verdict == "pass" else verdict), reasons + ["judge_unavailable"]
    if not j.get("damage_removed", True):
        verdict = "reject"
        reasons.append("not_repaired")
    residual = j.get("residual_at_repair_site", "none")
    if residual in ("moderate", "severe"):
        verdict = "reject"
        reasons.append(f"residual_{residual}")
    elif residual == "minor":
        verdict = "reject" if verdict == "reject" else "review"
        reasons.append("residual_minor")
    # A rebuilt member is a hard reject, not a review: it enters training as a
    # negative while showing structure the real corpus never contains, so the model
    # would learn the fabricated component as evidence of soundness.
    if j.get("structure_preserved") is False:
        verdict = "reject"
        reasons.append("geometry_changed")
    if j.get("looks_edited"):
        verdict = "reject" if verdict == "reject" else "review"
        reasons.append("looks_edited")
    if not j.get("authentic_survey_photo", True):
        verdict = "reject" if verdict == "reject" else "review"
        reasons.append("inauthentic")
    return verdict, reasons


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    api_key = None if args.skip_judge else sc.require_api_key()

    synth_root = Path(args.synth_dir) / args.category
    results_path = synth_root / "generation_results.jsonl"
    if not results_path.exists():
        print(f"no generation results at {results_path}")
        return 1

    paired = sc.load_category(Path(args.paired_dir), args.category)
    boxes_by_stem = {j["stem"]: j["boxes"] for j in paired["damaged"]}
    image_by_stem = {j["stem"]: j["image"] for j in paired["damaged"]}

    records = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                records.append(row)
    seen, uniq = set(), []
    for r in reversed(records):
        if r["stem"] not in seen:
            seen.add(r["stem"])
            uniq.append(r)
    records = list(reversed(uniq))
    if args.limit:
        records = records[: args.limit]

    print(f"qc {len(records)} images  category={args.category} "
          f"judge={'off' if args.skip_judge else args.model}")

    def run_one(rec: dict) -> dict:
        stem = rec["stem"]
        out: dict = {"stem": stem}
        try:
            with Image.open(image_by_stem[stem]) as im:
                original = im.convert("RGB")
            with Image.open(rec["output_image"]) as im:
                repaired = im.convert("RGB")
            mask = rebuild_mask(rec, original.size, boxes_by_stem[stem])
            m = local_metrics(original, repaired, mask)
            lv, flags = grade_local(m, args)
            out["metrics"] = m
            out["local_verdict"] = lv
            out["flags"] = flags
            if args.skip_judge:
                out["verdict"], out["reasons"] = lv, flags
                return out
            j, err = judge(rec, args, api_key, original, repaired)
            out["judge"] = j
            out["judge_error"] = err
            out["verdict"], out["reasons"] = combine(lv, flags, j)
        except Exception as exc:  # noqa: BLE001
            out["verdict"] = "reject"
            out["reasons"] = [f"{type(exc).__name__}: {exc}"]
        return out

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(run_one, records):
            rows.append(r)
            j = r.get("judge") or {}
            extra = ""
            if j:
                extra = (f" removed={j.get('damage_removed')} "
                         f"residual={j.get('residual_at_repair_site')} "
                         f"edited={j.get('looks_edited')}")
            print(f"  {r['verdict']:6s} {r['stem']:10s} {','.join(r.get('reasons', [])) or '-'}{extra}")

    qc_path = synth_root / "qc_results.json"
    qc_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n--- verdicts ---")
    for k in ("pass", "review", "reject"):
        if k in counts:
            print(f"  {k:6s} {counts[k]:4d}  ({counts[k]/len(rows):.0%})")
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in r.get("reasons", []):
            key = reason.split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("--- reasons ---")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {k:24s} {v}")
    print(f"\nwrote {qc_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
