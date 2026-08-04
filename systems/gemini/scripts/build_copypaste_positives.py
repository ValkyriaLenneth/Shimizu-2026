#!/usr/bin/env python3
"""S4 - synthetic positives by pasting REAL damage onto REAL sound elements.

Why this and not generation, after 2026-08-04
---------------------------------------------
Every generative route tried on this project failed for the same two reasons,
and copy-paste is immune to both.

**Geometry hallucination.** On steel, `gemini-3-pro-image` does not remove damage,
it replaces the member: a severed brace came back as a clean bolted splice, a
bracing crossing grew a solid gusset plate, a corroded rod became a differently
shaped part. The whole-frame geometry check found this in 45% of column_base and
more than half of brace outputs. Pasted pixels are photographed, so there is
nothing to hallucinate.

**Round-trip mismatch.** The 2026-08-03 plan warned that if only the synthetic
side passes through the model's re-encoding, "smooth repaired texture" becomes a
new shortcut. The 80-epoch dual-arm run on 2026-08-04 measured the cost of
ignoring it: replacing 19 of 82 real negatives with QC-clean, geometry-verified
S1 negatives dropped the arm below any P>=0.60 operating point at all, while the
control reached R=0.444. Copy-paste writes real pixels into real photographs and
applies the identical compositing path to both arms, so the low-level statistics
of positives and negatives stay matched.

What it attacks
---------------
The binding constraint measured on 2026-08-04 is data volume, not capacity
(RFDETRSmall was worse than medium) and not perception (train recall is 0.995 on
brace, with zero misses in the smallest quartile). The corpus has 179/235
training images and every one of them contains damage, so "element present" and
"damage present" remain perfectly correlated.

Pasting a real damage crop onto a real *sound* element image produces a positive
whose scene is a scene the corpus previously only ever saw as a negative. The
same sound image, left alone, is still a negative. That is a counterfactual pair
made entirely of photographed pixels.

Where to paste
--------------
Damage sits on the element, not on sky, floor or wall, so paste sites are taken
from the baseline model's own detections on the sound images. That model is an
element detector - that is precisely the shortcut it learned - so its boxes
localise the element for free. This reuses the `negcrop` insight from
2026-07-26 without repeating its mistake: there, crops were fed as negatives
while positives stayed whole images, so framing itself became discriminative.
Here nothing is cropped; full frames are pasted into and stay full frames.

Scale convention
----------------
Box area is close to a proxy for grade (2026-08-03, section 1.1): brace medians
run 0.87% / 6.93% / 16.4% for B/C/D, column_base 2.14% / 9.54% / 5.85%. A pasted
crop is therefore rescaled to a sample drawn from the source grade's own area
distribution, so the label stays true to the convention even though the pixels
moved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import synth_common as sc

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADES = {0: "B", 1: "C", 2: "D"}
GRADE_IDS = {v: k for k, v in GRADES.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--crops-dir", default="outputs/gemini_synth/grade_references/crops")
    p.add_argument("--audit-json", default="outputs/rfdetr_new_classes/empty_label_audit")
    p.add_argument("--split-dir", default=".local_artifacts/handoff_20260726/split")
    p.add_argument("--out-dir", default="outputs/gemini_synth/s4_copypaste")
    p.add_argument("--per-image", type=int, default=2, help="pasted damages per sound image")
    p.add_argument("--variants", type=int, default=2, help="synthetic images per sound image")
    p.add_argument("--min-site-score", type=float, default=0.20,
                   help="minimum baseline detection score for a paste site")
    p.add_argument("--feather", type=float, default=0.12, help="feather width as fraction of crop")
    p.add_argument("--photometry", type=float, default=1.0, help="0 disables median matching")
    p.add_argument("--grades", default="B,C,D", help="which grades to paste")
    # Symmetric mode. S1 failed on 2026-08-04 because only the synthetic side went
    # through Gemini's re-encoding, so "repaired texture" was separable from the
    # real positives by low-level statistics alone - exactly what the 2026-08-03
    # plan warned about. Building the negatives by the SAME paste operation, using
    # sound crops taken from real photographs, puts both arms on one path: every
    # image in the synthetic set has had a rectangle Poisson-blended into it, so
    # the blend itself carries no label information.
    p.add_argument("--mode", default="positives", choices=["positives", "negatives"],
                   help="positives: paste damage onto sound images. "
                        "negatives: paste sound element crops over the damage on real positives.")
    p.add_argument("--site-tries", type=int, default=8,
                   help="candidate paste sites scored by colour match; best is kept")
    p.add_argument("--no-seamless", dest="seamless", action="store_false", default=True,
                   help="disable Poisson blending, use feathered alpha")
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_sites(audit_dir: Path, category: str, min_score: float) -> dict[str, list]:
    """Element locations on the sound images, from the baseline model's detections."""
    path = Path(audit_dir) / f"{category}_audit.json"
    if not path.exists():
        return {}
    audit = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for rec in audit["records"]:
        dets = [d for d in rec.get("detections", []) if d["score"] >= min_score]
        if dets:
            out[rec["stem"]] = dets
    return out


def grade_area_pool(paired_dir: Path, category: str) -> dict[str, list[float]]:
    """Observed box-area fractions per grade, so pasted scale matches the convention."""
    pool: dict[str, list[float]] = defaultdict(list)
    for lab in sorted((paired_dir / category / "labels").glob("*.txt")):
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            if cls in GRADES:
                pool[GRADES[cls]].append(w * h)
    return pool


def site_cost(base: Image.Image, crop: Image.Image, cx: float, cy: float,
              area_frac: float) -> float:
    """Colour distance between a candidate paste site and the crop.

    A concrete spall dropped onto white-coated steel reads as a rectangle of the
    wrong colour, and a detector will happily learn "rectangular patch" instead of
    "damage" - trading one shortcut for another. Preferring the site whose local
    colour statistics are closest to the crop's keeps the paste plausible and,
    more importantly, keeps the pasted region from being separable from its
    surroundings by colour alone.
    """
    W, H = base.size
    target_area = area_frac * W * H
    ar = crop.width / max(1, crop.height)
    ph = max(8, int(round((target_area / max(ar, 1e-6)) ** 0.5)))
    pw = max(8, int(round(ph * ar)))
    x1 = max(0, min(W - pw, int(round(cx * W - pw / 2))))
    y1 = max(0, min(H - ph, int(round(cy * H - ph / 2))))
    region = np.asarray(base.crop((x1, y1, x1 + pw, y1 + ph)).resize((32, 32)), dtype=np.float32)
    src = np.asarray(crop.resize((32, 32)), dtype=np.float32)
    return float(np.abs(region.mean(axis=(0, 1)) - src.mean(axis=(0, 1))).sum()
                 + 0.5 * np.abs(region.std(axis=(0, 1)) - src.std(axis=(0, 1))).sum())


def paste_one(base: Image.Image, crop: Image.Image, cx: float, cy: float,
              area_frac: float, feather: float, photometry: float,
              rng: random.Random, seamless: bool = True
              ) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    """Composite *crop* into *base* centred at (cx, cy), scaled to *area_frac*."""
    W, H = base.size
    target_area = area_frac * W * H
    ar = crop.width / max(1, crop.height)
    ph = int(round((target_area / max(ar, 1e-6)) ** 0.5))
    pw = int(round(ph * ar))
    if pw < 12 or ph < 12 or pw > W * 0.9 or ph > H * 0.9:
        return None
    crop = crop.resize((pw, ph), Image.LANCZOS)

    x1 = int(round(cx * W - pw / 2))
    y1 = int(round(cy * H - ph / 2))
    x1 = max(0, min(W - pw, x1))
    y1 = max(0, min(H - ph, y1))
    x2, y2 = x1 + pw, y1 + ph

    region = base.crop((x1, y1, x2, y2))
    src = crop.convert("RGB")

    if photometry > 0:
        # Match the crop's per-channel median to the destination region so a crop
        # lifted from a differently-lit photograph does not read as a bright patch.
        a = np.asarray(src, dtype=np.float32)
        b = np.asarray(region, dtype=np.float32)
        for c in range(3):
            shift = float(np.median(b[..., c]) - np.median(a[..., c])) * photometry
            a[..., c] = np.clip(a[..., c] + shift, 0, 255)
        src = Image.fromarray(a.astype(np.uint8))

    out = base.copy()
    done = False
    if seamless:
        # Poisson blending solves for pixels whose gradients match the crop while
        # the boundary matches the destination, so the seam carries no step edge
        # at all. A feathered alpha only fades the step; it does not remove it,
        # and a detector can still key on the residual ring.
        try:
            import cv2

            dst = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR)
            patch = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
            mask = np.zeros((ph, pw), dtype=np.uint8)
            m = max(2, int(round(min(pw, ph) * 0.06)))
            mask[m:ph - m, m:pw - m] = 255
            if mask.any():
                centre = (x1 + pw // 2, y1 + ph // 2)
                mixed = cv2.seamlessClone(patch, dst, mask, centre, cv2.NORMAL_CLONE)
                out = Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
                done = True
        except Exception:
            done = False

    if not done:
        # Feathered alpha fallback: a hard edge is a high-frequency artefact the
        # detector can latch onto, which would replace one shortcut with another.
        fw = max(1, int(round(min(pw, ph) * feather)))
        alpha = np.ones((ph, pw), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, fw, dtype=np.float32)
        alpha[:fw, :] *= ramp[:, None]
        alpha[-fw:, :] *= ramp[::-1, None]
        alpha[:, :fw] *= ramp[None, :]
        alpha[:, -fw:] *= ramp[None, ::-1]
        blended = (np.asarray(src, dtype=np.float32) * alpha[..., None]
                   + np.asarray(region, dtype=np.float32) * (1 - alpha[..., None]))
        out.paste(Image.fromarray(blended.astype(np.uint8)), (x1, y1))

    box = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, pw / W, ph / H)
    return out, box


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.mode == "negatives":
        return build_negatives(args, rng)
    paired = Path(args.paired_dir)
    cat = args.category

    train_stems: set[str] | None = None
    sp = Path(args.split_dir) / f"{cat}_split.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        tr = d["splits"]["train"]
        train_stems = set(tr.keys()) if isinstance(tr, dict) else {
            x["stem"] if isinstance(x, dict) else x for x in tr}
        train_stems |= set(d.get("train_negatives", []))

    # Sound images = the zero-box images. They are the paste canvases.
    sound = []
    for lab in sorted((paired / cat / "labels").glob("*.txt")):
        if lab.read_text().strip():
            continue
        if train_stems is not None and lab.stem not in train_stems:
            continue
        img = sc.find_image(paired / cat / "images", lab.stem)
        if img is not None:
            sound.append((lab.stem, img))
    if args.limit:
        sound = sound[:args.limit]

    sites = load_sites(Path(args.audit_json), cat, args.min_site_score)
    areas = grade_area_pool(paired, cat)
    wanted = [g.strip() for g in args.grades.split(",") if g.strip()]
    crops: dict[str, list[Path]] = {}
    for g in wanted:
        d = Path(args.crops_dir) / cat / g
        crops[g] = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []

    print(f"category={cat}  sound images={len(sound)}  sites available={len(sites)}")
    print(f"crops per grade: " + ", ".join(f"{g}={len(crops[g])}" for g in wanted))
    print(f"area medians: " + ", ".join(
        f"{g}={np.median(areas[g]):.4f}" for g in wanted if areas.get(g)))
    if not sites:
        print("  NOTE: no audit detections found; falling back to random sites on the frame")

    out_root = Path(args.out_dir) / cat
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    made, records = 0, []
    grade_counter: Counter[str] = Counter()
    for stem, img_path in sound:
        with Image.open(img_path) as h:
            base0 = h.convert("RGB")
        det = sites.get(stem, [])
        for v in range(args.variants):
            base = base0.copy()
            boxes = []
            for k in range(args.per_image):
                g = rng.choice([g for g in wanted if crops[g]])
                cpath = rng.choice(crops[g])
                with Image.open(cpath) as ch:
                    crop = ch.convert("RGB")
                pool = areas.get(g) or [0.02]
                area_frac = rng.choice(pool)
                # Propose several sites on the element and keep the one whose local
                # colour statistics are closest to the crop, so the paste cannot be
                # separated from its surroundings by colour alone.
                cands = []
                for _ in range(args.site_tries):
                    if det:
                        d = rng.choice(det)
                        bx1, by1, bx2, by2 = d["box"]
                        ccx = (bx1 + bx2) / 2 / base.width
                        ccy = (by1 + by2) / 2 / base.height
                        ccx += rng.uniform(-0.25, 0.25) * (bx2 - bx1) / base.width
                        ccy += rng.uniform(-0.25, 0.25) * (by2 - by1) / base.height
                    else:
                        ccx, ccy = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
                    ccx, ccy = min(max(ccx, 0.05), 0.95), min(max(ccy, 0.05), 0.95)
                    cands.append((site_cost(base, crop, ccx, ccy, area_frac), ccx, ccy))
                _, cx, cy = min(cands)
                res = paste_one(base, crop, cx, cy, area_frac, args.feather,
                                args.photometry, rng, seamless=args.seamless)
                if res is None:
                    continue
                base, box = res
                boxes.append((GRADE_IDS[g], box))
                grade_counter[g] += 1
            if not boxes:
                continue
            name = f"{stem}_cp{v}"
            base.save(out_root / "images" / f"{name}.jpg", quality=95, subsampling=0)
            (out_root / "labels" / f"{name}.txt").write_text(
                "\n".join(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in boxes),
                encoding="utf-8")
            records.append({"stem": name, "source_sound_image": stem,
                            "n_boxes": len(boxes),
                            "grades": [GRADES[c] for c, _ in boxes],
                            "used_element_sites": bool(det)})
            made += 1

    (out_root / "manifest.json").write_text(json.dumps({
        "category": cat, "made": made, "sound_images": len(sound),
        "per_image": args.per_image, "variants": args.variants,
        "grades": dict(grade_counter), "seed": args.seed,
        "split_isolated": train_stems is not None,
        "records": records,
    }, ensure_ascii=False, indent=2))
    print(f"\n  生成 {made} 张合成正例 -> {out_root}")
    print(f"  粘贴的等级分布: {dict(grade_counter)}")
    return 0


def harvest_sound_crops(paired: Path, cat: str, sites: dict, rng: random.Random,
                        per_image: int = 3) -> list[Image.Image]:
    """Cut intact-element patches out of the zero-box photographs.

    These are the covering material for the counterfactual negatives: real
    pixels of a real sound element, photographed under real site lighting.
    """
    out = []
    for lab in sorted((paired / cat / "labels").glob("*.txt")):
        if lab.read_text().strip():
            continue
        det = sites.get(lab.stem, [])
        if not det:
            continue
        img_path = sc.find_image(paired / cat / "images", lab.stem)
        if img_path is None:
            continue
        with Image.open(img_path) as h:
            im = h.convert("RGB")
        for d in det[:per_image]:
            x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im.width, x2), min(im.height, y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            out.append(im.crop((x1, y1, x2, y2)))
    return out


def build_negatives(args, rng: random.Random) -> int:
    """Cover the annotated damage on real positives with real sound-element pixels."""
    paired = Path(args.paired_dir)
    cat = args.category

    train_stems = None
    sp = Path(args.split_dir) / f"{cat}_split.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        tr = d["splits"]["train"]
        train_stems = set(tr.keys()) if isinstance(tr, dict) else {
            x["stem"] if isinstance(x, dict) else x for x in tr}

    sites = load_sites(Path(args.audit_json), cat, args.min_site_score)
    sound_crops = harvest_sound_crops(paired, cat, sites, rng)
    if not sound_crops:
        print("no sound crops harvested; need the empty-label audit json")
        return 1
    print(f"category={cat}  sound crops harvested={len(sound_crops)}")

    out_root = Path(args.out_dir) / cat
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    made, records = 0, []
    for lab in sorted((paired / cat / "labels").glob("*.txt")):
        text = lab.read_text().strip()
        if not text:
            continue
        if train_stems is not None and lab.stem not in train_stems:
            continue
        img_path = sc.find_image(paired / cat / "images", lab.stem)
        if img_path is None:
            continue
        boxes = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append(tuple(float(v) for v in parts[1:]))
        if not boxes:
            continue
        with Image.open(img_path) as h:
            base = h.convert("RGB")
        covered = 0
        for cx, cy, w, hh in boxes:
            # Cover slightly wider than the box so the damage does not peek out.
            area = min(0.55, w * hh * 1.35)
            crop = rng.choice(sound_crops)
            res = paste_one(base, crop, cx, cy, area, args.feather,
                            args.photometry, rng, seamless=args.seamless)
            if res is None:
                continue
            base, _ = res
            covered += 1
        if not covered:
            continue
        name = f"{lab.stem}_cpneg"
        base.save(out_root / "images" / f"{name}.jpg", quality=95, subsampling=0)
        (out_root / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
        records.append({"stem": name, "source_positive": lab.stem, "boxes_covered": covered})
        made += 1

    (out_root / "manifest.json").write_text(json.dumps({
        "category": cat, "mode": "negatives", "made": made,
        "sound_crops": len(sound_crops), "seed": args.seed,
        "split_isolated": train_stems is not None, "records": records,
    }, ensure_ascii=False, indent=2))
    print(f"\n  生成 {made} 张 copy-paste 负例 -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
