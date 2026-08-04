#!/usr/bin/env python3
"""How much damage is present but unannotated? Measure the label noise directly.

Every intervention tried on 2026-08-04 assumed the labels are right. Three
observations say that assumption deserves a test:

* the calibrated QC judge read damage in 69% of the 82 column_base photographs
  the client delivered with EMPTY labels, and its grade distribution there
  (B 3 / C 7 / D 1 / none 5) is barely distinguishable from real positives
  (B 5 / C 9 / D 2)
* the 2026-07-24 pairing found 45 conflict clusters - 14 grade disagreements,
  20 coordinate drifts, 11 unannotated duplicates - none of them resolved
* three known cases (`f-00189` exposed rebar, `f-00322` spalling, `f-00203`
  corroded base) sit in the negative pool with visible damage

If the same kind of damage is sometimes boxed and sometimes treated as
background, the supervision is contradictory, and a detector's rational response
is to give such regions a middling score. That is exactly the failure measured
today: true damage ranked below false positives, recall ceiling 0.875/0.940 but
only 0.514/0.590 usable.

This script asks a vision model for a full damage inventory on each ANNOTATED
image and compares it against the ground-truth boxes. Regions the model finds
that no box covers are candidate under-annotations. The output is a rate, not a
verdict - the judge over-reads relative to the client's convention (that is
established), so the number is an upper bound on missing labels and the useful
signal is how it varies by damage type and by grade.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import synth_common as sc

SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--split-json", default=".local_artifacts/handoff_20260726/split")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--send-size", type=int, default=1280)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--min-severity", default="moderate",
                   choices=sorted(SEVERITY_RANK, key=SEVERITY_RANK.get),
                   help="only count unboxed regions at or above this severity")
    p.add_argument("--min-area", type=float, default=5e-4)
    p.add_argument("--cover-iou", type=float, default=0.10,
                   help="a found region counts as annotated if it overlaps a GT box this much")
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def audit_one(stem: str, img_path: Path, gt: list, args, key: str) -> dict:
    try:
        with Image.open(img_path) as h:
            image = h.convert("RGB")
    except Exception as exc:
        return {"stem": stem, "error": f"open_failed: {exc}"}

    inv, err = sc.inventory_damage(args.model, image, args.category, key,
                                   send_size=args.send_size, timeout=args.timeout)
    if inv is None:
        return {"stem": stem, "error": err or "inventory_failed"}

    floor = SEVERITY_RANK[args.min_severity]
    found, unboxed = [], []
    for r in inv.get("regions", []):
        b = r.get("box_2d") or []
        if len(b) != 4:
            continue
        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) / 1000.0 for v in b]
        if xmax <= xmin or ymax <= ymin:
            continue
        box = (xmin, ymin, xmax, ymax)
        area = (xmax - xmin) * (ymax - ymin)
        sev = str(r.get("severity", "none")).lower()
        if SEVERITY_RANK.get(sev, 0) < floor or area < args.min_area:
            continue
        kind = (r.get("damage_type") or r.get("kind") or "").lower()
        covered = max((iou(box, g) for g in gt), default=0.0) >= args.cover_iou
        rec = {"severity": sev, "area": round(area, 5), "kind": kind,
               # Normalised xyxy. Needed downstream: an under-annotated region is
               # only actionable if we can actually place the box.
               "box": [round(v, 6) for v in box],
               "covered": covered, "note": (r.get("description") or "")[:120]}
        found.append(rec)
        if not covered:
            unboxed.append(rec)
    return {"stem": stem, "gt_boxes": len(gt), "found": len(found),
            "unboxed": len(unboxed), "regions": found}


def main() -> int:
    args = parse_args()
    key = sc.require_api_key()
    rng = random.Random(args.seed)
    paired = Path(args.paired_dir)
    cat = args.category

    train = None
    sp = Path(args.split_json) / f"{cat}_split.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        tr = d["splits"]["train"]
        train = set(tr.keys()) if isinstance(tr, dict) else {
            x["stem"] if isinstance(x, dict) else x for x in tr}

    jobs = []
    for lab in sorted((paired / cat / "labels").glob("*.txt")):
        text = lab.read_text().strip()
        if not text:
            continue                       # annotated positives only
        if train is not None and lab.stem not in train:
            continue
        gt = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, w, h = [float(v) for v in parts[1:]]
            gt.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        img = sc.find_image(paired / cat / "images", lab.stem)
        if img is not None and gt:
            jobs.append((lab.stem, img, gt))
    rng.shuffle(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"category={cat}  annotated train images audited={len(jobs)}  "
          f"counting unboxed regions at severity>={args.min_severity}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(audit_one, s, p, g, args, key): s for s, p, g in jobs}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if "error" in r:
                print(f"[{i}/{len(jobs)}] ERR {r['stem']}: {r['error'][:50]}")
            else:
                print(f"[{i}/{len(jobs)}] {r['stem']:<10} GT框 {r['gt_boxes']}  "
                      f"判定发现 {r['found']}  其中无框覆盖 {r['unboxed']}")

    ok = [r for r in results if "error" not in r]
    tot_gt = sum(r["gt_boxes"] for r in ok)
    tot_found = sum(r["found"] for r in ok)
    tot_unboxed = sum(r["unboxed"] for r in ok)
    imgs_with_unboxed = sum(1 for r in ok if r["unboxed"] > 0)
    print(f"\n{'='*66}")
    print(f"  审查图片            : {len(ok)}")
    print(f"  GT 框总数           : {tot_gt}")
    print(f"  判定发现的损伤区域   : {tot_found}")
    print(f"  其中【无 GT 框覆盖】: {tot_unboxed}  "
          f"({tot_unboxed/max(1,tot_found):.0%} of found)")
    print(f"  含未标注损伤的图片   : {imgs_with_unboxed} / {len(ok)} "
          f"({imgs_with_unboxed/max(1,len(ok)):.0%})")
    kinds = Counter(x["kind"] for r in ok for x in r["regions"] if not x["covered"])
    sevs = Counter(x["severity"] for r in ok for x in r["regions"] if not x["covered"])
    print(f"  未标注区域的类型分布 : {dict(kinds.most_common(6))}")
    print(f"  未标注区域的严重度   : {dict(sevs)}")

    out = Path(args.out_json) if args.out_json else \
        Path(f"outputs/gemini_synth/annotation_completeness_{cat}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "category": cat, "model": args.model, "min_severity": args.min_severity,
        "images": len(ok), "gt_boxes": tot_gt, "found": tot_found,
        "unboxed": tot_unboxed, "images_with_unboxed": imgs_with_unboxed,
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
