#!/usr/bin/env python3
"""Triage unlabelled photographs into damage-free and damage-bearing.

Two pools of images in this project carry no boxes, and both are unusable until
someone decides which of the two things "no box" means:

* the 45 JSCA images delivered 2026-07-24 (`extra_unlabelled_images` in the
  match manifest) - 29 brace, 16 column_base. Never used for anything: they are
  the only real data in the project that has not entered training, which makes
  them the only admissible material for a held-out false-alarm benchmark.
* the 141 zero-box client images already used as background negatives, which
  the 2026-08-03 QC judge read as damaged in 69% of cases.

A false-alarm benchmark built on images that actually contain damage measures
nothing - a detection there is correct, not a false alarm. So the images are put
to a vision model first, and only the ones it reads as sound are eligible.

The model is asked for a damage inventory (the same call the S1 generator uses
to find unannotated damage), not a yes/no verdict: a region list can be checked
against, and its severity and area are what decide eligibility. Judgements are
recorded per image with the model's own reasons so a human can overrule them.

Calibration note: on the 2026-08-03 controls this judge flagged damage on 69% of
client-labelled-empty column_base images, so it over-reads relative to the
client's labelling convention. That direction is the safe one here - it removes
images from the benchmark rather than admitting damaged ones - but it means a
low eligible count is expected and is not by itself evidence the pool is dirty.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from PIL import Image

import synth_common as sc

CATEGORY_DIRS = {
    "brace": "20260724_学習用データ_追加分_ブレース,柱脚/ブレース_追加分_JSCA講習より",
    "column_base": "20260724_学習用データ_追加分_ブレース,柱脚/柱脚_追加分_JSCA講習より",
}
SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", default="data/raw_new_classes_20260630")
    p.add_argument("--category", default="brace", choices=sorted(CATEGORY_DIRS))
    p.add_argument("--images-dir", default="", help="override; triage any directory of images")
    p.add_argument("--out-json", default="")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--send-size", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--min-severity", default="minor",
                   choices=sorted(SEVERITY_RANK, key=SEVERITY_RANK.get),
                   help="regions at or above this severity disqualify an image")
    p.add_argument("--min-area", type=float, default=1e-4,
                   help="regions smaller than this fraction of the frame are ignored")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def judge_one(path: Path, args, api_key: str) -> dict:
    try:
        with Image.open(path) as im:
            image = im.convert("RGB")
            size = image.size
    except Exception as exc:  # unreadable file is a data problem, not a verdict
        return {"stem": path.stem, "error": f"open_failed: {exc}"}

    inventory, err = sc.inventory_damage(
        args.model, image, args.category, api_key,
        send_size=args.send_size, timeout=args.timeout,
    )
    if inventory is None:
        return {"stem": path.stem, "error": err or "inventory_failed"}

    floor = SEVERITY_RANK[args.min_severity]
    regions = []
    for r in inventory.get("regions", []):
        b = r.get("box_2d") or []
        area = 0.0
        if len(b) == 4:
            ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) / 1000.0 for v in b]
            area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        sev = str(r.get("severity", "none")).lower()
        regions.append({
            "severity": sev,
            "area": round(area, 5),
            "kind": r.get("damage_type") or r.get("kind") or "",
            "note": (r.get("description") or "")[:200],
            "counts": SEVERITY_RANK.get(sev, 0) >= floor and area >= args.min_area,
        })
    disqualifying = [r for r in regions if r["counts"]]
    return {
        "stem": path.stem,
        "image": str(path),
        "size": size,
        "sound": not disqualifying,
        "regions": regions,
        "disqualifying": len(disqualifying),
        "worst_severity": max((r["severity"] for r in regions),
                              key=lambda s: SEVERITY_RANK.get(s, 0), default="none"),
    }


def main() -> int:
    args = parse_args()
    api_key = sc.require_api_key()

    images_dir = Path(args.images_dir) if args.images_dir else \
        Path(args.raw_root) / CATEGORY_DIRS[args.category]
    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no images under {images_dir}")
        return 1

    print(f"category={args.category} model={args.model} images={len(paths)}")
    print(f"dir={images_dir}")

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_one, p, args, api_key): p for p in paths}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row = fut.result()
            results.append(row)
            if "error" in row:
                mark = "ERR "
            else:
                mark = "sound" if row["sound"] else "DMG "
            detail = "" if "error" in row else \
                f"worst={row['worst_severity']:<8} regions={len(row['regions'])} dq={row['disqualifying']}"
            print(f"[{i}/{len(paths)}] {mark} {row['stem']:<12} {detail}{row.get('error','')}")

    results.sort(key=lambda r: r["stem"])
    ok = [r for r in results if r.get("sound")]
    dmg = [r for r in results if "error" not in r and not r["sound"]]
    err = [r for r in results if "error" in r]

    print(f"\n{'='*60}")
    print(f"  眼中無損傷 (eligible for the benchmark) : {len(ok):>3} / {len(results)}")
    print(f"  判定に損傷あり (excluded)              : {len(dmg):>3}")
    print(f"  エラー                                  : {len(err):>3}")
    if dmg:
        from collections import Counter
        kinds = Counter(r["worst_severity"] for r in dmg)
        print(f"  excluded by worst severity: {dict(kinds)}")

    out = Path(args.out_json) if args.out_json else \
        Path(f"outputs/gemini_synth/triage_{args.category}_unlabelled.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "category": args.category,
        "images_dir": str(images_dir),
        "model": args.model,
        "min_severity": args.min_severity,
        "min_area": args.min_area,
        "counts": {"sound": len(ok), "damaged": len(dmg), "error": len(err)},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
