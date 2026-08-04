#!/usr/bin/env python3
"""False-alarm rate on photographs that carry no damage.

The frozen test split contains only images that carry damage, so the question
"does the model stay quiet on a sound element" has never had a number attached -
limitation 2 in the 2026-07-26 handoff, and the first item on every subsequent
plan. It matters more than it sounds: the shortcut finding says these models
detect the *element*, not the damage, and this is the measurement that expresses
that directly. It is also the only way a negatives-style intervention can be
seen to work at all, because on the damage-only test split its main effect is
invisible.

Give it a directory of images asserted to be damage-free and it reports, per
confidence threshold, what fraction of them the model fires on and how many
boxes it draws. Two pools are worth comparing:

* the 141 client zero-box images - already used as training negatives, so their
  number is optimistic; it measures memorisation as much as suppression
* S1 counterfactual negatives - the same scenes as real positives with the
  damage repaired away. The model has never seen these pixels. A model that has
  learnt damage stays quiet; a model that has learnt the element does not.

The comparison between the two is the point: a large gap means the quiet
behaviour on the training negatives did not generalise.

Resolution is recovered from the checkpoint (see checkpoint_resolution.py) so
preprocessing matches training - evaluating a 896 px model at the 576 px class
default would silently misreport.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import numpy as np
from PIL import Image

from checkpoint_resolution import resolution_from_checkpoint

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images-dir", required=True,
                   help="directory of images asserted to be damage-free")
    p.add_argument("--label", default="", help="name for this pool in the report")
    p.add_argument("--thresholds", default="0.20,0.25,0.30,0.40,0.50",
                   help="report at each of these; the delivery thresholds should be among them")
    p.add_argument("--floor", type=float, default=0.10,
                   help="detections below this are never counted")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from rfdetr import RFDETRMedium

    images_dir = Path(args.images_dir)
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no images under {images_dir}")
        return 1

    resolution = resolution_from_checkpoint(args.checkpoint)
    res_kw = {"resolution": resolution} if resolution is not None else {}
    label = args.label or images_dir.name
    print(f"pool       : {label}  ({len(paths)} images)")
    print(f"dir        : {images_dir}")
    print(f"checkpoint : {args.checkpoint}")
    if resolution is not None:
        print(f"  [resolution] building model at {resolution} px (from checkpoint)")

    model = RFDETRMedium(pretrain_weights=args.checkpoint, num_classes=3,
                         device=args.device, **res_kw)

    records = []
    for i, path in enumerate(paths, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        det = model.predict(image, threshold=args.floor)
        xyxy = np.asarray(det.xyxy).reshape(-1, 4)
        conf = np.asarray(det.confidence).reshape(-1)
        cls_ids = np.asarray(det.class_id).reshape(-1)
        keep = [(float(s), int(c), [float(v) for v in b])
                for s, c, b in zip(conf, cls_ids, xyxy, strict=False) if int(c) in GRADES]
        keep.sort(key=lambda t: -t[0])
        records.append({
            "stem": path.stem,
            "size": list(image.size),
            "max_score": keep[0][0] if keep else 0.0,
            "top_grade": GRADES[keep[0][1]] if keep else None,
            "scores": [s for s, _, _ in keep[:20]],
            # Boxes are needed to tell "fires on the element" from "fires on the
            # repaint": a detection that misses the repaired window entirely
            # cannot be an artefact of the edit.
            "boxes": [{"score": s, "grade": GRADES[c], "xyxy": b} for s, c, b in keep[:20]],
        })
        if i % 20 == 0:
            print(f"  {i}/{len(paths)}")

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    n = len(records)
    print(f"\n{'threshold':>10}{'报警图片':>10}{'比例':>9}{'误报框总数':>12}{'每图均值':>10}")
    print("-" * 51)
    rows = []
    for t in thresholds:
        fired = [r for r in records if r["max_score"] >= t]
        boxes = sum(len([s for s in r["scores"] if s >= t]) for r in records)
        rows.append({"threshold": t, "images_fired": len(fired),
                     "image_rate": len(fired) / n, "boxes": boxes, "boxes_per_image": boxes / n})
        print(f"{t:>10.2f}{len(fired):>10}{len(fired)/n:>9.3f}{boxes:>12}{boxes/n:>10.2f}")

    records.sort(key=lambda r: -r["max_score"])
    print(f"\n  最高分前 5 张: " +
          ", ".join(f"{r['stem']}={r['max_score']:.3f}({r['top_grade']})" for r in records[:5]))
    scores = [r["max_score"] for r in records]
    print(f"  峰值分数中位数 {float(np.median(scores)):.3f}   最大 {max(scores):.3f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "pool": label, "images_dir": str(images_dir), "checkpoint": args.checkpoint,
            "resolution": resolution, "n_images": n, "by_threshold": rows,
            "records": records,
        }, ensure_ascii=False, indent=2))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
