#!/usr/bin/env python3
"""Audit the empty-label images that were dropped from the training datasets.

141 images (59 ブレース, 82 柱脚) arrived with a label file containing no boxes and
were excluded on the client instruction that every delivered image contains damage.
They are worth 25% more training images for ブレース and 46% for 柱脚, and - more to
the point - they would be the *only* damage-free images in a corpus where every
training image currently contains damage. The measured failure mode is a flood of
false positives on rust, bolts, joints and shadow, which is exactly what a model
that has never seen a negative example does.

Before they can be used as background samples, one question has to be settled:

    "inspected, no damage"  -> valuable hard negatives
    "not yet annotated"     -> poison; training on them teaches the model to
                               suppress real damage

This script answers it empirically instead of by correspondence. It runs a trained
checkpoint over the excluded images and ranks them by the model's peak confidence.
The reasoning:

* If these images genuinely contain no damage, a model that already scores real
  damage at 0.3-0.5 should produce mostly weak, scattered responses on them.
* If a meaningful share carry confident detections that look like the training
  distribution, they are more likely unannotated than damage-free.

The verdict is not automatic - the same model has a known false-positive problem,
so a confident detection is evidence, not proof. The output is therefore built to
be *looked at*: a contact sheet sorted most-suspicious-first, which doubles as the
artifact to send the annotation team.

Runs on CPU by default so it does not contend with training on the GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from checkpoint_resolution import resolution_from_checkpoint

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADES = {0: "B", 1: "C", 2: "D"}
GRADE_COLOR = {0: (255, 196, 0), 1: (255, 108, 0), 2: (229, 30, 30)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=["brace", "column_base"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paired-root", default="data/new_classes_paired_20260724")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="floor for recording a detection; deliberately low so weak responses are visible")
    parser.add_argument("--report-thresholds", default="0.30,0.40,0.50",
                        help="confidence levels to tabulate counts at")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sheet-limit", type=int, default=24, help="images per contact sheet")
    parser.add_argument("--sheet-cols", type=int, default=6)
    parser.add_argument("--cell", type=int, default=420)
    parser.add_argument("--output-dir", default="outputs/rfdetr_new_classes/empty_label_audit")
    return parser.parse_args()


def empty_label_stems(paired_dir: Path) -> list[str]:
    out = []
    for label in sorted((paired_dir / "labels").glob("*.txt")):
        if not label.read_text(encoding="utf-8").strip():
            out.append(label.stem)
    return out


def main() -> int:
    args = parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    from rfdetr import RFDETRMedium

    paired = Path(args.paired_root) / args.category
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = empty_label_stems(paired)
    by_stem = {p.stem: p for p in (paired / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS}
    targets = [(s, by_stem[s]) for s in stems if s in by_stem]
    print(f"[{args.category}] {len(targets)} empty-label images to audit")

    _res = resolution_from_checkpoint(args.checkpoint)
    _res_kw = {"resolution": _res} if _res is not None else {}
    if _res is not None:
        print(f"  [resolution] building model at {_res} px (from checkpoint)")
    model = RFDETRMedium(
        pretrain_weights=args.checkpoint, num_classes=3, device=args.device, **_res_kw
    )

    records = []
    for index, (stem, path) in enumerate(targets, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        detections = model.predict(image, threshold=args.threshold)
        xyxy = np.asarray(detections.xyxy).reshape(-1, 4)
        conf = np.asarray(detections.confidence).reshape(-1)
        cls_ids = np.asarray(detections.class_id).reshape(-1)
        dets = [
            {"grade": GRADES[int(c)], "cls": int(c), "score": float(s), "box": [float(v) for v in b]}
            for b, s, c in zip(xyxy, conf, cls_ids, strict=False)
            if int(c) in GRADES
        ]
        dets.sort(key=lambda d: -d["score"])
        records.append({
            "stem": stem,
            "image": str(path),
            "size": list(image.size),
            "max_score": dets[0]["score"] if dets else 0.0,
            "top_grade": dets[0]["grade"] if dets else None,
            "detections": dets[:10],
            "n_detections": len(dets),
        })
        if index % 20 == 0:
            print(f"  {index}/{len(targets)}")

    records.sort(key=lambda r: -r["max_score"])

    levels = [float(v) for v in args.report_thresholds.split(",")]
    summary = {
        "category": args.category,
        "checkpoint": args.checkpoint,
        "images_audited": len(records),
        "record_threshold": args.threshold,
        "images_with_any_detection": sum(1 for r in records if r["n_detections"]),
        "counts_by_confidence": {
            f">={lv}": sum(1 for r in records if r["max_score"] >= lv) for lv in levels
        },
        "max_score_percentiles": {},
    }
    scores = sorted((r["max_score"] for r in records), reverse=True)
    if scores:
        import statistics
        summary["max_score_percentiles"] = {
            "p90": round(scores[max(0, int(len(scores) * 0.10) - 1)], 4),
            "p50": round(statistics.median(scores), 4),
            "max": round(scores[0], 4),
        }

    (out_dir / f"{args.category}_audit.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Contact sheet, most suspicious first - this is the artifact a human judges.
    cell, cols = args.cell, args.sheet_cols
    chosen = records[: args.sheet_limit]
    rows = (len(chosen) + cols - 1) // cols
    header = 34
    sheet = Image.new("RGB", (cols * cell, rows * (cell + header)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(chosen):
        with Image.open(rec["image"]) as handle:
            im = handle.convert("RGB")
        w, h = im.size
        scale = min(cell / w, cell / h)
        im2 = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        cx, cy = (i % cols) * cell, (i // cols) * (cell + header) + header
        sheet.paste(im2, (cx, cy))
        d2 = ImageDraw.Draw(sheet)
        for det in rec["detections"]:
            if det["score"] < 0.20:
                continue
            x1, y1, x2, y2 = (v * scale for v in det["box"])
            d2.rectangle([cx + x1, cy + y1, cx + x2, cy + y2],
                         outline=GRADE_COLOR[det["cls"]], width=3)
            d2.text((cx + x1 + 3, cy + y1 + 3), f"{det['grade']} {det['score']:.2f}",
                    fill=GRADE_COLOR[det["cls"]])
        label = f"{rec['stem']}  max={rec['max_score']:.2f}  n={rec['n_detections']}"
        draw.text((cx + 6, (i // cols) * (cell + header) + 10), label, fill=(240, 240, 240))
    sheet_path = out_dir / f"{args.category}_suspicious_top{len(chosen)}.jpg"
    sheet.save(sheet_path, quality=88)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsheet -> {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
