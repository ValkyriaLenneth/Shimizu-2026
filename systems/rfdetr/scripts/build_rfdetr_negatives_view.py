#!/usr/bin/env python3
"""Build a train view that adds the excluded empty-label images as background samples.

Why: every image in the current training corpus contains damage, so "element
present" and "damage present" are perfectly correlated and "find the brace" scores
the same training loss as "find the damage on the brace". The audit in
`docs/development_records/2026-07-26-new-classes-shortcut-learning-finding.md`
shows the models took that shortcut - they fire at 0.86-0.95 on intact braces and
sound column bases. No loss-function or learning-rate change can fix a corpus that
cannot distinguish the two hypotheses; only negatives can.

The 141 excluded images (59 ブレース, 82 柱脚) were delivered with label files that
contain zero boxes. In object detection a zero-box image is a valid and useful
training sample - a background sample - not a missing annotation. RF-DETR's YOLO
loader supports them natively.

Known risk, accepted deliberately by the 2026-07-26 decision to use all of them:
a minority carry real but unannotated damage (`f-00189` shows exposed rebar,
`f-00322` spalled concrete, `f-00203` a corroded steel base). Training on those as
background teaches suppression of genuine damage. The audit JSON ranks every image
by model confidence, so `--max-audit-score` can exclude the most suspicious ones
without waiting for annotator triage; the default keeps everything.

Only `train` changes. `test` and `valid` are copied byte-for-byte from the frozen
split, so results stay directly comparable with baseline_v1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADES = {0: "B", 1: "C", 2: "D"}
CATEGORY_LABELS = {"brace": "ブレース", "column_base": "柱脚"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="*", default=sorted(CATEGORY_LABELS))
    parser.add_argument("--source-suffix", default="bcd_20260725_test_as_valid")
    parser.add_argument("--out-suffix", default="bcd_20260725_neg_test_as_valid")
    parser.add_argument("--paired-root", default="data/new_classes_paired_20260724")
    parser.add_argument("--manifest", default="outputs/new_class_annotation_match_20260724/manifest.json")
    parser.add_argument("--audit-dir", default="outputs/rfdetr_new_classes/empty_label_audit")
    parser.add_argument("--max-audit-score", type=float, default=1.01,
                        help="drop negatives whose audit peak confidence exceeds this; default keeps all")
    # negatives_v1 showed a dose-response: 柱脚 at 31% negatives gained +0.097 recall
    # at the precision floor while ブレース at 20% gained +0.036. Repeating raises the
    # ratio without new photographs.
    parser.add_argument("--negative-repeat", type=int, default=1,
                        help="emit each negative N times to raise the negative fraction")
    # A whole-image negative may be too easy - much of the frame is floor, sky or
    # wall the model never confused for damage anyway. The sharper negative is an
    # *intact element* framed the way positives are framed. We have no element
    # boxes on these images, but the baseline model is effectively an element
    # detector (that is the shortcut it learned), so its own detections localise
    # the intact elements for us.
    parser.add_argument("--crop-negatives", action="store_true",
                        help="emit crops around baseline detections instead of whole images")
    parser.add_argument("--crop-context", type=float, default=3.0,
                        help="crop window as a multiple of the detection box, matching the crop2 positive view")
    parser.add_argument("--crop-min-score", type=float, default=0.20)
    parser.add_argument("--crops-per-image", type=int, default=3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def emit_negative_crops(image_path: Path, stem: str, dst_split: Path, detections: list,
                        context: float, min_score: float, per_image: int, repeat: int) -> int:
    """Crop around the baseline model's detections on an undamaged image.

    Those detections are where the model believes damage is, on an image that has
    none, so each crop is precisely a case the model currently gets wrong - framed
    the same way the positive crop view frames real damage.
    """
    from PIL import Image

    kept = [d for d in detections if d["score"] >= min_score][:per_image]
    if not kept:
        return 0
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    written = 0
    for index, det in enumerate(kept):
        x1, y1, x2, y2 = det["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = max(8.0, (x2 - x1) * context / 2)
        half_h = max(8.0, (y2 - y1) * context / 2)
        left, top = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        right, bottom = min(width, int(cx + half_w)), min(height, int(cy + half_h))
        if right - left < 16 or bottom - top < 16:
            continue
        crop = image.crop((left, top, right, bottom))
        for rep in range(repeat):
            name = f"{stem}_neg{index}" + (f"_r{rep}" if rep else "")
            crop.save(dst_split / "images" / f"{name}.jpg", quality=92)
            (dst_split / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
            written += 1
    return written


def copy_split(src_split: Path, dst_split: Path) -> dict:
    grades: Counter[str] = Counter()
    boxes = 0
    images = 0
    for label in sorted((src_split / "labels").glob("*.txt")):
        stem = label.stem
        image = next(
            (p for p in (src_split / "images").iterdir()
             if p.stem == stem and p.suffix.lower() in IMAGE_EXTS),
            None,
        )
        if image is None:
            continue
        link(image, dst_split / "images" / image.name)
        link(label, dst_split / "labels" / label.name)
        images += 1
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                boxes += 1
                grades[GRADES[int(parts[0])]] += 1
    return {"images": images, "boxes": boxes, "boxes_by_grade": dict(sorted(grades.items()))}


def main() -> int:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data_root = Path(args.data_root)
    summary: dict[str, dict] = {}

    for category in args.categories:
        src = data_root / f"rfdetr_{category}_{args.source_suffix}"
        dst = data_root / f"rfdetr_{category}_{args.out_suffix}"
        if dst.exists():
            if not args.overwrite:
                raise FileExistsError(f"{dst} exists; pass --overwrite")
            shutil.rmtree(dst)

        stats = {name: copy_split(src / name, dst / name) for name in ("train", "test", "valid")}

        # Scene groups already present in test - a negative sharing one would leak a
        # near-identical view of a test scene into train.
        pos = [r for r in manifest["records"]
               if r["category"] == category and r["source"] == "annotated"
               and r["is_representative"] and r["box_count"] > 0]
        test_stems = {p.stem for p in (src / "test" / "labels").glob("*.txt")}
        test_groups = {r["scene_group_id"] for r in pos if r["stem"] in test_stems}

        audit_path = Path(args.audit_dir) / f"{category}_audit.json"
        audit_score: dict[str, float] = {}
        audit_boxes: dict[str, list] = {}
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_score = {r["stem"]: r["max_score"] for r in audit["records"]}
            audit_boxes = {r["stem"]: r["detections"] for r in audit["records"]}
        elif args.crop_negatives:
            raise FileNotFoundError(f"--crop-negatives needs {audit_path}")

        negatives = [r for r in manifest["records"]
                     if r["category"] == category and r["source"] == "annotated"
                     and r["is_representative"] and r["box_count"] == 0]

        paired_images = {p.stem: p for p in (Path(args.paired_root) / category / "images").iterdir()
                         if p.suffix.lower() in IMAGE_EXTS}

        added, skipped_leak, skipped_score, missing = 0, [], [], []
        for record in negatives:
            stem = record["stem"]
            if record["scene_group_id"] in test_groups:
                skipped_leak.append(stem)
                continue
            if audit_score.get(stem, 0.0) > args.max_audit_score:
                skipped_score.append(stem)
                continue
            image = paired_images.get(stem)
            if image is None:
                missing.append(stem)
                continue

            if args.crop_negatives:
                added += emit_negative_crops(
                    image, stem, dst / "train", audit_boxes.get(stem, []),
                    args.crop_context, args.crop_min_score, args.crops_per_image,
                    args.negative_repeat,
                )
                continue

            for rep in range(args.negative_repeat):
                name = stem if rep == 0 else f"{stem}_r{rep}"
                target = dst / "train" / "images" / f"{name}{image.suffix}"
                if rep == 0:
                    link(image, target)
                else:
                    # A repeat must be a distinct file, and a hardlink would collide
                    # on stem; copying keeps the loader's stem->label pairing intact.
                    shutil.copy2(image, target)
                # An empty .txt keeps the dataset self-describing and satisfies tools
                # that pair every image with a label; the loader reads zero boxes from
                # it exactly as it would from an absent file.
                (dst / "train" / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
                added += 1

        names = {i: f"{CATEGORY_LABELS[category]}の損傷程度{g}" for i, g in GRADES.items()}
        (dst / "data.yaml").write_text(
            yaml.safe_dump(
                {"path": str(dst), "train": "train/images", "val": "valid/images",
                 "test": "test/images", "nc": len(names), "names": names},
                allow_unicode=True, sort_keys=False),
            encoding="utf-8")

        info = {
            "source": str(src),
            "train_positive_images": stats["train"]["images"],
            "negatives_added": added,
            "train_total_images": stats["train"]["images"] + added,
            "negative_fraction": round(added / max(1, stats["train"]["images"] + added), 3),
            "skipped_scene_group_leak": skipped_leak,
            "skipped_high_audit_score": skipped_score,
            "missing_image": missing,
            "splits": stats,
            "max_audit_score": args.max_audit_score,
        }
        (dst / "build_summary.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[category] = info
        print(f"[{category}] train {stats['train']['images']} positives + {added} negatives "
              f"= {info['train_total_images']} ({info['negative_fraction']:.0%} negative); "
              f"test {stats['test']['images']} imgs / {stats['test']['boxes']} boxes unchanged")
        if skipped_leak:
            print(f"  skipped {len(skipped_leak)} for scene-group leakage")
        if skipped_score:
            print(f"  skipped {len(skipped_score)} above audit score {args.max_audit_score}")

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
