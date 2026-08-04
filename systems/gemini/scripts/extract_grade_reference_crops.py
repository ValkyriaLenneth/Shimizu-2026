#!/usr/bin/env python3
"""Extract B/C/D damage crops from the paired new-class corpus.

Two products, both needed before any synthetic generation:

1. Per-grade crop banks (`crops/<cat>/<grade>/`) used as few-shot visual
   references when conditioning Gemini. The repository documents no B/C/D
   grading rubric - the semantics live only in the client's CVAT labels - so
   generation must be grounded in real crops rather than in a written
   description of each grade.

2. Contact sheets (`sheets/<cat>_<grade>.jpg`) for human/visual reading of what
   each grade actually looks like.

Crops are taken with context padding around the box, because a damage box at
3-5% of frame area carries almost no surrounding material otherwise, and the
grade is partly read from what the damage sits on.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

GRADE_NAMES = {0: "B", 1: "C", 2: "D"}
CATEGORIES = ("brace", "column_base")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--paired-dir",
        default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724",
        help="root holding <category>/{images,labels}",
    )
    p.add_argument("--out-dir", default="outputs/gemini_synth/grade_references")
    p.add_argument(
        "--context",
        type=float,
        default=1.6,
        help="crop side = box side * context, so 1.6 keeps ~60%% surrounding material",
    )
    p.add_argument("--min-crop", type=int, default=224, help="minimum crop side in px")
    p.add_argument("--sheet-cols", type=int, default=6)
    p.add_argument("--sheet-rows", type=int, default=5)
    p.add_argument("--sheet-cell", type=int, default=260)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--limit-per-grade", type=int, default=0, help="0 = no limit")
    return p.parse_args()


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        out.append((int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4])))
    return out


def crop_with_context(
    im: Image.Image,
    box: tuple[float, float, float, float],
    context: float,
    min_side: int,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Return the crop and the box position inside it, in crop-relative xyxy."""
    W, H = im.size
    cx, cy, bw, bh = box
    px_cx, px_cy = cx * W, cy * H
    px_bw, px_bh = bw * W, bh * H

    side = max(px_bw, px_bh) * context
    side = max(side, min_side)
    side = min(side, min(W, H))

    left = px_cx - side / 2
    top = px_cy - side / 2
    left = max(0.0, min(left, W - side))
    top = max(0.0, min(top, H - side))
    right, bottom = left + side, top + side

    crop = im.crop((int(left), int(top), int(right), int(bottom)))
    rel = (
        (px_cx - px_bw / 2 - left) / side,
        (px_cy - px_bh / 2 - top) / side,
        (px_cx + px_bw / 2 - left) / side,
        (px_cy + px_bh / 2 - top) / side,
    )
    return crop, rel


def build_sheet(
    entries: list[dict],
    out_path: Path,
    cols: int,
    rows: int,
    cell: int,
    mark_box: bool,
) -> None:
    if not entries:
        return
    sheet = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, e in enumerate(entries[: cols * rows]):
        with Image.open(e["crop_path"]) as c:
            c = c.convert("RGB").resize((cell, cell), Image.LANCZOS)
        x0, y0 = (i % cols) * cell, (i // cols) * cell
        sheet.paste(c, (x0, y0))
        if mark_box:
            rx0, ry0, rx1, ry1 = e["box_in_crop"]
            draw.rectangle(
                [x0 + rx0 * cell, y0 + ry0 * cell, x0 + rx1 * cell, y0 + ry1 * cell],
                outline=(0, 255, 120),
                width=2,
            )
        draw.text((x0 + 4, y0 + 4), e["stem"], fill=(255, 220, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    paired = Path(args.paired_dir)
    out = Path(args.out_dir)

    manifest: dict[str, dict] = {}

    for cat in CATEGORIES:
        img_dir, lab_dir = paired / cat / "images", paired / cat / "labels"
        if not img_dir.is_dir():
            print(f"skip {cat}: {img_dir} missing")
            continue

        by_grade: dict[str, list[dict]] = {g: [] for g in GRADE_NAMES.values()}
        empties: list[str] = []

        for lab in sorted(lab_dir.glob("*.txt")):
            boxes = read_boxes(lab)
            candidates = [p for p in img_dir.glob(lab.stem + ".*")]
            if not candidates:
                continue
            img_path = candidates[0]
            if not boxes:
                empties.append(img_path.name)
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    for idx, (cls, cx, cy, bw, bh) in enumerate(boxes):
                        grade = GRADE_NAMES.get(cls)
                        if grade is None:
                            continue
                        crop, rel = crop_with_context(
                            im, (cx, cy, bw, bh), args.context, args.min_crop
                        )
                        cdir = out / "crops" / cat / grade
                        cdir.mkdir(parents=True, exist_ok=True)
                        cpath = cdir / f"{lab.stem}_{idx}.jpg"
                        crop.save(cpath, quality=94)
                        by_grade[grade].append(
                            {
                                "stem": lab.stem,
                                "box_index": idx,
                                "source_image": img_path.name,
                                "source_size": list(im.size),
                                "box_norm": [cx, cy, bw, bh],
                                "box_area_frac": bw * bh,
                                "crop_path": str(cpath),
                                "box_in_crop": list(rel),
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {img_path.name}: {exc}")

        for grade, entries in by_grade.items():
            entries.sort(key=lambda e: -e["box_area_frac"])
            if args.limit_per_grade:
                del entries[args.limit_per_grade :]
            picked = list(entries)
            rng.shuffle(picked)
            build_sheet(
                picked,
                out / "sheets" / f"{cat}_{grade}.jpg",
                args.sheet_cols,
                args.sheet_rows,
                args.sheet_cell,
                mark_box=True,
            )
            print(f"{cat:12s} {grade}: {len(entries):4d} crops")

        # negatives contact sheet, straight from the empty-label images
        neg_entries = []
        for name in empties:
            neg_entries.append(
                {"crop_path": str(img_dir / name), "stem": Path(name).stem,
                 "box_in_crop": [0, 0, 0, 0]}
            )
        rng.shuffle(neg_entries)
        build_sheet(
            neg_entries,
            out / "sheets" / f"{cat}_NEGATIVE.jpg",
            args.sheet_cols,
            args.sheet_rows,
            args.sheet_cell,
            mark_box=False,
        )
        print(f"{cat:12s} negatives: {len(empties)} images")

        manifest[cat] = {
            "crops_per_grade": {g: len(v) for g, v in by_grade.items()},
            "negative_images": empties,
            "entries": by_grade,
        }

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f"\nwrote {out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
