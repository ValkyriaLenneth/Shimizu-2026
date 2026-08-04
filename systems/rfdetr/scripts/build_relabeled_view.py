#!/usr/bin/env python3
"""Add the boxes the annotators missed, and only those.

The 2026-08-04 completeness audit found that on column_base training images the
same damage type is sometimes boxed and sometimes left as background:

    spalling  17 boxed / 8 unboxed   (32% missing)
    cracking   8 boxed / 7 unboxed   (47% missing)
    corrosion  2 boxed / 14 unboxed  (88% - the client largely does not grade
                                      surface rust, so this is convention, not noise)

Contradictory supervision of that kind is a sufficient explanation for the
failure measured all day: true damage scored below false positives, a recall
ceiling of 0.875/0.940 against 0.514/0.590 usable, and synthetic data making
things worse in proportion to how much was added.

Two rules follow, and both matter more than the code.

**Only add damage types the client demonstrably grades.** Corrosion, fracture and
exposed rebar are almost never boxed in this corpus. Adding them would not repair
the labels, it would impose a different annotation standard than the one the test
split is scored against, and precision would collapse by construction. The
`--kinds` default therefore covers spalling and cracking only.

**Never touch test or valid.** They stay byte-identical to the frozen split, so
every number remains comparable with the delivered baseline. This does mean the
model is being taught to find damage that the test labels may also be missing -
which shows up as false positives and costs precision. That asymmetry is
deliberate and is the thing the experiment measures: if consistency of
supervision matters more than the extra unlabelled-in-test detections cost, the
recall-precision curve moves up despite it.

Grades for added boxes are taken from the OTHER boxes already on the same
photograph, not from the judge's severity. The audit shows the judge calls almost
everything "moderate" (63 of 64 on column_base), so a severity->grade map would
relabel every added box as C and double that class - replacing the client's
grading convention with the judge's. Damage on one element in one survey is
graded consistently, so the modal grade of the image's existing boxes is the
better estimate. Severity is used only as a fallback when an image somehow has no
graded box left.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADE_IDS = {"B": 0, "C": 1, "D": 2}
SEVERITY_TO_GRADE = {"minor": "B", "moderate": "C", "severe": "D"}
KIND_ALIASES = {
    "spalling": ("spall", "delamin", "flak"),
    "cracking": ("crack", "fissure"),
    "corrosion": ("corro", "rust"),
    "fracture": ("fract", "break"),
    "rebar": ("rebar", "reinforce"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--audit-json", default="")
    p.add_argument("--source-suffix", default="bcd_20260725_neg_test_as_valid")
    p.add_argument("--out-suffix", default="bcd_20260725_relabel_test_as_valid")
    p.add_argument("--data-root", default="data")
    p.add_argument("--kinds", default="spalling,cracking",
                   help="damage kinds to add; default = the kinds the client demonstrably grades")
    p.add_argument("--min-severity", default="moderate",
                   choices=["minor", "moderate", "severe"])
    p.add_argument("--min-area", type=float, default=5e-4)
    p.add_argument("--max-area", type=float, default=0.60)
    p.add_argument("--max-add-per-image", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def kind_bucket(raw: str) -> str:
    r = (raw or "").lower()
    for name, keys in KIND_ALIASES.items():
        if any(k in r for k in keys):
            return name
    return "other"


def main() -> int:
    args = parse_args()
    rank = {"minor": 1, "moderate": 2, "severe": 3}
    floor = rank[args.min_severity]
    wanted = {k.strip() for k in args.kinds.split(",") if k.strip()}

    audit_path = Path(args.audit_json) if args.audit_json else \
        Path(f"outputs/gemini_synth/annotation_completeness_full_{args.category}.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    data_root = Path(args.data_root)
    src = data_root / f"rfdetr_{args.category}_{args.source_suffix}"
    dst = data_root / f"rfdetr_{args.category}_{args.out_suffix}"
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=os.link)

    added = Counter()
    touched_images = 0
    skipped_kind = Counter()
    for rec in audit["results"]:
        if "error" in rec:
            continue
        lab = dst / "train" / "labels" / f"{rec['stem']}.txt"
        if not lab.exists():
            continue                       # image is not in this training view
        existing = lab.read_text(encoding="utf-8").strip()
        if not existing:
            continue                       # a negative; adding boxes would change its role
        # Modal grade already assigned on this photograph.
        own = Counter()
        for line in existing.splitlines():
            parts = line.split()
            if len(parts) == 5:
                try:
                    own[int(parts[0])] += 1
                except ValueError:
                    pass
        image_grade = None
        if own:
            top = own.most_common(1)[0][0]
            image_grade = {v: k for k, v in GRADE_IDS.items()}.get(top)
        new_lines = []
        for region in rec.get("regions", []):
            if region.get("covered"):
                continue
            k = kind_bucket(region.get("kind", ""))
            if k not in wanted:
                skipped_kind[k] += 1
                continue
            sev = str(region.get("severity", "")).lower()
            if rank.get(sev, 0) < floor:
                continue
            area = float(region.get("area", 0.0))
            if not (args.min_area <= area <= args.max_area):
                continue
            box = region.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            grade = image_grade or SEVERITY_TO_GRADE.get(sev, "C")
            new_lines.append(f"{GRADE_IDS[grade]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            added[grade] += 1
            if len(new_lines) >= args.max_add_per_image:
                break
        if new_lines:
            # copytree used os.link, so the label file shares an inode with the
            # source dataset. Writing in place would silently edit the source.
            # Break the link first.
            lab.unlink()
            lab.write_text(existing + "\n" + "\n".join(new_lines) + "\n", encoding="utf-8")
            touched_images += 1

    # report
    def count(split):
        L = dst / split / "labels"
        imgs = len(list((dst / split / "images").iterdir()))
        boxes, empty, g = 0, 0, Counter()
        for f in L.glob("*.txt"):
            t = f.read_text(encoding="utf-8").strip()
            if not t:
                empty += 1
                continue
            for line in t.splitlines():
                boxes += 1
                g[int(line.split()[0])] += 1
        return imgs, boxes, empty, g

    i, b, e, g = count("train")
    print(f"[{args.category}] 补标 {sum(added.values())} 个框，涉及 {touched_images} 张图")
    print(f"  按等级: {dict(added)}   (severity->grade: minor→B, moderate→C, severe→D)")
    print(f"  因类型被跳过的区域: {dict(skipped_kind)}  (只补 {sorted(wanted)})")
    print(f"  train: {i} 图 ({e} 负例), {b} 框  B/C/D = {g[0]}/{g[1]}/{g[2]}")
    ti, tb, te, tg = count("test")
    print(f"  test : {ti} 图, {tb} 框 —— 未改动")
    (dst / "relabel_manifest.json").write_text(json.dumps({
        "category": args.category, "source": str(src), "audit": str(audit_path),
        "kinds_added": sorted(wanted), "min_severity": args.min_severity,
        "boxes_added": dict(added), "images_touched": touched_images,
        "skipped_by_kind": dict(skipped_kind),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
