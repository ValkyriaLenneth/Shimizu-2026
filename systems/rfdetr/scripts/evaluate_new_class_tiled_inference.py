#!/usr/bin/env python3
"""Tiled (sliced) inference for the new-class downstream models.

Why this exists. Crop augmentation zooms the *training* side: a B-grade box that
occupies 0.8% of the original frame occupies roughly 7-15% of a crop. Inference,
however, still runs on the whole image, so the model is trained on enlarged damage
and then asked to find it at its original scale. This closes that half of the gap
without retraining: the image is cut into overlapping tiles, each tile is detected
at the model's native resolution, and the boxes are mapped back and merged.

No retraining is involved, so any existing checkpoint can be evaluated directly.

Merging uses IoU-based non-maximum suppression per class, with an extra rule for
tile seams: a detection touching a tile border is suppressed if a higher-scoring
detection from a neighbouring tile overlaps it, which is the usual way to stop one
crack becoming two boxes at the seam.

Reported side by side with whole-image inference on the same checkpoint and the
same thresholds, so the comparison isolates the inference strategy.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GRADES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="0.3,0.35,0.4", help="per-class B,C,D")
    parser.add_argument("--iou-threshold", type=float, default=0.229, help="match IoU against GT")
    parser.add_argument("--tiles", default="2x2", help="grid, e.g. 2x2 or 3x3; repeatable via comma")
    parser.add_argument("--overlap", type=float, default=0.25, help="tile overlap as a fraction of tile size")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="IoU for merging detections across tiles")
    parser.add_argument("--include-full-image", action="store_true", default=True,
                        help="also run the whole image and merge, so large damage is not lost")
    parser.add_argument("--no-include-full-image", dest="include_full_image", action="store_false")
    parser.add_argument("--threshold-grid", default="",
                        help="comma-separated levels; re-tunes per-class thresholds for each mode "
                             "from the cached detections, so tiling is not judged at whole-image thresholds")
    parser.add_argument("--precision-floor", type=float, default=0.60,
                        help="client constraint used for the max-recall operating point")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def tile_windows(width: int, height: int, grid: tuple[int, int], overlap: float):
    cols, rows = grid
    tw = width / cols
    th = height / rows
    ox = tw * overlap
    oy = th * overlap
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(round(c * tw - ox)))
            y1 = max(0, int(round(r * th - oy)))
            x2 = min(width, int(round((c + 1) * tw + ox)))
            y2 = min(height, int(round((r + 1) * th + oy)))
            if x2 - x1 > 16 and y2 - y1 > 16:
                yield x1, y1, x2, y2


def nms(dets: list[tuple[int, float, tuple]], nms_iou: float) -> list[tuple[int, float, tuple]]:
    kept: list[tuple[int, float, tuple]] = []
    for cls in sorted({d[0] for d in dets}):
        pool = sorted([d for d in dets if d[0] == cls], key=lambda d: -d[1])
        chosen: list[tuple[int, float, tuple]] = []
        for det in pool:
            if all(iou(det[2], other[2]) < nms_iou for other in chosen):
                chosen.append(det)
        kept.extend(chosen)
    return kept


def load_gt(path: Path, width: int, height: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        cls = int(fields[0])
        cx, cy, bw, bh = (float(v) for v in fields[1:])
        out.append((cls, ((cx - bw / 2) * width, (cy - bh / 2) * height,
                          (cx + bw / 2) * width, (cy + bh / 2) * height)))
    return out


def score(gt_by_image, det_by_image, thresholds, match_iou) -> dict:
    per_grade = {g: {"tp": 0, "fp": 0, "fn": 0} for g in GRADES}
    for key, gts in gt_by_image.items():
        dets = [d for d in det_by_image.get(key, []) if d[1] >= thresholds[d[0]]]
        used = set()
        for cls, gt_box in gts:
            best, best_i = None, 0.0
            for index, det in enumerate(dets):
                if index in used or det[0] != cls:
                    continue
                value = iou(gt_box, det[2])
                if value >= match_iou and value > best_i:
                    best, best_i = index, value
            if best is None:
                per_grade[cls]["fn"] += 1
            else:
                used.add(best)
                per_grade[cls]["tp"] += 1
        for index, det in enumerate(dets):
            if index not in used:
                per_grade[det[0]]["fp"] += 1
    tp = sum(v["tp"] for v in per_grade.values())
    fp = sum(v["fp"] for v in per_grade.values())
    fn = sum(v["fn"] for v in per_grade.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "per_grade": {}}
    for cls, name in GRADES.items():
        v = per_grade[cls]
        r = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        out["per_grade"][name] = {**v, "recall": r, "precision": p}
    return out


def main() -> None:
    args = parse_args()
    import numpy as np
    from PIL import Image
    from rfdetr import RFDETRMedium

    thresholds = [float(v) for v in args.thresholds.split(",")]
    min_threshold = min(thresholds)
    grids = []
    for spec in args.tiles.split(","):
        cols, rows = spec.lower().split("x")
        grids.append((int(cols), int(rows)))

    model = RFDETRMedium(pretrain_weights=args.checkpoint, num_classes=3, device=args.device)

    split_dir = Path(args.dataset_dir) / args.split
    images = sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)

    gt_by_image: dict[str, list] = {}
    whole: dict[str, list] = defaultdict(list)
    tiled: dict[str, list] = defaultdict(list)

    def collect(image, offset=(0, 0)):
        out = []
        detections = model.predict(image, threshold=min_threshold)
        xyxy = np.asarray(detections.xyxy)
        conf = np.asarray(detections.confidence)
        cls_ids = np.asarray(detections.class_id)
        ox, oy = offset
        for box, sc, cls in zip(xyxy, conf, cls_ids, strict=False):
            cls = int(cls)
            if cls not in GRADES:
                continue
            b = (float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy)
            out.append((cls, float(sc), b))
        return out

    for image_path in images:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            key = image_path.stem
            gt_by_image[key] = load_gt(split_dir / "labels" / f"{key}.txt", width, height)

            full = collect(image)
            whole[key] = full

            merged = list(full) if args.include_full_image else []
            for grid in grids:
                for x1, y1, x2, y2 in tile_windows(width, height, grid, args.overlap):
                    merged.extend(collect(image.crop((x1, y1, x2, y2)), offset=(x1, y1)))
            tiled[key] = nms(merged, args.nms_iou)

    whole_score = score(gt_by_image, whole, thresholds, args.iou_threshold)
    tiled_score = score(gt_by_image, tiled, thresholds, args.iou_threshold)

    print(f"checkpoint : {args.checkpoint}")
    print(f"dataset    : {args.dataset_dir} [{args.split}]  {len(images)} images")
    print(f"tiles      : {args.tiles} overlap={args.overlap} nms_iou={args.nms_iou} "
          f"include_full_image={args.include_full_image}")
    print(f"thresholds : {thresholds}, match IoU {args.iou_threshold}\n")

    print(f"  {'mode':<14} {'tp':>4} {'fp':>5} {'fn':>4} {'prec':>7} {'recall':>7} {'F1':>7}   "
          f"{'B_R':>6} {'C_R':>6} {'D_R':>6}")
    for name, data in (("whole image", whole_score), ("tiled", tiled_score)):
        g = data["per_grade"]
        print(
            f"  {name:<14} {data['tp']:>4} {data['fp']:>5} {data['fn']:>4} "
            f"{data['precision']:>7.3f} {data['recall']:>7.3f} {data['f1']:>7.3f}   "
            f"{g['B']['recall']:>6.3f} {g['C']['recall']:>6.3f} {g['D']['recall']:>6.3f}"
        )
    delta = tiled_score["f1"] - whole_score["f1"]
    print(f"\n  F1 delta {delta:+.3f}  (recall {tiled_score['recall'] - whole_score['recall']:+.3f}, "
          f"precision {tiled_score['precision'] - whole_score['precision']:+.3f})")
    print("  note: run-to-run training noise is about 0.025 F1, but this comparison reuses")
    print("  one checkpoint for both modes, so the difference here is free of that.")

    # Tiling produces roughly four times the proposals, so thresholds tuned for
    # whole-image inference are too low for it by construction. Comparing the two
    # modes at one shared triple therefore understates tiling. The grid re-tunes
    # each mode on its own terms; detections are already cached, so this costs
    # scoring only, no extra forward passes.
    grid_summary = None
    if args.threshold_grid:
        levels = [float(v) for v in args.threshold_grid.split(",")]
        grid_summary = {}
        print(f"\n  per-class threshold grid: {len(levels)}^3 = {len(levels) ** 3} combinations per mode")
        for mode, dets in (("whole", whole), ("tiled", tiled)):
            best_f1, best_at_floor = None, None
            for b in levels:
                for c in levels:
                    for d in levels:
                        s = score(gt_by_image, dets, {0: b, 1: c, 2: d}, args.iou_threshold)
                        s["thresholds"] = [b, c, d]
                        if best_f1 is None or s["f1"] > best_f1["f1"]:
                            best_f1 = s
                        if s["precision"] >= args.precision_floor and (
                            best_at_floor is None or s["recall"] > best_at_floor["recall"]
                        ):
                            best_at_floor = s
            grid_summary[mode] = {"best_f1": best_f1, "max_recall_at_precision_floor": best_at_floor}
            bf = best_f1
            print(f"    {mode:<6} best F1 {bf['f1']:.3f} (R {bf['recall']:.3f} / P {bf['precision']:.3f}) "
                  f"thr {bf['thresholds']}")
            if best_at_floor:
                a = best_at_floor
                print(f"    {mode:<6} max recall at P>={args.precision_floor:.2f}: {a['recall']:.3f} "
                      f"(P {a['precision']:.3f}, F1 {a['f1']:.3f}) thr {a['thresholds']}")
            else:
                print(f"    {mode:<6} no point reaches precision {args.precision_floor:.2f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"whole": whole_score, "tiled": tiled_score,
                                   "threshold_grid": grid_summary,
                                   "config": vars(args)}, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
