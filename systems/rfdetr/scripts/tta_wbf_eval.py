#!/usr/bin/env python3
"""Single model, test-time augmentation, boxes fused with WBF.

The 2-model WBF ensemble measured on 2026-08-04 lifted brace to recall 0.711 at
P>=0.40 (baseline 0.627) and column_base to 0.611 (baseline 0.556), at twice the
inference cost. TTA asks whether the same trade can be had while deploying ONE
model: the compute doubles either way, but a single checkpoint halves the memory
footprint and removes the operational cost of keeping two models in sync.

The mechanism is the same one that makes the ensemble work. NMS keeps only the
top-scoring box of an overlapping cluster, so a true positive that only one view
found is discarded; WBF averages the cluster's coordinates and accumulates its
confidences, so that box survives. Under a recall-first requirement - a missed
damage in a post-disaster survey being far worse than a false alarm - that
asymmetry is the one we want.

Augmentations are restricted to horizontal flip and mild rescale. Vertical flip
is excluded on purpose: these are survey photographs where gravity is meaningful
(a column base sits at the bottom of a column, spalling accumulates downward), so
flipping vertically produces views the model never saw and should never see.
Photometric augmentation is also excluded - the 2026-07-26 record shows strong
photometric augmentation cost -0.060 / -0.125 recall, because damage grade is
partly an appearance judgement and perturbing brightness perturbs the evidence.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import os

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import numpy as np
import torch
from PIL import Image

from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    IMAGE_EXTS, Prediction, match_counts, merge_counts, metric, read_targets,
)

GRADES = {0: "B", 1: "C", 2: "D"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--views", default="orig,hflip",
                   help="comma list from: orig, hflip, scale090, scale110")
    p.add_argument("--threshold-grid",
                   default="0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--iou-threshold", type=float, default=0.229)
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--floor", type=float, default=0.10)
    p.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg"])
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def make_view(im: Image.Image, view: str):
    if view == "orig":
        return im
    if view == "hflip":
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    if view.startswith("scale"):
        f = int(view.replace("scale", "")) / 100.0
        return im.resize((max(32, int(im.width * f)), max(32, int(im.height * f))), Image.LANCZOS)
    raise SystemExit(f"unknown view {view!r}")


def unmap(boxes: np.ndarray, view: str, W: int, H: int, vw: int, vh: int) -> np.ndarray:
    """Map boxes from the augmented view back to original image coordinates."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy().astype(np.float32)
    if view == "hflip":
        x1 = vw - b[:, 2]
        x2 = vw - b[:, 0]
        b[:, 0], b[:, 2] = x1, x2
    if view.startswith("scale"):
        b[:, [0, 2]] *= W / max(1, vw)
        b[:, [1, 3]] *= H / max(1, vh)
    return b


def main() -> int:
    args = parse_args()
    from ensemble_boxes import weighted_boxes_fusion

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    ds = Path(args.dataset_dir)
    image_dir, label_dir = ds / args.split / "images", ds / args.split / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"TTA views={views} over {len(images)} images")

    model = from_checkpoint_matched(args.checkpoint, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(args.device)

    fused = {}
    for path in images:
        with Image.open(path) as h:
            im = h.convert("RGB")
        W, H = im.size
        bl, sl, ll = [], [], []
        for v in views:
            vi = make_view(im, v)
            det = model.predict(vi, threshold=args.floor)
            xyxy = np.asarray(det.xyxy).reshape(-1, 4).astype(np.float32)
            conf = np.asarray(det.confidence).reshape(-1).astype(np.float32)
            cls = np.asarray(det.class_id).reshape(-1).astype(np.int64)
            keep = cls < args.num_classes
            xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
            xyxy = unmap(xyxy, v, W, H, vi.width, vi.height)
            norm = xyxy / np.array([W, H, W, H], dtype=np.float32) if len(xyxy) else xyxy
            norm = np.clip(norm, 0.0, 1.0)
            bl.append(norm.tolist()); sl.append(conf.tolist()); ll.append(cls.tolist())
        if not any(len(b) for b in bl):
            fused[path.name] = []
            continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=[1.0] * len(views),
                                        iou_thr=args.wbf_iou, skip_box_thr=0.0,
                                        conf_type=args.conf_type)
        fused[path.name] = [(int(c), float(sc), tuple(float(v) for v in (box * np.array([W, H, W, H]))))
                            for box, sc, c in zip(b, s, l, strict=False)]

    thresholds = [float(t) for t in args.threshold_grid.split(",") if t.strip()]
    rows = []
    for combo in itertools.product(thresholds, repeat=args.num_classes):
        total = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(args.num_classes)}
        for path in images:
            with Image.open(path) as h:
                size = h.size
            targets = read_targets(label_dir / f"{path.stem}.txt", size[0], size[1])
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in fused[path.name] if s >= combo[c]]
            merge_counts(total, match_counts(targets, sel, args.iou_threshold, args.num_classes))
        tp = sum(v["tp"] for v in total.values()); fp = sum(v["fp"] for v in total.values())
        fn = sum(v["fn"] for v in total.values())
        p_, r_, f_ = metric(tp, fp, fn)
        row = {"thresholds": ",".join(f"{t}" for t in combo)}
        for i, t in enumerate(combo):
            row[f"threshold_class_{i}"] = t
        for c in range(args.num_classes):
            cp, cr, cf = metric(total[c]["tp"], total[c]["fp"], total[c]["fn"])
            row.update({f"class_{c}_tp": total[c]["tp"], f"class_{c}_fp": total[c]["fp"],
                        f"class_{c}_fn": total[c]["fn"], f"class_{c}_precision": cp,
                        f"class_{c}_recall": cr, f"class_{c}_f1": cf})
        row.update({"images": len(images), "ignored_predictions": 0,
                    "precision": p_, "recall": r_, "f1": f_})
        rows.append(row)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    bf = max(rows, key=lambda r: r["f1"])
    print(f"  best F1 {bf['f1']:.3f} (R {bf['recall']:.3f}/P {bf['precision']:.3f})")
    for fl in (0.60, 0.50, 0.40, 0.30):
        ok = [r for r in rows if r["precision"] >= fl]
        if ok:
            b = max(ok, key=lambda r: r["recall"])
            print(f"  P>={fl:.2f}: recall {b['recall']:.3f} (P {b['precision']:.3f})")
        else:
            print(f"  P>={fl:.2f}: no feasible point")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
