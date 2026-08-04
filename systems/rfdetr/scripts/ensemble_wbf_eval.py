#!/usr/bin/env python3
"""Fuse several checkpoints with Weighted Boxes Fusion and score the result.

Why this fits this project specifically
---------------------------------------
The client requirement is recall-first: a missed damage in a post-disaster survey
is far worse than a false alarm, and the delivered models already ship at
precision 0.596-0.824 rather than anything higher. NMS resolves overlapping
predictions by *discarding* all but the highest-scoring box, so a true positive
that only one member of an ensemble found can still be thrown away. Weighted
Boxes Fusion instead averages the coordinates of every overlapping cluster and
accumulates their confidences, so a box found by one model survives - which is
exactly the asymmetry a recall-first task wants.

It also costs nothing to try. The 2026-08-04 runs left five trained column_base
checkpoints on disk that disagree with each other (their best epochs were
selected on different training views), and disagreement is what an ensemble
converts into coverage. No new data, no retraining.

The scoring path is identical to evaluate_rfdetr_class_threshold_grid.py - same
match IoU 0.229, same per-class threshold grid, same frozen test split - so the
numbers drop straight into the comparison table.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import numpy as np
import torch
from PIL import Image

from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    IMAGE_EXTS, Prediction, match_counts, merge_counts, metric, read_targets,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="repeatable; each is one ensemble member")
    p.add_argument("--weight", action="append", default=None,
                   help="repeatable, one per checkpoint; defaults to equal weights")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--threshold-grid",
                   default="0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--iou-threshold", type=float, default=0.229, help="match IoU for scoring")
    p.add_argument("--wbf-iou", type=float, default=0.55, help="IoU that groups boxes for fusion")
    p.add_argument("--floor", type=float, default=0.10, help="per-model detection floor")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg"])
    # TTA and ensembling are orthogonal: one adds views of the same weights, the
    # other adds weights over the same view. Both feed the same WBF pool, so they
    # compose without any change to the fusion step.
    p.add_argument("--tta-hflip", action="store_true",
                   help="also run each checkpoint on the horizontally flipped image")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ensemble_boxes import weighted_boxes_fusion

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))

    thresholds = [float(t) for t in args.threshold_grid.split(",") if t.strip()]
    n_views = 2 if args.tta_hflip else 1
    base_w = [float(w) for w in args.weight] if args.weight else [1.0] * len(args.checkpoint)
    if len(base_w) != len(args.checkpoint):
        raise SystemExit("--weight count must match --checkpoint count")
    weights = [w for w in base_w for _ in range(n_views)]

    ds = Path(args.dataset_dir)
    image_dir, label_dir = ds / args.split / "images", ds / args.split / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"ensemble of {len(args.checkpoint)} checkpoints over {len(images)} images")

    # Per-model predictions, kept per image so fusion happens image by image.
    per_model: list[dict[str, tuple[list, list, list]]] = []
    for ck in args.checkpoint:
        print(f"  running {Path(ck).parent.parent.name}/{Path(ck).name}")
        model = from_checkpoint_matched(ck, verbose=False)
        ctx = getattr(model, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(args.device)
        views = ["orig"] + (["hflip"] if args.tta_hflip else [])
        for view in views:
            preds: dict[str, tuple[list, list, list]] = {}
            for path in images:
                with Image.open(path) as h:
                    im = h.convert("RGB")
                W, H = im.size
                vi = im.transpose(Image.FLIP_LEFT_RIGHT) if view == "hflip" else im
                det = model.predict(vi, threshold=args.floor)
                xyxy = np.asarray(det.xyxy).reshape(-1, 4).astype(np.float32)
                conf = np.asarray(det.confidence).reshape(-1).astype(np.float32)
                cls = np.asarray(det.class_id).reshape(-1).astype(np.int64)
                keep = cls < args.num_classes
                xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
                if view == "hflip" and len(xyxy):
                    x1 = W - xyxy[:, 2].copy(); x2 = W - xyxy[:, 0].copy()
                    xyxy[:, 0], xyxy[:, 2] = x1, x2
                norm = xyxy / np.array([W, H, W, H], dtype=np.float32) if len(xyxy) else xyxy
                norm = np.clip(norm, 0.0, 1.0)
                preds[path.name] = (norm.tolist(), conf.tolist(), cls.tolist())
            per_model.append(preds)
        del model
        torch.cuda.empty_cache()

    # Fuse, then rescale back to pixels for scoring.
    fused: dict[str, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    for path in images:
        with Image.open(path) as h:
            W, H = h.size
        bl = [per_model[i][path.name][0] for i in range(len(per_model))]
        sl = [per_model[i][path.name][1] for i in range(len(per_model))]
        ll = [per_model[i][path.name][2] for i in range(len(per_model))]
        if not any(len(b) for b in bl):
            fused[path.name] = []
            continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=weights,
                                        iou_thr=args.wbf_iou, skip_box_thr=0.0,
                                        conf_type=args.conf_type)
        out = []
        for box, sc, cl in zip(b, s, l, strict=False):
            x1, y1, x2, y2 = box * np.array([W, H, W, H])
            out.append((int(cl), float(sc), (float(x1), float(y1), float(x2), float(y2))))
        fused[path.name] = out

    rows = []
    for combo in itertools.product(thresholds, repeat=args.num_classes):
        total = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(args.num_classes)}
        for path in images:
            label = label_dir / f"{path.stem}.txt"
            with Image.open(path) as h:
                size = h.size
            targets = read_targets(label, size[0], size[1])
            selected = [Prediction(cls=c, conf=s, xyxy=b)
                        for c, s, b in fused[path.name] if s >= combo[c]]
            merge_counts(total, match_counts(targets, selected, args.iou_threshold, args.num_classes))
        agg = {"tp": sum(v["tp"] for v in total.values()),
               "fp": sum(v["fp"] for v in total.values()),
               "fn": sum(v["fn"] for v in total.values())}
        p_, r_, f_ = metric(agg["tp"], agg["fp"], agg["fn"])
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
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["f1"])
    print(f"  best F1 {best['f1']:.3f} (R {best['recall']:.3f}/P {best['precision']:.3f})")
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
