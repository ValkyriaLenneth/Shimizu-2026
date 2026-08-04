#!/usr/bin/env python3
"""Keep the detector's boxes, replace its scores with a second-stage classifier's.

The 2026-08-04 measurements say the detector can already find the damage but
cannot rank it:

    recall ceiling at floored thresholds : 0.875 (column_base) / 0.940 (brace)
    recall at the delivery operating point: 0.514 / 0.590

so roughly 0.36 of recall sits in the box set and is thrown away by scoring.

**What the first attempt got wrong.** The classifier was trained to separate
annotated damage crops from element crops cut out of the zero-box photographs -
that is, it learned "does this REGION contain damage". At rescoring time it is
handed ~23 candidate boxes per image, most of them differently-framed windows on
the same element, and it calls all of them damaged. Precision collapsed to 0.085
and best F1 fell from 0.561 to 0.150. Region-level separability (AUC 0.924) is
not the same question as box-level correctness.

**The fix, which is the standard recipe** (false-positive reduction in pulmonary
nodule detection, Hard FP Suppression): the negatives must be *the detector's own
false positives*. Run the detector over the TRAIN split, keep every box that fails
to match a ground-truth box at the scoring IoU, and use those as the negative
class. Positives are the boxes that do match. The classifier then answers the
question rescoring actually asks - "is this box a real detection or one of the
mistakes this detector makes" - rather than "is there damage somewhere nearby".

Nothing about the output contract changes: boxes stay the detector's damage
boxes, grades stay B/C/D, the frozen test split and match IoU 0.229 are
untouched. Only the confidence attached to each box is replaced.

Grade assignment stays with the detector: the confusion analysis found only 2/83
and 1/72 boxes were detected-but-misgraded, so grading is not where recall is
lost.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
from pathlib import Path

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
    p.add_argument("--checkpoint", required=True, help="detector supplying the boxes")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--crops-dir", default="outputs/gemini_synth/grade_references/crops")
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--split-json", default=".local_artifacts/handoff_20260726/split")
    p.add_argument("--audit-dir", default="outputs/rfdetr_new_classes/empty_label_audit")
    p.add_argument("--floor", type=float, default=0.05,
                   help="detector floor; low, because the point is to recover boxes it ranked badly")
    p.add_argument("--context", type=float, default=1.6,
                   help="crop window as a multiple of the box, so the classifier sees some surround")
    p.add_argument("--threshold-grid",
                   default="0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--iou-threshold", type=float, default=0.229)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clf-device", default="")
    p.add_argument("--neg-ratio", type=float, default=3.0,
                   help="cap negatives at this multiple of positives")
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--out-prefix", required=True)
    return p.parse_args()


def train_stems(split_dir: Path, cat: str) -> set[str]:
    p = split_dir / f"{cat}_split.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text())
    tr = d["splits"]["train"]
    s = set(tr.keys()) if isinstance(tr, dict) else {
        x["stem"] if isinstance(x, dict) else x for x in tr}
    return s | set(d.get("train_negatives", []))


def build_classifier(args, device, detector):
    """Fit on the detector's own hits and misses over the TRAIN split."""
    import torchvision.transforms as T
    from sklearn.linear_model import LogisticRegression
    from evaluate_rfdetr_threshold_sweep import box_iou

    ds = Path(args.dataset_dir)
    img_dir, lab_dir = ds / "train" / "images", ds / "train" / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    pos_crops, neg_crops = [], []
    for path in images:
        with Image.open(path) as h:
            im = h.convert("RGB")
        W, H = im.size
        targets = read_targets(lab_dir / f"{path.stem}.txt", W, H)
        det = detector.predict(im, threshold=args.floor)
        xyxy = np.asarray(det.xyxy).reshape(-1, 4)
        cls = np.asarray(det.class_id).reshape(-1)
        for b, c in zip(xyxy, cls, strict=False):
            if int(c) not in GRADES:
                continue
            box = tuple(float(v) for v in b)
            hit = any(box_iou(box, t.xyxy) >= args.iou_threshold for t in targets)
            crop = crop_window(im, box, args.context)
            if crop is None:
                continue
            (pos_crops if hit else neg_crops).append(crop)
    # The detector fires far more often than it hits, so cap negatives to keep the
    # fit from being dominated by one image's worth of duplicates.
    rng = random.Random(args.seed)
    cap = max(len(pos_crops) * args.neg_ratio, 200)
    if len(neg_crops) > cap:
        neg_crops = rng.sample(neg_crops, int(cap))
    print(f"  classifier training crops (from detector on TRAIN): "
          f"matched {len(pos_crops)} / false-positive {len(neg_crops)}")
    if len(pos_crops) < 20 or len(neg_crops) < 20:
        raise SystemExit("not enough crops on one side to fit a classifier")

    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    backbone.eval().to(device)

    def feats(imgs):
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                t = torch.stack([tf(im) for im in imgs[i:i + 64]]).to(device)
                out.append(backbone(t).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384))

    X = np.concatenate([feats(pos_crops), feats(neg_crops)])
    y = np.array([1] * len(pos_crops) + [0] * len(neg_crops))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y)
    return backbone, tf, clf


def crop_window(im: Image.Image, box, context: float):
    W, H = im.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * context / 2, (y2 - y1) * context / 2
    wx1, wy1 = max(0, int(cx - hw)), max(0, int(cy - hh))
    wx2, wy2 = min(W, int(cx + hw)), min(H, int(cy + hh))
    if wx2 - wx1 < 16 or wy2 - wy1 < 16:
        return None
    return im.crop((wx1, wy1, wx2, wy2))


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    clf_dev = torch.device(args.clf_device or args.device)

    print("loading detector")
    model = from_checkpoint_matched(args.checkpoint, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = dev
    print("fitting rescoring classifier on the detector's own hits/misses over TRAIN")
    backbone, tf, clf = build_classifier(args, clf_dev, model)

    ds = Path(args.dataset_dir)
    image_dir, label_dir = ds / args.split / "images", ds / args.split / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    print(f"running detector over {len(images)} images at floor {args.floor}")
    per_image = {}
    for path in images:
        with Image.open(path) as h:
            im = h.convert("RGB")
        W, H = im.size
        det = model.predict(im, threshold=args.floor)
        xyxy = np.asarray(det.xyxy).reshape(-1, 4)
        conf = np.asarray(det.confidence).reshape(-1)
        cls = np.asarray(det.class_id).reshape(-1)
        rows, crops = [], []
        for b, s, c in zip(xyxy, conf, cls, strict=False):
            c = int(c)
            if c not in GRADES:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = (x2 - x1) * args.context / 2, (y2 - y1) * args.context / 2
            wx1, wy1 = max(0, int(cx - hw)), max(0, int(cy - hh))
            wx2, wy2 = min(W, int(cx + hw)), min(H, int(cy + hh))
            if wx2 - wx1 < 16 or wy2 - wy1 < 16:
                continue
            rows.append({"cls": c, "det": float(s), "xyxy": (x1, y1, x2, y2)})
            crops.append(im.crop((wx1, wy1, wx2, wy2)))
        if crops:
            with torch.no_grad():
                probs = []
                for i in range(0, len(crops), 64):
                    t = torch.stack([tf(c) for c in crops[i:i + 64]]).to(clf_dev)
                    f = backbone(t).cpu().numpy()
                    probs.append(clf.predict_proba(f)[:, 1])
                probs = np.concatenate(probs)
            for r, p_ in zip(rows, probs, strict=False):
                r["clf"] = float(p_)
                r["mix"] = float((r["det"] * p_) ** 0.5)
        per_image[path.name] = rows

    n_boxes = sum(len(v) for v in per_image.values())
    print(f"  {n_boxes} candidate boxes rescored ({n_boxes/max(1,len(images)):.1f} per image)")

    thresholds = [float(t) for t in args.threshold_grid.split(",") if t.strip()]
    for variant in ("det", "clf", "mix"):
        rows_out = []
        for combo in itertools.product(thresholds, repeat=args.num_classes):
            total = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
                     for c in range(args.num_classes)}
            for path in images:
                with Image.open(path) as h:
                    size = h.size
                targets = read_targets(label_dir / f"{path.stem}.txt", size[0], size[1])
                sel = [Prediction(cls=r["cls"], conf=r[variant], xyxy=r["xyxy"])
                       for r in per_image[path.name] if r.get(variant, 0.0) >= combo[r["cls"]]]
                merge_counts(total, match_counts(targets, sel, args.iou_threshold, args.num_classes))
            tp = sum(v["tp"] for v in total.values())
            fp = sum(v["fp"] for v in total.values())
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
            rows_out.append(row)
        out = Path(f"{args.out_prefix}_{variant}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n  === 打分方式: {variant} ===")
        for fl in (0.60, 0.50, 0.40, 0.30):
            ok = [r for r in rows_out if r["precision"] >= fl]
            if ok:
                b = max(ok, key=lambda r: r["recall"])
                print(f"    P>={fl:.2f}: recall {b['recall']:.3f} (P {b['precision']:.3f})")
            else:
                print(f"    P>={fl:.2f}: 无可行点")
        bf = max(rows_out, key=lambda r: r["f1"])
        print(f"    best F1 {bf['f1']:.3f} (R {bf['recall']:.3f}/P {bf['precision']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
