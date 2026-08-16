#!/usr/bin/env python3
"""Stress-test the deliverable the same way its rejected competitors were tested.

Three claims fell today when retrained under other seeds: the pseudo-label gain,
the synthetic-data gain, and the three-member fusion at 0.389, which dropped to
0.290 when its third member was retrained from the same data. The two-member
0.329 that remains does not involve any model trained today, so seed variation
cannot touch it -- but that is an argument, not a measurement, and the same
argument would have sounded fine for the others before they were checked.

What can still be wrong is the *tuning*. The parameters were chosen by taking a
maximum over 315 fusion configurations times 3 375 threshold triples on 72 boxes.
A maximum over that many cells overfits the test split even with fixed weights,
and the honest question is how much of the 0.329 is real separation from the old
0.302 rather than a luckier cell.

Two checks, neither needing a GPU or a retrain:

  bootstrap   resample the 45 test images with replacement, re-select the best
              configuration inside each resample, and report the spread. This is
              the sampling variability the single number hides.
  holdout     split the images in half, tune on one half, score the chosen
              configuration on the other. Tuning and evaluation no longer share
              data, which is the only way to see how much of the gain survives.
"""
from __future__ import annotations
import csv, itertools, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
MEMBERS = ["/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth",
           "/workspace/handoff_20260804/checkpoints/column_base_copypaste_epoch_075.pth"]
GRID = [0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.50]
MATCH_IOU, NC, TARGET = 0.229, 3, 0.70
CONFIGS = [(0.20,(1,2),"max"), (0.40,(1,1),"avg")]   # new vs old delivery setting
RNG = np.random.default_rng(20260816)

def fuse(raw, sizes, iou, w, conf):
    out = {}
    for name in raw[0]:
        bl, sl, ll = [], [], []
        for s in raw:
            r = s[name]; sc = np.asarray(r["scores"], np.float32); keep = sc >= 0.10
            bl.append(np.asarray(r["boxes"], np.float32).reshape(-1,4)[keep].tolist())
            sl.append(sc[keep].tolist()); ll.append(np.asarray(r["classes"], np.int64)[keep].tolist())
        W, H = sizes[name]
        if not any(len(b) for b in bl): out[name] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=list(w), iou_thr=iou,
                                        skip_box_thr=0.0, conf_type=conf)
        out[name] = [(int(c), float(x), tuple((np.asarray(bb)*np.array([W,H,W,H])).tolist()))
                     for bb, x, c in zip(b, s, l)]
    return out

def best_on(fused, targets, names):
    best = None
    for combo in itertools.product(GRID, repeat=NC):
        tot = {c: {"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
        for n in names:
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c,s,b in fused[n] if s >= combo[c]]
            merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
        tp=sum(v["tp"] for v in tot.values()); fp=sum(v["fp"] for v in tot.values()); fn=sum(v["fn"] for v in tot.values())
        p,r,_ = metric(tp,fp,fn)
        per=[metric(tot[c]["tp"],tot[c]["fp"],tot[c]["fn"])[1] for c in range(NC)]
        if r>=TARGET and all(v>=TARGET for v in per) and (best is None or p>best[0]):
            best=(p, combo)
    return best

def score_at(fused, targets, names, combo):
    tot = {c: {"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
    for n in names:
        sel = [Prediction(cls=c, conf=s, xyxy=b) for c,s,b in fused[n] if s >= combo[c]]
        merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
    tp=sum(v["tp"] for v in tot.values()); fp=sum(v["fp"] for v in tot.values()); fn=sum(v["fn"] for v in tot.values())
    p,r,_ = metric(tp,fp,fn)
    per=[metric(tot[c]["tp"],tot[c]["fp"],tot[c]["fn"])[1] for c in range(NC)]
    return p, r, all(v>=TARGET for v in per) and r>=TARGET

def main():
    device = sys.argv[1] if len(sys.argv)>1 else "cuda:0"
    imgs = sorted(p for p in (DS/"test"/"images").iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h: sizes[p.name] = h.size
        targets[p.name] = read_targets(DS/"test"/"labels"/f"{p.stem}.txt", *sizes[p.name])
    raw = []
    for ck in MEMBERS:
        m = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(m,"model",None)
        if ctx is not None and hasattr(ctx,"device"): ctx.device = torch.device(device)
        d = {}
        for p in imgs:
            with Image.open(p) as h: im = h.convert("RGB")
            det = m.predict(im, threshold=0.10)
            cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
            W,H = sizes[p.name]; xy = np.asarray(det.xyxy).reshape(-1,4)[keep]
            d[p.name] = {"boxes": np.clip(xy/np.array([W,H,W,H],np.float32),0,1).tolist(),
                         "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                         "classes": cls[keep].tolist()}
        raw.append(d); del m; torch.cuda.empty_cache()

    names = [p.name for p in imgs]
    for iou, w, conf in CONFIGS:
        fused = fuse(raw, sizes, iou, w, conf)
        full = best_on(fused, targets, names)
        tag = f"iou {iou} / {w[0]}:{w[1]} / {conf}"
        print(f"\n=== {tag} ===")
        print(f"全量调优(现在报的数): P = {full[0]:.3f}  阈值 {full[1]}")

        boots = []
        for _ in range(200):
            samp = list(RNG.choice(names, size=len(names), replace=True))
            b = best_on(fused, targets, samp)
            if b: boots.append(b[0])
        if boots:
            lo, hi = np.percentile(boots, [5, 95])
            print(f"自助重采样 200 次: 中位 {np.median(boots):.3f}, 90% 区间 [{lo:.3f}, {hi:.3f}]")

        halves = []
        for _ in range(30):
            perm = list(RNG.permutation(names))
            a, bb = perm[:len(perm)//2], perm[len(perm)//2:]
            sel = best_on(fused, targets, a)
            if not sel: continue
            p, r, ok = score_at(fused, targets, bb, sel[1])
            halves.append(p)
        if halves:
            print(f"半分调优/半分评估 30 次: 中位 {np.median(halves):.3f} "
                  f"(比全量调优低 {full[0]-np.median(halves):+.3f})")

if __name__ == "__main__":
    main()
