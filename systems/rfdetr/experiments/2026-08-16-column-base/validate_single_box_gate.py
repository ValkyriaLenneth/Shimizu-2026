#!/usr/bin/env python3
"""Is the single-box gate better than the union, or is +0.010 noise?

Taking the router's highest-scoring column-base box instead of the union of all
of them shrinks the gated region from 74% of the frame to 55%, and on the frozen
split it reads 0.405 precision against 0.395, with sound-image boxes 1.52 against
1.66 and all four recalls unchanged.

Both numbers sit below the detection floors established today: 0.045 for
precision on 72 boxes, 0.29 boxes per image on 29 sound photographs. Three
precision claims of larger size were retracted this afternoon after holdout
validation, so neither figure may be adopted on its face.

Two things make this case structurally different from those three, and both are
testable rather than asserted. The variant is not the maximum of a swept grid --
"use one box instead of all of them" is a single stated choice with no free
parameter. And at fixed thresholds a smaller gate can only remove more boxes,
never add any, so recall can only fall or stay while false alarms can only fall;
the measured recall did not move at all, which means nothing removed was a true
positive.

Paired bootstrap on both sides at the delivered thresholds, then a holdout for
the precision side where selection could still hide.
"""
from __future__ import annotations
import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import itertools, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

THR, GATE, NBOOT, NSPLIT = (0.12, 0.20, 0.12), 0.5, 2000, 400


def members(device, paths, sizes, single):
    m = from_checkpoint_matched(RG.ROUTER, device=device, verbose=False)
    ctx = getattr(m, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    out = {}
    for p in paths:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = m.predict(im, threshold=RG.ROUTER_THR)
        cls = np.asarray(det.class_id).reshape(-1)
        conf = np.asarray(det.confidence).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        sel = cls == RG.CB_CLASS
        if not sel.any():
            out[p.name] = None
            continue
        b = xy[sel]
        if single:
            x1, y1, x2, y2 = b[int(np.argmax(conf[sel]))]
        else:
            x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        W, H = sizes[p.name]
        dw, dh = (x2 - x1) * RG.MARGIN, (y2 - y1) * RG.MARGIN
        out[p.name] = [max(0, x1 - dw), max(0, y1 - dh), min(W, x2 + dw), min(H, y2 + dh)]
    del m; torch.cuda.empty_cache()
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sound = sorted(p for p in RG.SOUND.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, ssz, targets = {}, {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    for p in sound:
        with Image.open(p) as h:
            ssz[p.name] = h.size
    keys = [(m, v) for m in MEMBERS for v in RG.VIEWS]
    w = [RG.RATIO[i] / len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
    pt = fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU)
    ps = fuse(RG.detect(device, sound, ssz), keys, w, ssz, RG.IOU)

    tabs, snd = {}, {}
    names = sorted(targets)
    for tag, single in (("union", False), ("single", True)):
        mt, ms = members(device, imgs, sizes, single), members(device, sound, ssz, single)
        gt_ = RG.apply_gate(pt, mt, GATE)
        gs_ = RG.apply_gate(ps, ms, GATE)
        tp = np.zeros((len(names), NC), np.int32); fp = np.zeros_like(tp)
        for i, n in enumerate(names):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in gt_[n] if s >= THR[c]]
            merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            for c in range(NC):
                tp[i, c] = tot[c]["tp"]; fp[i, c] = tot[c]["fp"]
        tabs[tag] = (tp, fp, gt_)
        sn = sorted(gs_)
        snd[tag] = np.array([sum(1 for c, s, _ in gs_[n] if s >= THR[c]) for n in sn])

    def prec(t, idx):
        tp, fp, _ = t
        a, b = tp[idx].sum(), fp[idx].sum()
        return a / max(a + b, 1)

    full = np.arange(len(names))
    print(f"全量: 并集 {prec(tabs['union'], full):.4f}  单框 {prec(tabs['single'], full):.4f}")
    print(f"健全图 箱/张: 并集 {snd['union'].mean():.3f}  单框 {snd['single'].mean():.3f}")

    rng = np.random.default_rng(20260816)
    dp, db = [], []
    for _ in range(NBOOT):
        i1 = rng.integers(0, len(names), len(names))
        dp.append(prec(tabs["single"], i1) - prec(tabs["union"], i1))
        i2 = rng.integers(0, len(snd["union"]), len(snd["union"]))
        db.append(snd["single"][i2].mean() - snd["union"][i2].mean())
    for lab, d, unit in (("precision", np.array(dp), ""), ("健全 箱/张", np.array(db), "")):
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"配对自助 {lab:12s} 均值 {d.mean():+.4f}{unit}  95% [{lo:+.4f}, {hi:+.4f}]  "
              f"单框更好 {np.mean(d < 0) if '箱' in lab else np.mean(d > 0):.1%}")

    # Holdout on precision: thresholds re-chosen inside each half under the same
    # four-target rule, so the comparison is not anchored to one tuned triple.
    def best_thr(gt_, idx):
        win = None
        for combo in itertools.product(GRID, repeat=NC):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            for i in idx:
                n = names[i]
                sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in gt_[n] if s >= combo[c]]
                merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
            fn = sum(v["fn"] for v in tot.values())
            p_, r_, _ = metric(tp, fp, fn)
            per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
            if r_ >= RG.TARGET and all(v >= RG.TARGET for v in per) and (win is None or p_ > win[0]):
                win = (p_, combo)
        return win

    def prec_at(gt_, idx, combo):
        tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
        for i in idx:
            n = names[i]
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in gt_[n] if s >= combo[c]]
            merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
        tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
        return tp / max(tp + fp, 1)

    rng = np.random.default_rng(20260816)
    d = []
    for _ in range(NSPLIT):
        perm = rng.permutation(len(names)); a, b = perm[:len(perm)//2], perm[len(perm)//2:]
        ws, wu = best_thr(tabs["single"][2], a), best_thr(tabs["union"][2], a)
        if ws and wu:
            d.append(prec_at(tabs["single"][2], b, ws[1]) - prec_at(tabs["union"][2], b, wu[1]))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n留出验证 {len(d)} 次配对: 均值 {d.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]")
    print(f"  更好 {np.mean(d>0):.1%} / 更差 {np.mean(d<0):.1%} / 打平 {np.mean(d==0):.1%}")
    Path("/workspace/exp_cb/single_gate_check.json").write_text(json.dumps(
        {"paired_precision": [float(np.mean(dp)), float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))],
         "paired_bpi": [float(np.mean(db)), float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
         "holdout_mean": float(d.mean()), "holdout_worse": float(np.mean(d < 0)),
         "holdout_n": len(d)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
