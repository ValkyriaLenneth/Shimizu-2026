#!/usr/bin/env python3
"""Is the new fusion setting actually better than the shipped one?

The claim on record is that changing four inference parameters -- wbf_iou 0.40
to 0.20, weights 1:1 to 1:2 -- lifts precision from 0.302 to 0.329 at the same
four-target recall. That is a +0.027 gap read off a single evaluation of 45
images, and today three larger-looking gains than this one collapsed when they
were checked properly.

The unpaired bootstrap already run puts a 90% interval of [0.276, 0.533] around
the new setting, which is far too wide to separate the two on its own. But that
interval answers the wrong question: both settings are scored on the *same* 45
images, so most of that spread is shared between them -- an unusually easy
resample lifts both. What matters is the difference on each resample.

So this pairs them. Every bootstrap resample is scored under both settings, each
tuned independently inside that resample, and the paired difference is recorded.
If the difference stays positive across resamples, the parameter change is real
and separable from the sampling noise the two settings have in common. If it
straddles zero, then 0.329 and 0.302 are the same number seen twice, and the
delivery documentation has to say so.

No training, no seeds -- both members are shipped checkpoints, so this is the
one claim of today that seed variance cannot touch. Only the tuning is at issue.
"""
from __future__ import annotations
import itertools, json, sys
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
GRID = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MATCH_IOU, NC, TARGET = 0.229, 3, 0.70
NEW = (0.20, (1.0, 2.0), "max")     # candidate
OLD = (0.40, (1.0, 1.0), "avg")     # shipped 2026-08-04
NBOOT = 400
OUT = Path("/workspace/exp_cb/e19_paired")


def fuse(raw, sizes, iou, w, conf, floor=0.10):
    out = {}
    for name in raw[0]:
        bl, sl, ll = [], [], []
        for s in raw:
            r = s[name]
            sc = np.asarray(r["scores"], np.float32); keep = sc >= floor
            bl.append(np.asarray(r["boxes"], np.float32).reshape(-1, 4)[keep].tolist())
            sl.append(sc[keep].tolist())
            ll.append(np.asarray(r["classes"], np.int64)[keep].tolist())
        W, H = sizes[name]
        if not any(len(b) for b in bl):
            out[name] = []
            continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=list(w), iou_thr=iou,
                                        skip_box_thr=0.0, conf_type=conf)
        out[name] = [(int(c), float(x), tuple((np.asarray(bb) * np.array([W, H, W, H])).tolist()))
                     for bb, x, c in zip(b, s, l)]
    return out


def tabulate(fused, targets, names):
    """Per-image, per-class, per-threshold counts, so resampling is pure lookup.

    Bootstrapping over 400 resamples times 1 728 threshold triples times two
    settings is far too much work to redo box matching each time. Matching
    depends only on the image and the threshold, never on which resample the
    image landed in, so it is done once here and every resample afterwards is
    a sum over rows of this table.
    """
    G = len(GRID)
    tp = np.zeros((len(names), NC, G), np.int32)
    fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for i, n in enumerate(names):
        for gi, t in enumerate(GRID):
            total = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in fused[n] if s >= t]
            merge_counts(total, match_counts(targets[n], sel, MATCH_IOU, NC))
            for c in range(NC):
                tp[i, c, gi] = total[c]["tp"]; fp[i, c, gi] = total[c]["fp"]; fn[i, c, gi] = total[c]["fn"]
    return tp, fp, fn


def best_precision(tp, fp, fn, idx):
    """Highest precision among threshold triples meeting all four recall targets."""
    TP = tp[idx].sum(0); FP = fp[idx].sum(0); FN = fn[idx].sum(0)   # (NC, G)
    best = None
    for combo in itertools.product(range(len(GRID)), repeat=NC):
        t = np.array([TP[c, combo[c]] for c in range(NC)])
        f = np.array([FP[c, combo[c]] for c in range(NC)])
        m = np.array([FN[c, combo[c]] for c in range(NC)])
        den = t + m
        per = np.where(den > 0, t / np.maximum(den, 1), 0.0)
        if not np.all((den == 0) | (per >= TARGET)):
            continue
        st, sf, sm = t.sum(), f.sum(), m.sum()
        r = st / (st + sm) if st + sm else 0.0
        if r < TARGET:
            continue
        p = st / (st + sf) if st + sf else 0.0
        if best is None or p > best[0]:
            best = (p, combo)
    return best


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])

    raw = []
    for ck in MEMBERS:
        m = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        d = {}
        for p in imgs:
            with Image.open(p) as h:
                im = h.convert("RGB")
            det = m.predict(im, threshold=0.10)
            cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
            W, H = sizes[p.name]
            xy = np.asarray(det.xyxy).reshape(-1, 4)[keep]
            d[p.name] = {"boxes": np.clip(xy / np.array([W, H, W, H], np.float32), 0, 1).tolist(),
                         "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                         "classes": cls[keep].tolist()}
        raw.append(d); del m; torch.cuda.empty_cache()

    names = [p.name for p in imgs]
    tables = {}
    for tag, (iou, w, conf) in (("new", NEW), ("old", OLD)):
        f = fuse(raw, sizes, iou, w, conf)
        tables[tag] = tabulate(f, targets, names)
        full = best_precision(*tables[tag], np.arange(len(names)))
        print(f"{tag}: 全量 P = {full[0]:.3f}  阈值 "
              f"{tuple(GRID[i] for i in full[1])}", flush=True)

    rng = np.random.default_rng(20260816)
    diffs, pn, po = [], [], []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(names), len(names))
        a = best_precision(*tables["new"], idx)
        b = best_precision(*tables["old"], idx)
        if a and b:
            pn.append(a[0]); po.append(b[0]); diffs.append(a[0] - b[0])
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n配对自助 {len(d)} 次(同一重采样同时评两个配置):")
    print(f"  新配置中位 {np.median(pn):.3f} / 旧配置中位 {np.median(po):.3f}")
    print(f"  配对差中位 {np.median(d):+.3f}  95% 区间 [{lo:+.3f}, {hi:+.3f}]")
    print(f"  新配置更好的比例 {np.mean(d > 0):.1%}  (打平 {np.mean(d == 0):.1%})")
    verdict = "站得住" if lo > 0 else ("方向可信但未达显著" if np.mean(d > 0) >= 0.90 else "落在噪声内")
    print(f"\n结论: 参数改动 {verdict}")
    (OUT / "paired.json").write_text(json.dumps(
        {"n": len(d), "median_new": float(np.median(pn)), "median_old": float(np.median(po)),
         "median_diff": float(np.median(d)), "ci95": [float(lo), float(hi)],
         "p_new_better": float(np.mean(d > 0)), "verdict": verdict}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
