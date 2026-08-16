#!/usr/bin/env python3
"""Separate the parameter change from the tuning noise around it.

The paired bootstrap just run re-tuned thresholds inside every resample, and its
95% interval straddled zero. But that test charges the parameter change for two
different sources of variation at once: the fusion setting itself, and the fact
that a 1 728-cell threshold search lands somewhere different in each resample.
The second has nothing to do with the change being tested.

Both settings happen to select the identical threshold triple on the full data
(0.07 / 0.15 / 0.05), so the two can be compared with thresholds held fixed.
Then the only thing differing between the paired arms is wbf_iou and the member
weights, which is exactly the claim. This is the more powerful test, and if the
difference is real anywhere it will show here.

The same run also reports how often the four-target constraint is satisfiable at
all. The first bootstrap found a feasible point in only 99 of 400 resamples,
which -- if it holds up -- says something more consequential than any precision
number: that "all four grades at 0.70" is a property of this particular sample of
45 images rather than an established property of the model. That belongs in the
client conversation regardless of how the precision comparison lands.
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
FIXED = (0.07, 0.15, 0.05)
MATCH_IOU, NC, TARGET, NBOOT = 0.229, 3, 0.70, 2000
NEW = (0.20, (1.0, 2.0), "max")
OLD = (0.40, (1.0, 1.0), "avg")
OUT = Path("/workspace/exp_cb/e19_paired")


def fuse(raw, sizes, iou, w, conf, floor=0.10):
    out = {}
    for name in raw[0]:
        bl, sl, ll = [], [], []
        for s in raw:
            r = s[name]; sc = np.asarray(r["scores"], np.float32); keep = sc >= floor
            bl.append(np.asarray(r["boxes"], np.float32).reshape(-1, 4)[keep].tolist())
            sl.append(sc[keep].tolist()); ll.append(np.asarray(r["classes"], np.int64)[keep].tolist())
        W, H = sizes[name]
        if not any(len(b) for b in bl):
            out[name] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=list(w), iou_thr=iou,
                                        skip_box_thr=0.0, conf_type=conf)
        out[name] = [(int(c), float(x), tuple((np.asarray(bb) * np.array([W, H, W, H])).tolist()))
                     for bb, x, c in zip(b, s, l)]
    return out


def tabulate(fused, targets, names):
    G = len(GRID)
    tp = np.zeros((len(names), NC, G), np.int32); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for i, n in enumerate(names):
        for gi, t in enumerate(GRID):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in fused[n] if s >= t]
            merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
            for c in range(NC):
                tp[i, c, gi] = tot[c]["tp"]; fp[i, c, gi] = tot[c]["fp"]; fn[i, c, gi] = tot[c]["fn"]
    return tp, fp, fn


def at_fixed(tab, idx):
    tp, fp, fn = tab
    gi = [GRID.index(t) for t in FIXED]
    t = np.array([tp[idx, c, gi[c]].sum() for c in range(NC)])
    f = np.array([fp[idx, c, gi[c]].sum() for c in range(NC)])
    m = np.array([fn[idx, c, gi[c]].sum() for c in range(NC)])
    st, sf, sm = t.sum(), f.sum(), m.sum()
    return (st / (st + sf) if st + sf else 0.0), (st / (st + sm) if st + sm else 0.0)


def feasible_any(tab, idx):
    tp, fp, fn = tab
    TP = tp[idx].sum(0); FP = fp[idx].sum(0); FN = fn[idx].sum(0)
    for combo in itertools.product(range(len(GRID)), repeat=NC):
        t = np.array([TP[c, combo[c]] for c in range(NC)])
        m = np.array([FN[c, combo[c]] for c in range(NC)])
        den = t + m
        per = np.where(den > 0, t / np.maximum(den, 1), 0.0)
        if not np.all((den == 0) | (per >= TARGET)):
            continue
        if t.sum() / max(t.sum() + m.sum(), 1) >= TARGET:
            return True
    return False


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h: sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    raw = []
    for ck in MEMBERS:
        m = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"): ctx.device = torch.device(device)
        d = {}
        for p in imgs:
            with Image.open(p) as h: im = h.convert("RGB")
            det = m.predict(im, threshold=0.10)
            cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
            W, H = sizes[p.name]; xy = np.asarray(det.xyxy).reshape(-1, 4)[keep]
            d[p.name] = {"boxes": np.clip(xy / np.array([W, H, W, H], np.float32), 0, 1).tolist(),
                         "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                         "classes": cls[keep].tolist()}
        raw.append(d); del m; torch.cuda.empty_cache()

    names = [p.name for p in imgs]
    tabs = {tag: tabulate(fuse(raw, sizes, *cfg), targets, names)
            for tag, cfg in (("new", NEW), ("old", OLD))}
    full = np.arange(len(names))
    for tag in ("new", "old"):
        p, r = at_fixed(tabs[tag], full)
        print(f"{tag} @ 固定阈值 {FIXED}: P = {p:.3f}  R = {r:.3f}", flush=True)

    rng = np.random.default_rng(20260816)
    diffs, feas_new = [], 0
    for _ in range(NBOOT):
        idx = rng.integers(0, len(names), len(names))
        pn, _ = at_fixed(tabs["new"], idx)
        po, _ = at_fixed(tabs["old"], idx)
        diffs.append(pn - po)
        if len(diffs) <= 400:
            feas_new += feasible_any(tabs["new"], idx)
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n固定阈值配对自助 {NBOOT} 次(只有融合参数不同):")
    print(f"  配对差均值 {d.mean():+.4f}  中位 {np.median(d):+.4f}")
    print(f"  95% 区间 [{lo:+.4f}, {hi:+.4f}]")
    print(f"  新配置更好 {np.mean(d > 0):.1%} / 打平 {np.mean(d == 0):.1%} / 更差 {np.mean(d < 0):.1%}")
    verdict = "站得住" if lo > 0 else ("方向一致但未达显著" if np.mean(d > 0) >= 0.90 else "落在噪声内")
    print(f"  结论: {verdict}")
    print(f"\n四项达标点在 400 次重采样中存在 {feas_new}/400 = {feas_new/4:.0f}%")
    print("  (换一批同分布的 45 张测试图,四项全达标未必还能成立)")
    (OUT / "paired_fixed.json").write_text(json.dumps(
        {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
         "p_new_better": float(np.mean(d > 0)), "verdict": verdict,
         "feasible_rate": feas_new / 400}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
