#!/usr/bin/env python3
"""Test the method on data that did not produce it.

The column-base finding is that adding a horizontal-flip view to the WBF
ensemble roughly doubles how often the four-target constraint admits a solution
-- 26.6% to 46.4% of bootstrap resamples, with 198 discordant resamples all
pointing the same way. It is the one result this week that survived every check.

But it was found on 45 images and 72 boxes, and it was found by searching that
same split. Every guard applied so far -- holdout tuning, paired bootstrap,
selecting on a quantity that was never optimised -- controls for how the number
was extracted from those 45 images. None of them can rule out that the effect is
a property of this particular test set.

The only evidence that can is a replication on data the search never touched.
Brace is the right choice: a different structural element, a different 58-image
test split, different training runs behind its two shipped checkpoints, and the
same B/C/D grading and delivery protocol. If flipping raises feasibility there
too, the mechanism is about how WBF converts disagreement into coverage, which is
element-independent. If it does not, the column-base result is a local artifact
and the freeze document has to be amended to say so.

This is method validation for the column-base delivery, not brace work: no brace
configuration is being tuned or proposed here.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

DS = Path("/workspace/Shimizu-2026/data/rfdetr_brace_bcd_20260725_test_as_valid")
FRZ = Path("/workspace/handoff_20260815_brace_recall_freeze/checkpoints")
MEMBERS = {"brl032": FRZ / "brace_brl_ignore035_epoch_032.pth",
           "cps058": FRZ / "brace_cpsym33_epoch_058.pth"}
RATIO = (1.0, 2.0)          # same member weighting the column-base config uses
GRID = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
IOUS = [0.20, 0.30, 0.40, 0.50]
MATCH_IOU, NC, TARGET, FLOOR, NBOOT = 0.229, 3, 0.70, 0.10, 800
OUT = Path("/workspace/exp_cb/e23_brace")
import itertools


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def predict(device, paths, sizes):
    store = {}
    for tag, ck in MEMBERS.items():
        m = from_checkpoint_matched(str(ck), device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in ("id", "hflip"):
            d = {}
            for p in paths:
                with Image.open(p) as h:
                    im = h.convert("RGB")
                if view == "hflip":
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                W, H = im.size
                det = m.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
                bn = np.clip(np.asarray(det.xyxy).reshape(-1, 4)[keep]
                             / np.array([W, H, W, H], np.float32), 0, 1)
                d[p.name] = {"boxes": unflip(bn, view).tolist(),
                             "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist()}
            store[(tag, view)] = d
        del m; torch.cuda.empty_cache()
    return store


def fuse(store, views, sizes, iou):
    keys = [(m, v) for m in MEMBERS for v in views]
    w = [RATIO[i] / len(views) for i in range(len(MEMBERS)) for _ in views]
    out = {}
    for name in sizes:
        bl = [np.asarray(store[k][name]["boxes"], np.float32).reshape(-1, 4).tolist() for k in keys]
        sl = [list(store[k][name]["scores"]) for k in keys]
        ll = [list(store[k][name]["classes"]) for k in keys]
        W, H = sizes[name]
        if not any(len(b) for b in bl):
            out[name] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=w, iou_thr=iou,
                                        skip_box_thr=0.0, conf_type="max")
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


def best(tab, idx):
    tp, fp, fn = tab
    TP, FP, FN = tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0)
    win = None
    for combo in itertools.product(range(len(GRID)), repeat=NC):
        t = np.array([TP[c, combo[c]] for c in range(NC)])
        f = np.array([FP[c, combo[c]] for c in range(NC)])
        m = np.array([FN[c, combo[c]] for c in range(NC)])
        den = t + m
        per = np.where(den > 0, t / np.maximum(den, 1), 0.0)
        if not np.all((den == 0) | (per >= TARGET)):
            continue
        st, sf, sm = t.sum(), f.sum(), m.sum()
        if st / max(st + sm, 1) < TARGET:
            continue
        p = st / max(st + sf, 1)
        if win is None or p > win[0]:
            win = (p, combo)
    return win


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    n = len(imgs)
    print(f"ブレース 冻结测试集: {n} 图 / {sum(len(v) for v in targets.values())} 框", flush=True)
    store = predict(device, imgs, sizes)
    names = [p.name for p in imgs]

    tabs, res = {}, []
    for views in (("id",), ("id", "hflip")):
        for iou in IOUS:
            tab = tabulate(fuse(store, views, sizes, iou), targets, names)
            tabs[("+".join(views), iou)] = tab
            win = best(tab, np.arange(n))
            res.append({"views": "+".join(views), "iou": iou,
                        "precision": None if not win else round(win[0], 4)})
            print(f"  {'+'.join(views):10s} iou {iou}: "
                  f"P {'--' if not win else f'{win[0]:.3f}'}", flush=True)

    # Compare at each wbf_iou so the flip is isolated from the fusion setting.
    print(f"\n{NBOOT} 次自助重采样,四项 >=0.70 是否存在可行阈值:")
    print(f"{'wbf_iou':>8} {'仅原图':>8} {'+翻转':>8} {'都行':>6} {'仅翻转':>7} {'仅原图独有':>10} {'p':>10}")
    from math import comb
    out = {}
    for iou in IOUS:
        rng = np.random.default_rng(20260816)
        both = oh = od = 0
        for _ in range(NBOOT):
            idx = rng.integers(0, n, n)
            fh = best(tabs[("id+hflip", iou)], idx) is not None
            fd = best(tabs[("id", iou)], idx) is not None
            both += fh and fd; oh += fh and not fd; od += fd and not fh
        disc = oh + od
        p = (sum(comb(disc, i) for i in range(oh, disc + 1)) / 2 ** disc) if disc else 1.0
        print(f"{iou:8.2f} {(both+od)/NBOOT:7.1%} {(both+oh)/NBOOT:7.1%} "
              f"{both:6d} {oh:7d} {od:10d} {p:10.2e}")
        out[str(iou)] = {"rate_id": (both + od) / NBOOT, "rate_hflip": (both + oh) / NBOOT,
                         "both": both, "only_hflip": oh, "only_id": od, "p": float(p)}
    (OUT / "replication.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    wins = sum(1 for v in out.values() if v["only_hflip"] > v["only_id"])
    print(f"\n{wins}/{len(IOUS)} 个 wbf_iou 上翻转的可行率更高 -> "
          f"{'柱脚结论在独立数据上复现' if wins >= 3 else '未复现,柱脚结论需标注为局部'}")


if __name__ == "__main__":
    main()
