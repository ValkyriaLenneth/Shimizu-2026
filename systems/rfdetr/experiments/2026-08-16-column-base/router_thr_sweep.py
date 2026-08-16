#!/usr/bin/env python3
"""Lower the router's own threshold so the gate covers more images.

The gate fails to act on 13-14% of photographs because the router finds no column
base in them at its threshold of 0.30, and those photographs are the ones that
fire most: every one of the four ungated sound images produces a box, against 84%
of the gated ones. Raising coverage therefore attacks the part of the false-alarm
rate the gate currently cannot reach.

The router's threshold is a free parameter of the delivered pipeline, not of the
router's own training, so moving it costs nothing and risks nothing that cannot
be measured here. Two things could go wrong and both are reported: a lower
threshold may locate a *wrong* region, which would gate away real damage and cost
recall, and it may return a larger union of boxes, which would make the gate
permissive enough to stop removing anything.

The delivered 0.30 is included so the comparison is against the shipped
configuration rather than an idealised one.
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

THRS = [0.30, 0.20, 0.15, 0.10, 0.05]
GATE = 0.5


def member_boxes_at(device, paths, sizes, rthr):
    m = from_checkpoint_matched(RG.ROUTER, device=device, verbose=False)
    ctx = getattr(m, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    out = {}
    for p in paths:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = m.predict(im, threshold=rthr)
        cls = np.asarray(det.class_id).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        sel = cls == RG.CB_CLASS
        if not sel.any():
            out[p.name] = None
            continue
        b = xy[sel]
        W, H = sizes[p.name]
        x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
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

    print(f"{'路由阈值':>8} {'测试覆盖':>9} {'健全覆盖':>9} {'构件框占幅':>11} "
          f"{'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'箱/张':>7}")
    rows = []
    for rthr in THRS:
        mt = member_boxes_at(device, imgs, sizes, rthr)
        ms = member_boxes_at(device, sound, ssz, rthr)
        cov_t = sum(1 for v in mt.values() if v) / len(mt)
        cov_s = sum(1 for v in ms.values() if v) / len(ms)
        # A gate that covers the whole frame removes nothing; report the size.
        frac = [((v[2]-v[0])*(v[3]-v[1])) / (ssz[n][0]*ssz[n][1])
                for n, v in ms.items() if v]
        gt_ = RG.apply_gate(pt, mt, GATE)
        gs_ = RG.apply_gate(ps, ms, GATE)
        best = None
        for combo in itertools.product(GRID, repeat=NC):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            for n, dd in gt_.items():
                sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dd if s >= combo[c]]
                merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
            fn = sum(v["fn"] for v in tot.values())
            p_, r_, _ = metric(tp, fp, fn)
            per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
            if r_ >= RG.TARGET and all(v >= RG.TARGET for v in per) and (best is None or p_ > best[0]):
                fired = sum(1 for d in gs_.values() if any(x[1] >= combo[x[0]] for x in d))
                boxes = sum(len([x for x in d if x[1] >= combo[x[0]]]) for d in gs_.values())
                best = (p_, r_, per, list(combo), fired / len(gs_), boxes / len(gs_))
        if not best:
            print(f"{rthr:8.2f} {cov_t:8.0%} {cov_s:8.0%} {np.median(frac):10.0%}   无四项达标点")
            continue
        p_, r_, per, combo, fire, bpi = best
        print(f"{rthr:8.2f} {cov_t:8.0%} {cov_s:8.0%} {np.median(frac):10.0%} "
              f"{p_:7.3f} {r_:7.3f} {per[0]:6.3f} {per[1]:6.3f} {per[2]:6.3f} "
              f"{fire:5.0%} {bpi:7.2f}")
        rows.append({"router_thr": rthr, "cov_test": cov_t, "cov_sound": cov_s,
                     "member_frac": float(np.median(frac)), "precision": p_,
                     "recall": r_, "per": per, "thr": combo, "fire": fire, "bpi": bpi})
    print("\n判读: 覆盖率上升而误报下降 -> 采纳;"
          "\n      构件框占幅趋近 100% -> 门控名存实亡,收益是假的;"
          "\n      recall 下降 -> 低阈值定位到了错误区域。")
    Path("/workspace/exp_cb/router_thr_sweep.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
