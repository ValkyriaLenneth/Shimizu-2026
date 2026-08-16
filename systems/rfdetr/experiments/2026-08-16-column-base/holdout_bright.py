#!/usr/bin/env python3
"""The last candidate: does adding exposure views buy precision, or look like it?

The view search finished 96 configurations. None exceeded the adopted setting on
feasibility -- 46.5% is a ceiling that horizontal flip alone reaches and no
combination of views passes. But several configurations tie at that ceiling, and
among them the two exposure views score a higher point-estimate precision than
the adopted setting: 0.415 against 0.383.

That is exactly the shape of the claim that failed earlier today. A precision
maximum picked out of a large sweep and read off the same 45 images that picked
it is not evidence; the flip view's own 0.383 looked significant under a paired
bootstrap and collapsed under holdout validation, because the bootstrap resamples
images but not the act of selection.

So the same test decides this one. Both configurations are tuned on a random half
of the images and scored on the other half, paired on identical splits, and only
splits where both admit a feasible tuning point are counted. If the gap survives,
the exposure views join the delivered configuration. If it does not, the adopted
setting stands and this is recorded as the sixth retraction.

Predictions for all eight views are already cached, so this costs no GPU time.
"""
from __future__ import annotations
import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from view_search import CACHE, VIEWS
from tta_fusion import MEMBERS, GRID, NC, DS, fuse, tabulate, best
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import read_targets

RATIO = (1.0, 2.0)
IOU = 0.40
CFG = {"adopted (id+hflip)": ("id", "hflip"),
       "candidate (+br)":    ("id", "hflip", "br085", "br115")}
NSPLIT = 400


def at(tab, idx, combo):
    tp, fp, fn = tab
    t = sum(tp[idx, c, combo[c]].sum() for c in range(NC))
    f = sum(fp[idx, c, combo[c]].sum() for c in range(NC))
    return t / max(t + f, 1)


def main():
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    store = {tuple(k.split("|")): v
             for k, v in torch.load(CACHE, weights_only=False).items()}
    names = [p.name for p in imgs]
    n = len(names)

    tabs = {}
    for label, views in CFG.items():
        keys = [(m, v) for m in MEMBERS for v in views]
        w = [RATIO[i] / len(views) for i in range(len(MEMBERS)) for _ in views]
        tabs[label] = tabulate(fuse(store, keys, w, sizes, IOU), targets, names)
        win = best(tabs[label], np.arange(n))
        print(f"{label:22s} 全量 P = {win[0]:.3f}  阈值 {[GRID[i] for i in win[1]]}", flush=True)

    rng = np.random.default_rng(20260816)
    d, both = [], 0
    for _ in range(NSPLIT):
        perm = rng.permutation(n)
        a, b = perm[:n // 2], perm[n // 2:]
        wins = {k: best(tabs[k], a) for k in CFG}
        if not all(wins.values()):
            continue
        both += 1
        d.append(at(tabs["candidate (+br)"], b, wins["candidate (+br)"][1])
                 - at(tabs["adopted (id+hflip)"], b, wins["adopted (id+hflip)"][1]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n留出验证 {NSPLIT} 次划分,两者均可调优的 {both} 次配对:")
    print(f"  precision 差 中位 {np.median(d):+.4f}  95% 区间 [{lo:+.4f}, {hi:+.4f}]")
    print(f"  候选更好 {np.mean(d > 0):.1%}")
    adopt = lo > 0
    print(f"  结论: {'采纳亮度视图' if adopt else '不采纳 —— 0.415 是挑选偏差,已采纳配置保持不变'}")
    Path("/workspace/exp_cb/e22_holdout.json").write_text(json.dumps(
        {"n_paired": both, "median_diff": float(np.median(d)),
         "ci95": [float(lo), float(hi)], "p_better": float(np.mean(d > 0)),
         "adopt": bool(adopt)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
