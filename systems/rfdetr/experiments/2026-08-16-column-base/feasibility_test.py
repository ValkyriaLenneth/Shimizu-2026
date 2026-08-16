#!/usr/bin/env python3
"""Test the one quantity today's selection could not have inflated.

The precision claim for the flip view did not survive holdout validation: the
gap fell from +0.054 to +0.027 with a confidence interval straddling zero once
selection and evaluation stopped sharing data. That number was chosen as the
maximum over 28 configurations, so its collapse is the expected outcome.

But the same run surfaced a quantity that was never optimised for, and therefore
carries no selection bias at all: how often the four-target constraint admits any
solution. Across 200 half-splits the current deliverable found a feasible
threshold triple in 28% of them and the flip configuration in 48%.

That matters more here than precision does. The previous round established that
on the full 45 images the constraint is satisfiable in only 25% of bootstrap
resamples -- "all four grades at 0.70" is a property of this particular sample,
not of the model -- and that is the single largest threat to the delivery claim.
A configuration that widens the region where the constraint can be met is buying
robustness in exactly the place the deliverable is thinnest.

This measures it directly and paired: on each bootstrap resample of the full 45
images, both configurations are asked whether any threshold triple meets all four
targets, and the two answers are compared on the same resample. Feasibility is a
binary outcome, so the paired disagreement counts (one feasible, the other not)
are what carry the evidence -- reported as a McNemar-style split rather than a
difference of rates.
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
from tta_fusion import (MEMBERS, BASE_W, GRID, NC, TARGET, OUT, DS,
                        fuse, tabulate, best)
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import read_targets

CFG = {"deliverable": ([(m, "id") for m in MEMBERS], [BASE_W[m] for m in MEMBERS], 0.20),
       "hflip":       ([(m, v) for m in MEMBERS for v in ("id", "hflip")],
                       [BASE_W[m] / 2 for m in MEMBERS for _ in range(2)], 0.40)}
NBOOT = 1000


def main():
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    store = {tuple(k.split("|")): v
             for k, v in torch.load(OUT / "views.pt", weights_only=False).items()}
    names = [p.name for p in imgs]
    tabs = {k: tabulate(fuse(store, ks, w, sizes, iou), targets, names)
            for k, (ks, w, iou) in CFG.items()}

    full = np.arange(len(names))
    for k in CFG:
        win = best(tabs[k], full)
        print(f"{k:12s} 全量 P={win[0]:.3f} R={win[2]:.3f} "
              f"B={win[3][0]:.3f} C={win[3][1]:.3f} D={win[3][2]:.3f} "
              f"阈值 {[GRID[i] for i in win[1]]}", flush=True)

    rng = np.random.default_rng(20260816)
    both = only_h = only_d = neither = 0
    for _ in range(NBOOT):
        idx = rng.integers(0, len(names), len(names))
        fh = best(tabs["hflip"], idx) is not None
        fd = best(tabs["deliverable"], idx) is not None
        both += fh and fd
        only_h += fh and not fd
        only_d += fd and not fh
        neither += not fh and not fd

    rh, rd = (both + only_h) / NBOOT, (both + only_d) / NBOOT
    print(f"\n{NBOOT} 次自助重采样中,四项 >=0.70 是否存在可行阈值:")
    print(f"  现交付可行 {rd:.1%}   +hflip 可行 {rh:.1%}")
    print(f"  配对分解: 两者都行 {both}  仅 hflip 行 {only_h}  仅现交付行 {only_d}  都不行 {neither}")
    disc = only_h + only_d
    if disc:
        # Under the null that the flip view changes nothing, each discordant
        # resample is an even coin flip; the exact binomial tail is the p-value.
        from math import comb
        p = sum(comb(disc, i) for i in range(only_h, disc + 1)) / 2 ** disc
        print(f"  不一致 {disc} 次,其中 hflip 占 {only_h} ({only_h/disc:.1%})  "
              f"精确二项 p = {p:.2e}")
        verdict = "站得住" if p < 0.05 else "落在噪声内"
    else:
        p, verdict = 1.0, "无差异"
    print(f"  结论: 可行率提升 {verdict}")
    (OUT / "feasibility.json").write_text(json.dumps(
        {"rate_deliverable": rd, "rate_hflip": rh, "both": both, "only_hflip": only_h,
         "only_deliverable": only_d, "neither": neither,
         "p_exact": float(p), "verdict": verdict}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
