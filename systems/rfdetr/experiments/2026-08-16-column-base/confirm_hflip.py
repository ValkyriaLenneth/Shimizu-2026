#!/usr/bin/env python3
"""Two things the paired bootstrap on the winner cannot answer.

Adding a horizontal-flip view at wbf_iou 0.40 raised precision from 0.329 to
0.384, and the paired bootstrap put that gap outside zero. But that winner was
chosen as the maximum over 28 swept configurations and then tested on the same
45 images that chose it; the bootstrap resamples the images, not the selection,
so it cannot say how much of the gap is the selection itself. The plateau across
wbf_iou 0.40-0.55 argues against a lucky cell, but that is a shape argument, not
a measurement.

Check one, selection bias: tune on a random half of the images and score the
chosen configuration on the held-out half, for both the flip setting and the
current deliverable. Selection and evaluation no longer share data, so whatever
survives is not an artifact of picking the maximum.

Check two, the cost side: the flip view doubles the boxes entering fusion, and
this task's real failure mode is firing on sound column bases -- 96% of the
client's 28 sound photographs already trigger the current deliverable. A
precision gain on damaged images that arrives with more false alarms on sound
ones is not an improvement worth shipping, and the frozen test split contains no
sound images to reveal it.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
from tta_fusion import (MEMBERS, BASE_W, GRID, NC, TARGET, OUT, DS, SOUND,
                        fuse, tabulate, best, at, predict_views, render, unrender)
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import read_targets

CFG = {"deliverable": ([(m, "id") for m in MEMBERS], [BASE_W[m] for m in MEMBERS], 0.20),
       "hflip":       ([(m, v) for m in MEMBERS for v in ("id", "hflip")],
                       [BASE_W[m] / 2 for m in MEMBERS for _ in range(2)], 0.40)}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
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

    # Both configurations must be tuned on the *same* split for the difference to
    # mean anything. They are not feasible on the same splits -- the four-target
    # constraint fails more often for one than the other -- so a split is only
    # counted when both admit a tuning point, and how often each is feasible at
    # all is reported separately, since that is its own result.
    rng = np.random.default_rng(20260816)
    held = {k: [] for k in CFG}
    feas = {k: 0 for k in CFG}
    paired = []
    for _ in range(200):
        perm = rng.permutation(len(names))
        a, b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        wins = {k: best(tabs[k], a) for k in CFG}
        for k in CFG:
            if wins[k]:
                feas[k] += 1
                held[k].append(at(tabs[k], b, wins[k][1]))
        if all(wins.values()):
            paired.append(at(tabs["hflip"], b, wins["hflip"][1])
                          - at(tabs["deliverable"], b, wins["deliverable"][1]))
    print("半分调优 / 半分评估 200 次(挑选与评估不共享数据):")
    for k in CFG:
        v = np.array(held[k])
        print(f"  {k:12s} 留出中位 {np.median(v):.3f}   四项可达标 {feas[k]}/200 = {feas[k]/2:.0f}%")
    d = np.array(paired)
    print(f"  同划分配对 n={len(d)}")
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  留出集上的差 中位 {np.median(d):+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  "
          f"更好 {np.mean(d>0):.1%}")

    sound = sorted(SOUND.glob("*.jpg"))
    if not sound:
        print("\n无健全图,跳过误报检查")
        return
    ssz = {}
    for p in sound:
        with Image.open(p) as h:
            ssz[p.name] = h.size
    sstore = predict_views(device, sound, ssz)
    print(f"\n健全图 {len(sound)} 张的误报(阈值取各自全量最优):")
    out = {}
    for k, (ks, w, iou) in CFG.items():
        win = best(tabs[k], np.arange(len(names)))
        thr = [GRID[i] for i in win[1]]
        f = fuse(sstore, ks, w, ssz, iou)
        fired = sum(1 for d_ in f.values() if any(x[1] >= thr[x[0]] for x in d_))
        boxes = sum(len([x for x in d_ if x[1] >= thr[x[0]]]) for d_ in f.values())
        print(f"  {k:12s} P={win[0]:.3f} 阈值 {thr}  发火 {fired}/{len(sound)} = "
              f"{fired/len(sound):.0%},{boxes/len(sound):.2f} 框/张")
        out[k] = {"precision": win[0], "thresholds": thr,
                  "fire_rate": fired / len(sound), "boxes_per_image": boxes / len(sound)}
    (OUT / "confirm.json").write_text(json.dumps(
        {"holdout_median_diff": float(np.median(d)), "holdout_ci95": [float(lo), float(hi)],
         "sound": out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
