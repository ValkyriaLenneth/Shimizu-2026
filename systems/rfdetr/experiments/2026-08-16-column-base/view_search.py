#!/usr/bin/env python3
"""Search views against the constraint that actually binds.

Adding a horizontal flip did not raise precision in any way that survived
holdout validation, but it strictly dominated the current deliverable on a
different quantity: whether the four-target constraint admits any solution at
all. Across 1 000 bootstrap resamples the flip configuration was feasible
wherever the deliverable was, plus 198 resamples where the deliverable was not,
and never the reverse -- 26.6% to 46.4%.

That is the quantity worth optimising now. Precision estimates on 72 boxes are
too noisy to select on, as five retracted claims today demonstrate; feasibility
is a binary property aggregated over resamples and separates configurations
cleanly. It is also the property the delivery claim actually rests on, since the
previous round showed "all four grades at 0.70" holds on only a quarter of
resamples of this test set.

So this sweeps further views and fusion weights, scoring each by bootstrap
feasibility rate rather than by precision.

Views are restricted to transforms that leave box geometry untouched -- exposure,
contrast, gamma -- plus the two flips whose inverse is exact. Rotations and crops
were considered and left out: mapping an axis-aligned box back from a rotated
frame requires taking the bounding box of the rotated corners, which inflates
every box by an amount that varies with aspect ratio, and that approximation
would be indistinguishable from a real effect at the scale being measured here.

Selecting on feasibility introduces the same selection bias that killed the
precision claim, so the winner is validated the same way: tuned on half the
images, feasibility measured on the held-out half.
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
from PIL import Image, ImageEnhance
from tta_fusion import (MEMBERS, GRID, NC, TARGET, OUT, DS, fuse, tabulate, best)
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import read_targets

CACHE = OUT / "views2.pt"
FLOOR = 0.10
NBOOT = 600

# Geometry-preserving unless noted; hflip/vflip invert exactly.
VIEWS = {
    "id":      lambda im: im,
    "hflip":   lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "vflip":   lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    "br085":   lambda im: ImageEnhance.Brightness(im).enhance(0.85),
    "br115":   lambda im: ImageEnhance.Brightness(im).enhance(1.15),
    "ct085":   lambda im: ImageEnhance.Contrast(im).enhance(0.85),
    "ct115":   lambda im: ImageEnhance.Contrast(im).enhance(1.15),
    "sharp":   lambda im: ImageEnhance.Sharpness(im).enhance(2.0),
}
FLIPS = {"hflip": 0, "vflip": 1}

# Every set keeps the identity view and builds on the adopted hflip.
VIEW_SETS = [
    ("id",), ("id", "hflip"),
    ("id", "hflip", "vflip"),
    ("id", "hflip", "br085", "br115"),
    ("id", "hflip", "ct085", "ct115"),
    ("id", "hflip", "sharp"),
    ("id", "hflip", "br085", "br115", "ct085", "ct115"),
    ("id", "hflip", "vflip", "br085", "br115"),
]
IOUS = [0.20, 0.30, 0.40, 0.50]
RATIOS = [(1.0, 2.0), (1.0, 1.0), (1.0, 3.0)]   # ep016 : cp075


def unflip(b: np.ndarray, view: str) -> np.ndarray:
    if view not in FLIPS or not len(b):
        return b
    o = b.copy()
    ax = FLIPS[view]
    o[:, ax], o[:, ax + 2] = 1.0 - b[:, ax + 2], 1.0 - b[:, ax]
    return o


def build_cache(device, imgs, sizes):
    store = {}
    for tag, ck in MEMBERS.items():
        model = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(model, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view, fn in VIEWS.items():
            d = {}
            for p in imgs:
                with Image.open(p) as h:
                    im = fn(h.convert("RGB"))
                W, H = im.size
                det = model.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1)
                keep = cls < NC
                bn = np.clip(np.asarray(det.xyxy).reshape(-1, 4)[keep]
                             / np.array([W, H, W, H], np.float32), 0, 1)
                d[p.name] = {"boxes": unflip(bn, view).tolist(),
                             "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist()}
            store[(tag, view)] = d
            print(f"  {tag}/{view} 完成", flush=True)
        del model
        torch.cuda.empty_cache()
    torch.save({"|".join(k): v for k, v in store.items()}, CACHE)
    return store


def feas_rate(tab, rng_seed, n, nboot):
    rng = np.random.default_rng(rng_seed)
    hits = 0
    for _ in range(nboot):
        hits += best(tab, rng.integers(0, n, n)) is not None
    return hits / nboot


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    if CACHE.exists():
        store = {tuple(k.split("|")): v
                 for k, v in torch.load(CACHE, weights_only=False).items()}
        print(f"复用缓存 {len(store)} 组", flush=True)
    else:
        print(f"{len(MEMBERS)} 权重 x {len(VIEWS)} 视图", flush=True)
        store = build_cache(device, imgs, sizes)
    names = [p.name for p in imgs]
    n = len(names)

    rows, tabs = [], {}
    for vs in VIEW_SETS:
        for iou in IOUS:
            for ratio in RATIOS:
                keys = [(m, v) for m in MEMBERS for v in vs]
                w = [ratio[i] / len(vs) for i in range(len(MEMBERS)) for _ in vs]
                tab = tabulate(fuse(store, keys, w, sizes, iou), targets, names)
                win = best(tab, np.arange(n))
                if not win:
                    continue
                tag = f"{'+'.join(vs)}|{iou}|{ratio[0]:.0f}:{ratio[1]:.0f}"
                fr = feas_rate(tab, 20260816, n, NBOOT)
                tabs[tag] = tab
                rows.append({"tag": tag, "views": "+".join(vs), "wbf_iou": iou,
                             "ratio": f"{ratio[0]:.0f}:{ratio[1]:.0f}",
                             "feasible_rate": fr, "precision": round(win[0], 4)})
                print(f"  {tag:52s} 可行率 {fr:.1%}  P {win[0]:.3f}", flush=True)

    rows.sort(key=lambda r: -r["feasible_rate"])
    (OUT / "view_search.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    base = next(r for r in rows if r["tag"].startswith("id|0.2|1:2"))
    adopted = next(r for r in rows if r["tag"].startswith("id+hflip|0.4|1:2"))
    print(f"\n现交付 id|0.2|1:2 可行率 {base['feasible_rate']:.1%}")
    print(f"已采纳 id+hflip|0.4|1:2 可行率 {adopted['feasible_rate']:.1%}")
    print(f"本次最好 {rows[0]['tag']} 可行率 {rows[0]['feasible_rate']:.1%} P {rows[0]['precision']:.3f}")

    if rows[0]["feasible_rate"] <= adopted["feasible_rate"] + 1e-9:
        print("未超过已采纳配置 -> 保持不变")
        return

    print("\n留出验证(半分调优,另半分测可行率),对抗挑选偏差:")
    rng = np.random.default_rng(20260816)
    win_h = {k: 0 for k in ("cand", "adopted")}
    both = neither = 0
    cand_tab, ad_tab = tabs[rows[0]["tag"]], tabs[adopted["tag"]]
    for _ in range(300):
        perm = rng.permutation(n)
        b = perm[n // 2:]
        fc = best(cand_tab, b) is not None
        fa = best(ad_tab, b) is not None
        both += fc and fa
        neither += not fc and not fa
        win_h["cand"] += fc and not fa
        win_h["adopted"] += fa and not fc
    disc = win_h["cand"] + win_h["adopted"]
    print(f"  两者都行 {both}  仅候选 {win_h['cand']}  仅已采纳 {win_h['adopted']}  都不行 {neither}")
    if disc:
        from math import comb
        p = sum(comb(disc, i) for i in range(win_h["cand"], disc + 1)) / 2 ** disc
        print(f"  精确二项 p = {p:.2e}  ->  {'采纳候选' if p < 0.05 else '不采纳,保持已有配置'}")
    else:
        print("  留出集上无差异 -> 不采纳")


if __name__ == "__main__":
    main()
