#!/usr/bin/env python3
"""Re-choose the operating point against the cost the client actually bears.

Every threshold search on this project has maximised precision subject to the
four-target recall constraint. That objective was never examined, and the
inspector-cost decomposition shows it is close to the wrong one: at a 5% damage
prevalence, 92% of the boxes a person reviews come from false alarms on sound
photographs and only 8% from the damaged ones that precision describes. The
search has been optimising the small term.

Thresholds move both terms at once, so the operating point that minimises total
review burden need not be the one that maximises precision. This searches the
same grid under the same four-target constraint, but scores each candidate by
boxes reviewed per 100 photographs, and reports what the change costs in
precision -- which is real, just far smaller in the client's units.

Prevalence is reported across a range rather than assumed, since the optimum can
move with it and the client knows their own.

Nothing is retrained: this is the delivered four-view configuration, re-cut.
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
from tta_fusion import MEMBERS, GRID, NC, DS, OUT, fuse, tabulate, best
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import read_targets

SOUND = Path("/workspace/sound_20260807/column_base")
VIEWS, RATIO, IOU, CONF, FLOOR = ("id", "hflip"), (1.0, 2.0), 0.40, "max", 0.10
MATCH_IOU, TARGET, GRADES = 0.229, 0.70, "BCD"
PREVS = [0.02, 0.05, 0.10, 0.20, 0.30]


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def predict(device, paths, sizes):
    store = {}
    for tag, ck in MEMBERS.items():
        m = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in VIEWS:
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


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sound = sorted(p for p in SOUND.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, ssz, targets = {}, {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    for p in sound:
        with Image.open(p) as h:
            ssz[p.name] = h.size
    keys = [(m, v) for m in MEMBERS for v in VIEWS]
    w = [RATIO[i] / len(VIEWS) for i in range(len(MEMBERS)) for _ in VIEWS]
    ft = fuse(predict(device, imgs, sizes), keys, w, sizes, IOU)
    fs = fuse(predict(device, sound, ssz), keys, w, ssz, IOU)
    names = sorted(targets)
    tab = tabulate(ft, targets, names)
    ngt = sum(len(v) for v in targets.values())
    gt_per = ngt / len(names)

    # Sound-side boxes per image at every threshold triple, precomputed.
    rows = []
    TP, FP, FN = (t.sum(0) for t in tab)
    for combo in itertools.product(range(len(GRID)), repeat=NC):
        t = np.array([TP[c, combo[c]] for c in range(NC)])
        f = np.array([FP[c, combo[c]] for c in range(NC)])
        m = np.array([FN[c, combo[c]] for c in range(NC)])
        den = t + m
        per = np.where(den > 0, t / np.maximum(den, 1), 0.0)
        if not np.all((den == 0) | (per >= TARGET)):
            continue
        st, sf, sm = t.sum(), f.sum(), m.sum()
        r = st / max(st + sm, 1)
        if r < TARGET:
            continue
        p_ = st / max(st + sf, 1)
        thr = [GRID[i] for i in combo]
        kept = [d for dets in fs.values() for d in dets if d[1] >= thr[d[0]]]
        fired = sum(1 for dets in fs.values() if any(x[1] >= thr[x[0]] for x in dets))
        rows.append({"thr": thr, "precision": p_, "recall": r,
                     "per": per.tolist(),
                     "sound_bpi": len(kept) / len(fs), "fire": fired / len(fs)})
    print(f"四项达标点 {len(rows)} 个(冻结集 {len(names)} 图 / {ngt} 框,"
          f"健全图 {len(fs)} 张)\n")

    def cost(r, prev):
        return (100 * prev * gt_per * r["recall"] / max(r["precision"], 1e-9)
                + 100 * (1 - prev) * r["sound_bpi"])

    cur = max(rows, key=lambda r: r["precision"])
    print(f"现行目标(precision 最大): 阈值 {cur['thr']}  P {cur['precision']:.3f}  "
          f"R {cur['recall']:.3f}  健全 {cur['fire']:.0%} / {cur['sound_bpi']:.2f} 框/张")
    print(f"\n{'损伤占比':>8} {'现行点过目框':>12} {'最优点过目框':>12} {'省下':>8}  "
          f"{'最优阈值':>18} {'P':>6} {'健全框/张':>9}")
    out = {}
    for prev in PREVS:
        bestr = min(rows, key=lambda r: cost(r, prev))
        c0, c1 = cost(cur, prev), cost(bestr, prev)
        print(f"{prev:8.0%} {c0:12.1f} {c1:12.1f} {c0-c1:7.1f}  "
              f"{str(bestr['thr']):>18} {bestr['precision']:6.3f} {bestr['sound_bpi']:9.2f}")
        out[str(prev)] = {"current_cost": c0, "best_cost": c1, "best": bestr}
    Path("/workspace/exp_cb/operating_point.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
