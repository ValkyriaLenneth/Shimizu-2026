#!/usr/bin/env python3
"""Use the router's other four classes as evidence against a detection.

The gate keeps a box when it overlaps the column base the router found. That
treats the router's other four outputs -- ceiling, wall, RC column, brace -- as
silence, when they are actually evidence: a damage box sitting inside a region
the router calls a wall is wrong-place on stronger grounds than merely being
outside the column base, because something else has claimed that pixel.

This matters where the current gate is weakest. It does nothing on the 13-14% of
photographs where no column base is found, and those are the images that fire
most -- every one of the four ungated sound photographs produces a box. A
negative gate does not need the column base to be located, so it can act exactly
there.

Two variants, both keeping the existing positive gate:

  negative-only   drop a box that lies mostly inside another class's region
  combined        the positive gate where a member was found, the negative gate
                  everywhere else

Reported against the delivered configuration. A box can legitimately straddle a
boundary -- a column base sits against a wall -- so the overlap required is swept
rather than assumed, and recall is watched for exactly that failure.
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

OTHER = [0, 1, 2, 3]          # 天井 / 壁类 / RC柱 / ブレース
NEG_OVERLAPS = [0.9, 0.8, 0.7, 0.6]


def router_regions(device, paths, sizes):
    """Column-base box (single, dilated) plus the other classes' boxes."""
    m = from_checkpoint_matched(RG.ROUTER, device=device, verbose=False)
    ctx = getattr(m, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    member, others = {}, {}
    for p in paths:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = m.predict(im, threshold=RG.ROUTER_THR)
        cls = np.asarray(det.class_id).reshape(-1)
        conf = np.asarray(det.confidence).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        sel = cls == RG.CB_CLASS
        if sel.any():
            b = xy[sel][int(np.argmax(conf[sel]))]
            W, H = sizes[p.name]
            dw, dh = (b[2]-b[0]) * RG.MARGIN, (b[3]-b[1]) * RG.MARGIN
            member[p.name] = [max(0, b[0]-dw), max(0, b[1]-dh), min(W, b[2]+dw), min(H, b[3]+dh)]
        else:
            member[p.name] = None
        o = cls == cls  # all
        keep = np.isin(cls, OTHER)
        others[p.name] = xy[keep].tolist()
    del m; torch.cuda.empty_cache()
    return member, others


def inside(box, region):
    x1, y1 = max(box[0], region[0]), max(box[1], region[1])
    x2, y2 = min(box[2], region[2]), min(box[3], region[3])
    i = max(0.0, x2-x1) * max(0.0, y2-y1)
    a = (box[2]-box[0]) * (box[3]-box[1])
    return i / a if a > 0 else 0.0


def apply_neg(fused, member, others, ov, only_when_no_member):
    out = {}
    for name, dets in fused.items():
        mb = member.get(name)
        if only_when_no_member and mb is not None:
            out[name] = dets; continue
        keep = []
        for c, s, b in dets:
            if any(inside(b, r) >= ov for r in others.get(name, [])):
                continue
            keep.append((c, s, b))
        out[name] = keep
    return out


def score(ft, fs, targets, n_sound):
    best = None
    for combo in itertools.product(GRID, repeat=NC):
        tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
        for n, dd in ft.items():
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dd if s >= combo[c]]
            merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
        tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
        fn = sum(v["fn"] for v in tot.values())
        p_, r_, _ = metric(tp, fp, fn)
        per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
        if r_ >= RG.TARGET and all(v >= RG.TARGET for v in per):
            bpi = sum(len([x for x in d if x[1] >= combo[x[0]]]) for d in fs.values()) / n_sound
            if best is None or bpi < best[-1]:
                fired = sum(1 for d in fs.values() if any(x[1] >= combo[x[0]] for x in d))
                best = (p_, r_, per, list(combo), fired / n_sound, bpi)
    return best


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
    mt, ot = router_regions(device, imgs, sizes)
    ms, os_ = router_regions(device, sound, ssz)
    gt0, gs0 = RG.apply_gate(pt, mt, 0.5), RG.apply_gate(ps, ms, 0.5)

    base = score(gt0, gs0, targets, len(ssz))
    print(f"{'配置':30s} {'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'箱/张':>7}")
    print(f"{'现交付(仅正门控)':30s} {base[0]:7.3f} {base[1]:7.3f} {base[2][0]:6.3f} "
          f"{base[2][1]:6.3f} {base[2][2]:6.3f} {base[4]:5.0%} {base[5]:7.2f}")
    rows = []
    for ov in NEG_OVERLAPS:
        for tag, only in (("仅未定位图上加负门控", True), ("全图加负门控", False)):
            ft = apply_neg(gt0, mt, ot, ov, only)
            fs = apply_neg(gs0, ms, os_, ov, only)
            b = score(ft, fs, targets, len(ssz))
            if not b:
                print(f"{f'{tag} ov={ov}':30s}   无四项达标点"); continue
            print(f"{f'{tag} ov={ov}':30s} {b[0]:7.3f} {b[1]:7.3f} {b[2][0]:6.3f} "
                  f"{b[2][1]:6.3f} {b[2][2]:6.3f} {b[4]:5.0%} {b[5]:7.2f}")
            rows.append({"variant": tag, "overlap": ov, "precision": b[0],
                         "recall": b[1], "per": b[2], "fire": b[4], "bpi": b[5]})
    if rows:
        best = min(rows, key=lambda r: (round(r["bpi"], 4), -r["precision"]))
        print(f"\n基准 {base[5]:.2f} 箱/张 (P {base[0]:.3f});"
              f"最好 {best['variant']} ov={best['overlap']} "
              f"{best['bpi']:.2f} 箱/张 (P {best['precision']:.3f})")
        print("-> " + ("无改善,负门控不采纳" if best["bpi"] >= base[5] - 1e-9
                       else "有改善,须再经留出验证"))
    Path("/workspace/exp_cb/negative_gate.json").write_text(
        json.dumps({"baseline": {"p": base[0], "bpi": base[5]}, "variants": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
