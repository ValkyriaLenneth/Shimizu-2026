#!/usr/bin/env python3
"""Make the gate a discount on confidence rather than a yes/no cut.

The gate keeps a detection when at least half its area falls inside the member
box and drops it otherwise. That is a step function over a quantity that is
continuous: a box 49% inside is discarded while one at 51% survives untouched,
and the sweep of that cut point found a plateau across 0.4-0.5, which is what a
step placed on a smooth underlying relationship looks like.

A softer form uses the same evidence without the discontinuity: scale the
detection's score by how much of it lies inside the member, so a box mostly on
the column base keeps most of its confidence and one hanging off the edge loses
most of it. The threshold search then decides what survives, which is where that
decision already lives for every other part of the configuration.

Three shapes are tried, all keeping the pass-through for images where the router
finds nothing:

  linear      score * inside_fraction
  power       score * inside_fraction ** k, sharper for larger k
  floored     score * max(inside_fraction, f), so a box fully outside keeps a
              little confidence rather than none

Judged on sound-image boxes at matched recall, as every gate change has been.
A winner still needs a holdout before adoption.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)


def frac_inside(b, m):
    x1, y1 = max(b[0], m[0]), max(b[1], m[1])
    x2, y2 = min(b[2], m[2]), min(b[3], m[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (b[2] - b[0]) * (b[3] - b[1])
    return i / a if a > 0 else 0.0


def soft(fused, members, mode, k):
    out = {}
    for name, dets in fused.items():
        mb = members.get(name)
        if mb is None:
            out[name] = dets; continue
        keep = []
        for c, s, b in dets:
            f = frac_inside(b, mb)
            if mode == "linear":
                w = f
            elif mode == "power":
                w = f ** k
            else:                      # floored
                w = max(f, k)
            if s * w > 0:
                keep.append((c, s * w, b))
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
    mt, ms = RG.member_boxes(device, imgs, sizes), RG.member_boxes(device, sound, ssz)

    base = score(RG.apply_gate(pt, mt, 0.5), RG.apply_gate(ps, ms, 0.5), targets, len(ssz))
    print(f"{'门控形态':26s} {'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'箱/张':>7}")
    print(f"{'硬门控 g=0.5(现交付)':26s} {base[0]:7.3f} {base[1]:7.3f} {base[2][0]:6.3f} "
          f"{base[2][1]:6.3f} {base[2][2]:6.3f} {base[4]:5.0%} {base[5]:7.2f}")
    rows = []
    for mode, ks in (("linear", [None]), ("power", [0.5, 2.0, 3.0]), ("floored", [0.2, 0.4, 0.6])):
        for k in ks:
            ft = soft(pt, mt, mode, k or 1.0)
            fs = soft(ps, ms, mode, k or 1.0)
            b = score(ft, fs, targets, len(ssz))
            tag = f"软门控 {mode}" + (f" k={k}" if k is not None else "")
            if not b:
                print(f"{tag:26s}   无四项达标点"); continue
            print(f"{tag:26s} {b[0]:7.3f} {b[1]:7.3f} {b[2][0]:6.3f} {b[2][1]:6.3f} "
                  f"{b[2][2]:6.3f} {b[4]:5.0%} {b[5]:7.2f}")
            rows.append({"mode": mode, "k": k, "precision": b[0], "recall": b[1],
                         "fire": b[4], "bpi": b[5]})
    if rows:
        best = min(rows, key=lambda r: (round(r["bpi"], 4), -r["precision"]))
        print(f"\n基准 {base[5]:.2f} 箱/张 (P {base[0]:.3f});"
              f"最好软门控 {best['mode']} k={best['k']} {best['bpi']:.2f} (P {best['precision']:.3f})")
        print("-> " + ("无改善,保持硬门控" if best["bpi"] >= base[5] - 1e-9
                       else "有改善,须再经留出验证"))
    Path("/workspace/exp_cb/soft_gate.json").write_text(
        json.dumps({"baseline": {"p": base[0], "bpi": base[5]}, "variants": rows},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
