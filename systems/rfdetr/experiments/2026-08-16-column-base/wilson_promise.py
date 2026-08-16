#!/usr/bin/env python3
"""State the recalls as confidence bounds, and price the test set that tightens them.

The earlier attempt asked how often a resample beats the point estimate, which is
about 50% per grade by construction and says nothing about sample size. A promise
that can be relied on has to be a lower bound, not a point: "B recall is at least
x with 95% confidence" is a claim that either holds or does not on new data, and
its distance from the point estimate shrinks as the test set grows.

So this reports, for each grade, the Wilson 95% lower bound at the current box
count and at multiples of it, holding the observed recall fixed. The gap between
the point estimate and the bound is what the client is buying when they annotate
more images, and it is quantified per grade rather than as a single request.

Wilson rather than normal-approximation intervals because D has ten boxes, where
the normal approximation is unusable.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)
from PIL import Image

THR = (0.12, 0.20, 0.12)
GRADES = "BCD"
SCALES = [1, 2, 3, 5, 10]
Z = 1.96


def wilson(k, n, z=Z):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    keys = [(m, v) for m in MEMBERS for v in RG.VIEWS]
    w = [RG.RATIO[i] / len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
    ft = RG.apply_gate(fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU),
                       RG.member_boxes(device, imgs, sizes), 0.5)
    tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
    for n, dd in ft.items():
        sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dd if s >= THR[c]]
        merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))

    print("当前测试集下的 recall 与 Wilson 95% 区间:\n")
    print(f"{'级':>3} {'命中/真值':>10} {'点估计':>7} {'95% 下界':>9} {'95% 上界':>9} {'下界差距':>9}")
    base = {}
    for c in range(NC):
        k = tot[c]["tp"]; n = tot[c]["tp"] + tot[c]["fn"]
        lo, hi = wilson(k, n)
        print(f"{GRADES[c]:>3} {f'{k}/{n}':>10} {k/n:7.3f} {lo:9.3f} {hi:9.3f} {k/n-lo:9.3f}")
        base[GRADES[c]] = (k, n, k/n, lo)

    print(f"\n若按同样的 recall 扩充测试集,95% 下界的变化:\n")
    print(f"{'规模':>6} " + "  ".join(f"{g+' 框数':>7} {g+' 下界':>8}" for g in GRADES))
    out = []
    for s_ in SCALES:
        row = {"scale": s_}
        cells = []
        for g in GRADES:
            k, n, p, _ = base[g]
            lo, _ = wilson(int(round(k*s_)), n*s_)
            cells.append(f"{n*s_:7d} {lo:8.3f}")
            row[g] = {"n": n*s_, "lower": float(lo)}
        print(f"{s_:5d}x " + "  ".join(cells))
        out.append(row)
    print("\n读法: 点估计不随规模变化(假定 recall 不变),变的是下界与它的距离。")
    print("      能写进合同的是下界;当前 D 级下界仅 "
          f"{base['D'][3]:.3f},即 10 个框支撑不起任何有意义的承诺。")
    Path("/workspace/exp_cb/wilson_promise.json").write_text(
        json.dumps({"current": {g: {"tp": base[g][0], "n": base[g][1],
                                    "point": base[g][2], "lower": base[g][3]}
                                for g in GRADES}, "scaled": out},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
