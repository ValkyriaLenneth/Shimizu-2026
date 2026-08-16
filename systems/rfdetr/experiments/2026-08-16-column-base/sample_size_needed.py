#!/usr/bin/env python3
"""How many test boxes would make the delivered promise reliable?

The three recalls written into the delivery hold together on only 17% of
resamples of the frozen split, because C carries 15 boxes and D just 10. The
request to the client has so far been "extend the C/D test samples", which is
correct but unquantified: they cannot act on it without a number.

The number is measurable. Resampling the frozen split at its own size gives 17%;
resampling at larger sizes gives the curve, and the point where it crosses a
usable threshold is what to ask for. Sampling with replacement beyond the
original size assumes new images resemble the ones in hand, which is the same
assumption every figure in this package already rests on, and it is stated here
rather than buried.

Reported for the delivered configuration and the candidate alike, since the
answer should not depend on which is chosen.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)
from PIL import Image

POINTS = {"现交付": (0.12, 0.20, 0.12), "候补": (0.15, 0.20, 0.12)}
SCALES = [1, 2, 3, 5, 8, 12]
NBOOT = 600
GRADES = "BCD"


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
    names = sorted(targets)
    nbox = sum(len(v) for v in targets.values())

    tabs = {}
    for lab, thr in POINTS.items():
        tp = np.zeros((len(names), NC), np.int32); fn = np.zeros_like(tp)
        for i, n in enumerate(names):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in ft[n] if s >= thr[c]]
            merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            for c in range(NC):
                tp[i, c] = tot[c]["tp"]; fn[i, c] = tot[c]["fn"]
        full = np.arange(len(names))
        per = [tp[full, c].sum() / max(tp[full, c].sum() + fn[full, c].sum(), 1) for c in range(NC)]
        tabs[lab] = (tp, fn, per)
        print(f"{lab}: 承诺 " + " / ".join(f"{GRADES[c]} {per[c]:.3f}" for c in range(NC)))

    print(f"\n当前 {len(names)} 图 / {nbox} 框。按倍数放大后,三级承诺同时成立的比例:")
    print(f"{'规模':>10} {'图数':>6} {'框数':>6} " +
          "  ".join(f"{lab:>8}" for lab in POINTS))
    out = []
    for k in SCALES:
        row = {}
        for lab, (tp, fn, per) in tabs.items():
            rng = np.random.default_rng(20260816)
            hold = 0
            for _ in range(NBOOT):
                idx = rng.integers(0, len(names), len(names) * k)
                ok = True
                for c in range(NC):
                    t, m = tp[idx, c].sum(), fn[idx, c].sum()
                    if t / max(t + m, 1) < per[c]:
                        ok = False; break
                hold += ok
            row[lab] = hold / NBOOT
        print(f"{k:9d}x {len(names)*k:6d} {nbox*k:6d} " +
              "  ".join(f"{row[lab]:7.0%}" for lab in POINTS))
        out.append({"scale": k, "images": len(names) * k, "boxes": nbox * k, **row})
    print("\n注: 放大采样假定新增图与现有图同分布 —— 与本包全部数字所依赖的假设相同。")
    print("    比例趋近 50% 是自助法的上界(承诺取自全量估计,重采样对称分布)。")
    Path("/workspace/exp_cb/sample_size_needed.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
