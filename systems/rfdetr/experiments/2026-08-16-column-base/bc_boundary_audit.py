#!/usr/bin/env python3
"""Is the B/C boundary consistent enough for a B recall figure to mean anything?

Three independent measurements today point the remaining false-alarm problem back
at grade B: it carries 83% of the surviving false alarms, its detections sit just
above threshold at a median score of 0.207, and the only geometric difference
between false alarms and true damage tracks B's own shape rather than error. The
baseline configuration records eleven grade contradictions in the annotations,
six of them 柱脚 B-versus-C pairs, but that count came from an earlier audit and
has never been checked against what the boxes actually look like.

It matters for the decision now sitting with the client. They are being asked
whether B recall may fall from 0.702 to 0.681 -- one box of 47 -- and that
question presumes the B label is stable enough for 0.702 to be a real quantity.
If B and C are drawn from overlapping distributions, a recall figure for either
is partly reporting annotator disagreement.

Measured from the frozen data alone: how separable B and C are by the properties
an annotator would use -- box area, aspect, and the model's own confidence -- and
how often the delivered configuration puts a B box where the corpus says C.
No model is trained and no label is changed.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import read_targets

SPLITS = ("train", "test")
GRADES = "BCD"
THR = {0: 0.12, 1: 0.20, 2: 0.12}


def boxiou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    # --- ground-truth geometry, all labelled data -----------------------------
    per = {g: {"area": [], "aspect": []} for g in GRADES}
    for split in SPLITS:
        d = DS / split
        for p in sorted((d / "images").iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab = d / "labels" / f"{p.stem}.txt"
            if not lab.exists():
                continue
            with Image.open(p) as h:
                W, H = h.size
            for line in lab.read_text().splitlines():
                if not line.strip():
                    continue
                c, x, y, w, h_ = (float(v) for v in line.split()[:5])
                per[GRADES[int(c)]]["area"].append(w * h_)
                per[GRADES[int(c)]]["aspect"].append((w * W) / max(h_ * H, 1e-6))
    print("标注框几何(全部已标注数据):")
    print(f"{'级':>3} {'框数':>5} {'面积占比 p25/中位/p75':>26} {'宽高比 中位':>12}")
    for g in GRADES:
        a = np.array(per[g]["area"]); r = np.array(per[g]["aspect"])
        print(f"{g:>3} {len(a):5d} "
              f"{np.percentile(a,25):8.4f}/{np.median(a):.4f}/{np.percentile(a,75):.4f}   "
              f"{np.median(r):11.2f}")
    ab, ac = np.array(per["B"]["area"]), np.array(per["C"]["area"])
    # Overlap of the two area distributions: the share of B that falls inside
    # C's interquartile range and vice versa. High on both sides means the
    # property an annotator grades on does not separate them.
    clo, chi = np.percentile(ac, [25, 75]); blo, bhi = np.percentile(ab, [25, 75])
    print(f"\nB 落在 C 的四分位区间内: {np.mean((ab >= clo) & (ab <= chi)):.0%}")
    print(f"C 落在 B 的四分位区间内: {np.mean((ac >= blo) & (ac <= bhi)):.0%}")

    # --- how the delivered configuration confuses them ------------------------
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
    pairs, hits, miss = Counter(), 0, 0
    for name, dd in ft.items():
        gts = [(t.cls, t.xyxy) for t in targets[name]]
        used = set()
        for c, s, b in sorted(dd, key=lambda x: -x[1]):
            if s < THR[c]:
                continue
            same = [i for i, (gc, gb) in enumerate(gts)
                    if gc == c and i not in used and boxiou(b, gb) >= RG.MATCH_IOU]
            if same:
                used.add(same[0]); hits += 1; continue
            other = [(i, gc) for i, (gc, gb) in enumerate(gts)
                     if gc != c and boxiou(b, gb) >= RG.MATCH_IOU]
            if other:
                pairs[f"预测{GRADES[c]} 实为{GRADES[other[0][1]]}"] += 1
            else:
                miss += 1
    print(f"\n交付配置在冻结集上: 命中 {hits},位置对等级错 {sum(pairs.values())},"
          f"位置错 {miss}")
    for k, v in pairs.most_common():
        print(f"  {k}: {v}")
    bc = pairs["预测B 实为C"] + pairs["预测C 实为B"]
    print(f"\nB<->C 互错 {bc} 次,占等级错的 {bc/max(sum(pairs.values()),1):.0%}")
    Path("/workspace/exp_cb/bc_boundary.json").write_text(json.dumps(
        {"gt_geometry": {g: {"n": len(per[g]["area"]),
                             "area_median": float(np.median(per[g]["area"]))} for g in GRADES},
         "b_in_c_iqr": float(np.mean((ab >= clo) & (ab <= chi))),
         "c_in_b_iqr": float(np.mean((ac >= blo) & (ac <= bhi))),
         "confusions": dict(pairs)}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
