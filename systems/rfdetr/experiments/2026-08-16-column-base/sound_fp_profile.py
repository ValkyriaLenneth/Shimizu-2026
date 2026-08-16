#!/usr/bin/env python3
"""What still fires on a sound column base after the gate?

The gate removed the boxes that sat away from the member, taking false alarms
from 1.86 to 1.66 per sound image. What remains is a different failure: 86% of
sound photographs still produce at least one box, and those boxes are *on* the
column base, where the gate has no reason to touch them. They are the dominant
remaining cost -- at a 5% damage prevalence, 92% of what an inspector reviews
comes from sound imagery.

Nothing has ever characterised them. This does, along four axes that each imply a
different remedy:

  score      if they cluster just above threshold, they are marginal detections
             and the operating point is the lever; if they are confident, the
             model genuinely believes there is damage and thresholds cannot help
  grade      a single grade dominating points at that grade's decision boundary
  size       boxes much smaller than real damage suggest texture, not structure
  position   clustering within the member box (base, edges, centre) suggests a
             recurring physical feature - bolt, shadow, joint, stain

Compared throughout against the same statistics for true positives on the damaged
split, since "small and low-scoring" only means something relative to what real
damage looks like to this model.
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
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import read_targets

THR = {0: 0.12, 1: 0.20, 2: 0.12}
GRADES = "BCD"


def boxiou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def describe(rows, label, member):
    if not rows:
        print(f"{label}: 无框"); return {}
    sc = np.array([r["score"] for r in rows])
    # Images where the router found no member are ungated, so their boxes carry
    # no member-relative geometry. They are counted but excluded from the
    # geometric statistics rather than silently dropped.
    geo = [r for r in rows if r.get("area_frac_member") is not None]
    ar = np.array([r["area_frac_member"] for r in geo]) if geo else np.array([])
    g = Counter(r["grade"] for r in rows)
    print(f"\n{label} —— {len(rows)} 个框(其中 {len(geo)} 个位于路由器定位到的构件内)")
    print(f"  分数    中位 {np.median(sc):.3f}  p25 {np.percentile(sc,25):.3f}  "
          f"p75 {np.percentile(sc,75):.3f}  最大 {sc.max():.3f}")
    print(f"  等级    " + "  ".join(f"{k} {v} ({v/len(rows):.0%})" for k, v in sorted(g.items())))
    if len(ar):
        print(f"  占构件框面积  中位 {np.median(ar):.1%}  p90 {np.percentile(ar,90):.1%}")
    if member and geo:
        cx = np.array([r["cx_in_member"] for r in geo])
        cy = np.array([r["cy_in_member"] for r in geo])
        print(f"  构件内相对位置  横向中位 {np.median(cx):.2f}  纵向中位 {np.median(cy):.2f}  "
              f"(0=左/上, 1=右/下)")
        lo = np.mean(cy > 0.6)
        print(f"  落在构件下 40% 的比例 {lo:.0%}")
    return {"n": len(rows), "n_in_member": len(geo), "score_median": float(np.median(sc)),
            "grades": dict(g),
            "area_median": float(np.median(ar)) if len(ar) else None}


def collect(fused, members, targets=None):
    rows = []
    for name, dets in fused.items():
        mb = members.get(name)
        for c, s, b in dets:
            if s < THR[c]:
                continue
            if targets is not None:
                hit = any(t.cls == c and boxiou(b, t.xyxy) >= 0.229 for t in targets[name])
                if not hit:
                    continue
            w, h = b[2] - b[0], b[3] - b[1]
            rec = {"image": name, "grade": GRADES[c], "score": s,
                   "w": w, "h": h,
                   "area_frac_member": (w * h) / max((mb[2]-mb[0])*(mb[3]-mb[1]), 1) if mb else None}
            if mb:
                rec["cx_in_member"] = float(np.clip(((b[0]+b[2])/2 - mb[0]) / max(mb[2]-mb[0], 1), 0, 1))
                rec["cy_in_member"] = float(np.clip(((b[1]+b[3])/2 - mb[1]) / max(mb[3]-mb[1], 1), 0, 1))
            rows.append(rec)
    return rows


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
    mt = RG.member_boxes(device, imgs, sizes)
    ms = RG.member_boxes(device, sound, ssz)
    ft = RG.apply_gate(fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU), mt, 0.5)
    fs = RG.apply_gate(fuse(RG.detect(device, sound, ssz), keys, w, ssz, RG.IOU), ms, 0.5)

    fired = sum(1 for d in fs.values() if any(s >= THR[c] for c, s, _ in d))
    print(f"健全图 {len(fs)} 张,发火 {fired} 张 = {fired/len(fs):.0%}")
    a = describe(collect(fs, ms), "健全图上幸存的框(全部为误报)", True)
    b = describe(collect(ft, mt, targets), "受损图上的真阳性(对照)", True)

    if a and b:
        print(f"\n对比: 误报分数中位 {a['score_median']:.3f} vs 真阳性 {b['score_median']:.3f}")
        if a.get("area_median") and b.get("area_median"):
            print(f"      误报面积中位 {a['area_median']:.1%} vs 真阳性 {b['area_median']:.1%}")
        print("\n判读: 误报若明显低分且更小 -> 边缘检测,工作点是杠杆;"
              "\n      若分数相当 -> 模型确信有损伤,阈值救不了,需要别的特征")
    Path("/workspace/exp_cb/sound_fp_profile.json").write_text(
        json.dumps({"sound": a, "true_positive": b}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
