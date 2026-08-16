#!/usr/bin/env python3
"""Is there signal in where a box sits *within* the member, rather than in frame?

The false-alarm profile compared box positions in frame coordinates and found
false alarms and true positives almost indistinguishable -- median vertical
position 0.64 for both -- which closed the door on a second geometric filter.
But frame coordinates conflate two things: where the member is in the photograph,
and where the damage is on the member. A column base photographed from above and
one photographed level put the same physical location at different frame heights.

The gate now supplies the member box, so the second quantity is available for the
first time: each detection's centre expressed as a fraction of the member box
rather than of the frame. If real damage concentrates somewhere on the member --
at the base where it meets the floor, say -- and false alarms do not, that is a
filter the frame-coordinate analysis could not have seen.

Reported for true positives on the damaged split against surviving false alarms
on the sound photographs, both under the delivered configuration, and only for
boxes whose member the router actually located.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
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


def rel(dets, members, targets=None):
    out = []
    for name, dd in dets.items():
        m = members.get(name)
        if m is None:
            continue
        mw, mh = max(m[2]-m[0], 1), max(m[3]-m[1], 1)
        for c, s, b in dd:
            if s < THR[c]:
                continue
            if targets is not None:
                if not any(t.cls == c and boxiou(b, t.xyxy) >= RG.MATCH_IOU for t in targets[name]):
                    continue
            out.append({"grade": GRADES[c], "score": s,
                        "cx": float(np.clip(((b[0]+b[2])/2 - m[0]) / mw, 0, 1)),
                        "cy": float(np.clip(((b[1]+b[3])/2 - m[1]) / mh, 0, 1)),
                        "rw": (b[2]-b[0]) / mw, "rh": (b[3]-b[1]) / mh})
    return out


def show(rows, label):
    if not rows:
        print(f"{label}: 无"); return {}
    cy = np.array([r["cy"] for r in rows]); cx = np.array([r["cx"] for r in rows])
    rw = np.array([r["rw"] for r in rows]); rh = np.array([r["rh"] for r in rows])
    print(f"\n{label} —— {len(rows)} 个框(构件相对坐标)")
    print(f"  纵向 cy   p25 {np.percentile(cy,25):.2f}  中位 {np.median(cy):.2f}  "
          f"p75 {np.percentile(cy,75):.2f}")
    print(f"  横向 cx   p25 {np.percentile(cx,25):.2f}  中位 {np.median(cx):.2f}  "
          f"p75 {np.percentile(cx,75):.2f}")
    print(f"  相对宽高  宽 中位 {np.median(rw):.2f}  高 中位 {np.median(rh):.2f}")
    for lo, hi, tag in ((0.0, 0.33, "构件上三分之一"), (0.33, 0.67, "中三分之一"),
                        (0.67, 1.01, "下三分之一")):
        print(f"    {tag}: {np.mean((cy >= lo) & (cy < hi)):.0%}")
    return {"n": len(rows), "cy_median": float(np.median(cy)),
            "cx_median": float(np.median(cx)),
            "thirds": [float(np.mean((cy >= a) & (cy < b))) for a, b in
                       ((0, .33), (.33, .67), (.67, 1.01))]}


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
    mt, ms = RG.member_boxes(device, imgs, sizes), RG.member_boxes(device, sound, ssz)
    ft = RG.apply_gate(fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU), mt, 0.5)
    fs = RG.apply_gate(fuse(RG.detect(device, sound, ssz), keys, w, ssz, RG.IOU), ms, 0.5)
    a = show(rel(ft, mt, targets), "受损图上的真阳性")
    b = show(rel(fs, ms), "健全图上幸存的误报")
    if a and b:
        d = abs(a["cy_median"] - b["cy_median"])
        print(f"\n纵向中位差 {d:.2f} —— " +
              ("有可利用的分离" if d >= 0.15 else "无可利用的分离,构件内位置不是判别信息"))
    Path("/workspace/exp_cb/member_relative.json").write_text(
        json.dumps({"true_positive": a, "false_alarm": b}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
