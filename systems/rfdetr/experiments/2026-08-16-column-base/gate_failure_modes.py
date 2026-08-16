#!/usr/bin/env python3
"""Where does the router fail to find the member, and what does that cost?

The delivered configuration now depends on a component it did not depend on this
morning: the 2026-07-07 five-class router locates the column base, and detections
away from it are discarded. The gate is deliberately permissive -- an image where
the router finds nothing is passed through ungated -- so a router failure never
costs recall. But it does silently cost the gate's benefit, and nobody has looked
at which images those are.

Two things follow from knowing that. If the failures share a property (small
member, unusual framing, low light), the same property probably describes a slice
of the client's real intake where the false-alarm improvement will not appear,
and that belongs in the delivery note rather than being discovered on site. And
if the ungated images are also the ones that fire most, the gate's measured
benefit is an average over a population that splits into two very different
halves.

Reports, for the frozen test split and the client's sound photographs: how often
the router finds nothing, what those images look like beside the ones it handles,
and what the gate is and is not doing on each group.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import read_targets

THR = {0: 0.12, 1: 0.20, 2: 0.12}


def stats(paths, sizes, members, fused_gated, fused_plain, targets=None):
    found = [p.name for p in paths if members.get(p.name)]
    miss = [p.name for p in paths if not members.get(p.name)]
    out = {}
    for label, group in (("路由器定位成功", found), ("路由器未定位", miss)):
        if not group:
            print(f"  {label}: 0 张"); continue
        mp = np.array([sizes[n][0] * sizes[n][1] / 1e6 for n in group])
        ar = np.array([sizes[n][0] / sizes[n][1] for n in group])
        bright = []
        for n in group[:60]:
            p = next(q for q in paths if q.name == n)
            with Image.open(p) as h:
                bright.append(float(np.asarray(h.convert("L").resize((64, 64))).mean()))
        kept = sum(len([x for x in fused_gated[n] if x[1] >= THR[x[0]]]) for n in group)
        raw = sum(len([x for x in fused_plain[n] if x[1] >= THR[x[0]]]) for n in group)
        fired = sum(1 for n in group if any(x[1] >= THR[x[0]] for x in fused_gated[n]))
        line = (f"  {label}: {len(group)} 张 · 中位 {np.median(mp):.2f}MP · "
                f"宽高比中位 {np.median(ar):.2f} · 亮度中位 {np.median(bright):.0f}")
        print(line)
        print(f"      门控前 {raw/len(group):.2f} 箱/张 -> 门控后 {kept/len(group):.2f}  "
              f"发火 {fired}/{len(group)} = {fired/len(group):.0%}")
        if targets is not None:
            gt = sum(len(targets[n]) for n in group)
            print(f"      真值框 {gt} 个({gt/len(group):.2f}/张)")
        out[label] = {"n": len(group), "mp_median": float(np.median(mp)),
                      "bpi_before": raw / len(group), "bpi_after": kept / len(group),
                      "fire": fired / len(group)}
    return out


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
    dt, ds = RG.detect(device, imgs, sizes), RG.detect(device, sound, ssz)
    mt, ms = RG.member_boxes(device, imgs, sizes), RG.member_boxes(device, sound, ssz)
    pt, ps = fuse(dt, keys, w, sizes, RG.IOU), fuse(ds, keys, w, ssz, RG.IOU)
    gt_, gs_ = RG.apply_gate(pt, mt, 0.5), RG.apply_gate(ps, ms, 0.5)

    print("冻结测试集(45 图):")
    a = stats(imgs, sizes, mt, gt_, pt, targets)
    print("\n客户健全柱脚照片(29 张):")
    b = stats(sound, ssz, ms, gs_, ps)
    print("\n判读: 若未定位组的图像属性与定位成功组明显不同,"
          "\n      则现场同类图上门控的误报改善不会出现,应写进交付说明。")
    Path("/workspace/exp_cb/gate_failure_modes.json").write_text(
        json.dumps({"test": a, "sound": b}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
