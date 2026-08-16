#!/usr/bin/env python3
"""Put both sides of the trade into the unit the client actually experiences.

An intervention that costs precision on damaged photographs while buying quiet on
sound ones cannot be judged from the two numbers as reported: they are measured on
different image populations and in different units, so "precision fell 0.076" and
"false boxes fell 0.4 per sound image" cannot be added or compared.

They become comparable once both are expressed as boxes a person has to look at.
An inspector works through a batch of site photographs of which some fraction
show damage. Each damaged photograph yields the true detections plus whatever
false ones the precision implies; each sound photograph yields only false ones.
Summing over a batch gives one number per configuration -- boxes reviewed per 100
photographs -- and a second, the share of real damage found, which must not be
traded away silently.

Damage prevalence is not known for this client's inspection workflow and is not
guessed here: the comparison is reported across a range, and the value at which
one configuration overtakes the other is solved for directly. That crossover is
the quantity to put to the client, since they know their own prevalence and this
turns the decision into one fact they already hold.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np


def boxes_per_100(prev: float, recall: float, precision: float, gt_per_damaged: float,
                  sound_bpi: float) -> float:
    """Boxes an inspector reviews per 100 photographs.

    On damaged images the model emits recall*gt true boxes; precision fixes the
    ratio of true to total, so total = true/precision. Sound images contribute
    their measured boxes per image directly.
    """
    dmg = 100 * prev
    snd = 100 * (1 - prev)
    true_boxes = dmg * gt_per_damaged * recall
    dmg_boxes = true_boxes / max(precision, 1e-9)
    return dmg_boxes + snd * sound_bpi


def main():
    cfgs = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {
        # Delivered configuration, measured 2026-08-16 on the frozen protocol.
        "现交付 (+翻转)": {"recall": 0.708, "precision": 0.383, "sound_bpi": 1.86},
    }
    gt = 72 / 45     # boxes per damaged image in the frozen split
    print(f"每张受损照片平均 {gt:.2f} 个真实框(冻结测试集 72/45)\n")
    prevs = [0.05, 0.10, 0.20, 0.30, 0.50]
    names = list(cfgs)
    print(f"{'损伤占比':>8} " + " ".join(f"{n:>16}" for n in names) + "   每100张照片需过目的框数")
    for p in prevs:
        row = [boxes_per_100(p, c["recall"], c["precision"], gt, c["sound_bpi"])
               for c in cfgs.values()]
        print(f"{p:8.0%} " + " ".join(f"{v:16.1f}" for v in row))
    print(f"\n{'损伤占比':>8} " + " ".join(f"{n:>16}" for n in names) + "   其中真实损伤被找到的比例")
    for p in prevs:
        print(f"{p:8.0%} " + " ".join(f"{c['recall']:16.1%}" for c in cfgs.values()))

    if len(names) == 2:
        a, b = (cfgs[n] for n in names)
        f = lambda p: (boxes_per_100(p, a["recall"], a["precision"], gt, a["sound_bpi"])
                       - boxes_per_100(p, b["recall"], b["precision"], gt, b["sound_bpi"]))
        lo, hi = 0.0, 1.0
        if f(lo) * f(hi) < 0:
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
            print(f"\n交叉点: 损伤占比 {(lo+hi)/2:.1%} 时两者过目框数相等")
            print(f"  低于该值 -> {names[0] if f(0.01)<0 else names[1]} 更省人力")
            print(f"  高于该值 -> {names[1] if f(0.01)<0 else names[0]} 更省人力")
        else:
            better = names[0] if f(0.5) < 0 else names[1]
            print(f"\n在 0-100% 全部损伤占比下,{better} 的过目框数都更低 —— 无交叉点")
        if abs(a["recall"] - b["recall"]) > 0.005:
            print(f"  注意: 两者 recall 不同 ({a['recall']:.3f} vs {b['recall']:.3f}),"
                  f"省下的人力部分来自漏检,不能只看框数")


if __name__ == "__main__":
    main()
