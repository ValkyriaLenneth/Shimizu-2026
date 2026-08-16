#!/usr/bin/env python3
"""Re-tune the gate strength now that the gated region has changed shape.

The gate strength g -- the fraction of a detection's area that must lie inside
the member box -- was chosen as 0.5 from {0.1, 0.3, 0.5, 0.7, 0.9} while the gate
used the union of every router box, covering 74% of the frame. Switching to the
single highest-scoring box shrank that to 55%, which changes what any given g
means: the same fraction of a smaller region is a stricter filter.

This project has been caught by exactly this once before -- 315 ensemble subsets
were rejected only because they were scored at a wbf_iou chosen for a different
configuration -- so the parameter is re-swept rather than carried over.

Judged on sound-image boxes at matched recall rather than on precision, since
precision picked from a sweep has failed holdout validation three times today.
Any winner still has to clear a holdout before it can be adopted.
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
from PIL import Image
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

GATES = [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


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
    mt = RG.member_boxes(device, imgs, sizes)
    ms = RG.member_boxes(device, sound, ssz)

    print(f"{'门控 g':>8} {'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'箱/张':>7} {'阈值':>18}")
    rows = []
    for g in GATES:
        gt_, gs_ = RG.apply_gate(pt, mt, g), RG.apply_gate(ps, ms, g)
        best = None
        for combo in itertools.product(GRID, repeat=NC):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            for n, dd in gt_.items():
                sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dd if s >= combo[c]]
                merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
            fn = sum(v["fn"] for v in tot.values())
            p_, r_, _ = metric(tp, fp, fn)
            per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
            if r_ >= RG.TARGET and all(v >= RG.TARGET for v in per):
                bpi = sum(len([x for x in d if x[1] >= combo[x[0]]]) for d in gs_.values()) / len(gs_)
                # Minimise sound-image boxes, not precision.
                if best is None or bpi < best[6]:
                    fired = sum(1 for d in gs_.values() if any(x[1] >= combo[x[0]] for x in d))
                    best = (p_, r_, per, list(combo), fired / len(gs_), None, bpi)
        if not best:
            print(f"{str(g):>8}   无四项达标点"); continue
        p_, r_, per, combo, fire, _, bpi = best
        print(f"{str(g):>8} {p_:7.3f} {r_:7.3f} {per[0]:6.3f} {per[1]:6.3f} {per[2]:6.3f} "
              f"{fire:5.0%} {bpi:7.2f} {str(combo):>18}")
        rows.append({"gate": g, "precision": p_, "recall": r_, "per": per,
                     "thr": combo, "fire": fire, "bpi": bpi})
    cur = [r for r in rows if r["gate"] == 0.5]
    ok = [r for r in rows if r["gate"] is not None]
    if cur and ok:
        b = min(ok, key=lambda r: r["bpi"])
        print(f"\n现配置 g=0.5: {cur[0]['bpi']:.2f} 箱/张,P {cur[0]['precision']:.3f}")
        print(f"误报最低 g={b['gate']}: {b['bpi']:.2f} 箱/张,P {b['precision']:.3f}")
        print("-> " + ("与现配置相同,无需改动" if b["gate"] == 0.5
                       else f"g={b['gate']} 更优,须再经留出验证"))
    Path("/workspace/exp_cb/gate_strength_single.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
