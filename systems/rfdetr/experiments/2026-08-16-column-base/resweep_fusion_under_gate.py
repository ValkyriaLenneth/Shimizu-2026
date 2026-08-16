"""Does the gate move the fusion optimum?

wbf_iou 0.40 and the 1:2 member weighting were chosen before the router gate
existed, on detections that still carried every off-member box. The gate now
removes those, and fusion parameters interact strongly with what is being fused
-- earlier on this project, 315 member subsets failed only because they were
evaluated at the wrong wbf_iou.

So the pair is re-swept with the gate in place. Judged on false alarms at fixed
recall rather than on precision, since precision picked out of a sweep did not
survive holdout validation either for the flip view or for the gate itself.
"""
import itertools, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)
from PIL import Image

GATE = 0.5
IOUS = [0.20, 0.30, 0.40, 0.50, 0.60]
RATIOS = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"

imgs = sorted(p for p in (DS/"test"/"images").iterdir()
              if p.suffix.lower() in {".jpg",".jpeg",".png"})
sound = sorted(p for p in RG.SOUND.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
sizes, ssz, targets = {}, {}, {}
for p in imgs:
    with Image.open(p) as h: sizes[p.name] = h.size
    targets[p.name] = read_targets(DS/"test"/"labels"/f"{p.stem}.txt", *sizes[p.name])
for p in sound:
    with Image.open(p) as h: ssz[p.name] = h.size
dt = RG.detect(device, imgs, sizes)
ds = RG.detect(device, sound, ssz)
mt = RG.member_boxes(device, imgs, sizes)
ms = RG.member_boxes(device, sound, ssz)
keys = [(m,v) for m in MEMBERS for v in RG.VIEWS]

print(f"{'wbf_iou':>8} {'权重':>6} {'最高P':>7} {'R':>7} {'健全发火':>9} {'框/张':>7}")
rows=[]
for iou in IOUS:
    for ratio in RATIOS:
        w = [ratio[i]/len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
        ft = RG.apply_gate(fuse(dt, keys, w, sizes, iou), mt, GATE)
        fs = RG.apply_gate(fuse(ds, keys, w, ssz, iou), ms, GATE)
        best=None
        for combo in itertools.product(GRID, repeat=NC):
            tot={c:{"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
            for n,dd in ft.items():
                sel=[Prediction(cls=c,conf=s,xyxy=b) for c,s,b in dd if s>=combo[c]]
                merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            tp=sum(v["tp"] for v in tot.values()); fp=sum(v["fp"] for v in tot.values())
            fn=sum(v["fn"] for v in tot.values())
            p,r,_=metric(tp,fp,fn)
            per=[metric(tot[c]["tp"],tot[c]["fp"],tot[c]["fn"])[1] for c in range(NC)]
            if r>=RG.TARGET and all(v>=RG.TARGET for v in per) and (best is None or p>best[0]):
                best=(p,r,combo)
        if best is None:
            print(f"{iou:8.2f} {f'{ratio[0]:.0f}:{ratio[1]:.0f}':>6}   无四项达标点"); continue
        p,r,combo=best
        fired=sum(1 for d in fs.values() if any(x[1]>=combo[x[0]] for x in d))
        boxes=sum(len([x for x in d if x[1]>=combo[x[0]]]) for d in fs.values())
        print(f"{iou:8.2f} {f'{ratio[0]:.0f}:{ratio[1]:.0f}':>6} {p:7.3f} {r:7.3f} "
              f"{fired/len(fs):8.0%} {boxes/len(fs):7.2f}")
        rows.append({"iou":iou,"ratio":f"{ratio[0]:.0f}:{ratio[1]:.0f}","p":p,
                     "fire":fired/len(fs),"bpi":boxes/len(fs)})
cur=[r for r in rows if r["iou"]==0.40 and r["ratio"]=="1:2"]
if cur and rows:
    b=min(rows,key=lambda r:r["bpi"])
    print(f"\n现配置 iou0.40/1:2: {cur[0]['bpi']:.2f} 框/张")
    print(f"误报最低: iou{b['iou']}/{b['ratio']} = {b['bpi']:.2f} 框/张 (P {b['p']:.3f})")
Path("/workspace/exp_cb/gate_x_fusion.json").write_text(json.dumps(rows, indent=2))
