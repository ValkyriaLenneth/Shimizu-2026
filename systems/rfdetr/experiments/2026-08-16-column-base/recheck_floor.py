import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
"""Re-check the member detection floor under the current gate.

conf_type was fixed to "max" on the morning's fusion sweep, when WBF combined two
checkpoints. It now combines four inputs -- two checkpoints each seen twice,
once mirrored -- and the choice matters more there: "max" takes the single most
confident of four opinions, while averaging pulls a box down unless several of
them agree. With the flip views added, two of the four inputs are near-duplicates
of the other two, which is exactly the situation where the two rules diverge.

Judged on sound-image boxes at matched recall, as every gate and fusion change
has been, with the delivered setting included for comparison.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

FLOORS = [0.05, 0.10, 0.15, 0.20]
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
imgs = sorted(p for p in (DS/"test"/"images").iterdir()
              if p.suffix.lower() in {".jpg",".jpeg",".png"})
sound = sorted(p for p in RG.SOUND.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
sizes, ssz, targets = {}, {}, {}
for p in imgs:
    with Image.open(p) as h: sizes[p.name] = h.size
    targets[p.name] = read_targets(DS/"test"/"labels"/f"{p.stem}.txt", *sizes[p.name])
for p in sound:
    with Image.open(p) as h: ssz[p.name] = h.size
mt, ms = RG.member_boxes(device, imgs, sizes), RG.member_boxes(device, sound, ssz)
keys = [(m,v) for m in MEMBERS for v in RG.VIEWS]
w = [RG.RATIO[i]/len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]

def fuse_ct(store, szs, ct):
    out={}
    for n in szs:
        bl,sl,ll=[],[],[]
        for k in keys:
            r=store[k][n]
            bl.append(np.asarray(r["boxes"],np.float32).reshape(-1,4).tolist())
            sl.append(list(r["scores"])); ll.append(list(r["classes"]))
        W,H=szs[n]
        if not any(len(b) for b in bl): out[n]=[]; continue
        b,s,l=weighted_boxes_fusion(bl,sl,ll,weights=w,iou_thr=RG.IOU,
                                    skip_box_thr=0.0,conf_type=ct)
        out[n]=[(int(c),float(x),tuple((np.asarray(bb)*np.array([W,H,W,H])).tolist()))
                for bb,x,c in zip(b,s,l)]
    return out

print(f"{'floor':>20s} {'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'箱/张':>7}")
rows=[]
for fl in FLOORS:
    # The floor decides which boxes enter fusion at all; the gate can only remove
    # boxes that already entered, so the two act at different stages and the
    # floor has never been re-checked since the gate was added.
    RG.FLOOR = fl
    dt, ds = RG.detect(device, imgs, sizes), RG.detect(device, sound, ssz)
    ct = "max"
    ft = RG.apply_gate(fuse_ct(dt, sizes, ct), mt, 0.5)
    fs = RG.apply_gate(fuse_ct(ds, ssz, ct), ms, 0.5)
    best=None
    for combo in itertools.product(GRID, repeat=NC):
        tot={c:{"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
        for n,dd in ft.items():
            sel=[Prediction(cls=c,conf=s,xyxy=b) for c,s,b in dd if s>=combo[c]]
            merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
        tp=sum(v["tp"] for v in tot.values()); fp=sum(v["fp"] for v in tot.values())
        fn=sum(v["fn"] for v in tot.values())
        p_,r_,_=metric(tp,fp,fn)
        per=[metric(tot[c]["tp"],tot[c]["fp"],tot[c]["fn"])[1] for c in range(NC)]
        if r_>=RG.TARGET and all(v>=RG.TARGET for v in per):
            bpi=sum(len([x for x in d if x[1]>=combo[x[0]]]) for d in fs.values())/len(fs)
            if best is None or bpi<best[-1]:
                fired=sum(1 for d in fs.values() if any(x[1]>=combo[x[0]] for x in d))
                best=(p_,r_,per,list(combo),fired/len(fs),bpi)
    if not best: print(f"{fl:20.2f}   无四项达标点"); continue
    p_,r_,per,combo,fire,bpi = best
    print(f"{fl:20.2f} {p_:7.3f} {r_:7.3f} {per[0]:6.3f} {per[1]:6.3f} {per[2]:6.3f} "
          f"{fire:5.0%} {bpi:7.2f}")
    rows.append({"floor":fl,"precision":p_,"fire":fire,"bpi":bpi,"thr":combo})
if rows:
    b=min(rows,key=lambda r:(round(r["bpi"],4),-r["precision"]))
    cur=[r for r in rows if abs(r["floor"]-0.10)<1e-9][0]
    print(f"\n现配置 floor 0.10: {cur['bpi']:.2f} 箱/张 (P {cur['precision']:.3f})")
    print(f"最优 floor {b['floor']:.2f}: {b['bpi']:.2f} 箱/张 (P {b['precision']:.3f})")
    print("-> " + ("与现配置相同,无需改动" if abs(b["floor"]-0.10)<1e-9 else "须再经留出验证"))
Path("/workspace/exp_cb/floor_regate.json").write_text(json.dumps(rows, indent=2))
