import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
"""Harden the evidence behind the recommendation before it reaches the client.

The candidate point cleared both checks, but the holdout arm rested on 95 paired
splits with a lower interval bound sitting exactly at zero, and it is the one
number the client is being asked to act on. Two additions:

  more splits    600 instead of 95, so the "1% worse" figure is not itself noise
  the B floor    swept rather than fixed at 0.68, so the recommendation is a
                 point on a curve the client can move along rather than a single
                 setting they must accept or reject

Precision and sound-image boxes are both reported per floor. Nothing is retrained.
"""
import itertools, sys, json
from pathlib import Path
import numpy as np
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)
from PIL import Image

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
keys = [(m,v) for m in MEMBERS for v in RG.VIEWS]
w = [RG.RATIO[i]/len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
ft = RG.apply_gate(fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU),
                   RG.member_boxes(device, imgs, sizes), 0.5)
fs = RG.apply_gate(fuse(RG.detect(device, sound, ssz), keys, w, ssz, RG.IOU),
                   RG.member_boxes(device, sound, ssz), 0.5)
names = sorted(targets); G = GRID

tp=np.zeros((len(names),NC,len(G)),np.int32); fp=np.zeros_like(tp); fn=np.zeros_like(tp)
for i,n in enumerate(names):
    for gi,t in enumerate(G):
        tot={c:{"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
        sel=[Prediction(cls=c,conf=s,xyxy=b) for c,s,b in ft[n] if s>=t]
        merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
        for c in range(NC):
            tp[i,c,gi]=tot[c]["tp"]; fp[i,c,gi]=tot[c]["fp"]; fn[i,c,gi]=tot[c]["fn"]

def prec(idx, combo):
    s=sum(tp[idx,c,combo[c]].sum() for c in range(NC)); f=sum(fp[idx,c,combo[c]].sum() for c in range(NC))
    return s/max(s+f,1)
def best_under(idx, bfloor):
    TP,FP,FN=tp[idx].sum(0),fp[idx].sum(0),fn[idx].sum(0); win=None
    for combo in itertools.product(range(len(G)), repeat=NC):
        t=np.array([TP[c,combo[c]] for c in range(NC)]); f=np.array([FP[c,combo[c]] for c in range(NC)])
        m=np.array([FN[c,combo[c]] for c in range(NC)])
        den=t+m; per=np.where(den>0,t/np.maximum(den,1),0.0); fl=[bfloor,0.70,0.70]
        if not all((den[c]==0) or per[c]>=fl[c] for c in range(NC)): continue
        p=t.sum()/max(t.sum()+f.sum(),1)
        if win is None or p>win[0]: win=(p,combo)
    return win
def sound_bpi(combo):
    thr=[G[i] for i in combo]
    return sum(len([x for x in d if x[1]>=thr[x[0]]]) for d in fs.values())/len(fs)

full=np.arange(len(names))
print(f"{'B 下限':>7} {'阈值':>18} {'P':>7} {'B':>6} {'箱/张':>7}  留出 {'更好':>6} {'更差':>6} {'打平':>6} {'均值差':>8}")
out=[]
for bf in (0.70, 0.68, 0.65, 0.60, 0.55):
    wf = best_under(full, bf)
    if not wf: continue
    TPs,FNs=tp[full].sum(0),fn[full].sum(0)
    b_rec = TPs[0,wf[1][0]]/max(TPs[0,wf[1][0]]+FNs[0,wf[1][0]],1)
    rng=np.random.default_rng(20260816); d=[]
    for _ in range(600):
        perm=rng.permutation(len(names)); a,b=perm[:len(perm)//2], perm[len(perm)//2:]
        wc,w0 = best_under(a,bf), best_under(a,0.70)
        if wc and w0: d.append(prec(b,wc[1])-prec(b,w0[1]))
    d=np.array(d)
    print(f"{bf:7.2f} {str([G[i] for i in wf[1]]):>18} {wf[0]:7.3f} {b_rec:6.3f} "
          f"{sound_bpi(wf[1]):7.2f}  n={len(d):3d} {np.mean(d>0):6.1%} {np.mean(d<0):6.1%} "
          f"{np.mean(d==0):6.1%} {d.mean():+8.4f}")
    out.append({"b_floor":bf,"thr":[G[i] for i in wf[1]],"precision":float(wf[0]),
                "b_recall":float(b_rec),"bpi":float(sound_bpi(wf[1])),
                "holdout_better":float(np.mean(d>0)),"holdout_worse":float(np.mean(d<0)),
                "holdout_mean":float(d.mean()),"n":len(d)})
Path("/workspace/exp_cb/candidate_curve.json").write_text(json.dumps(out, indent=2))
print("\n留出验证每行 600 次配对;'更差'比例是关键 —— 被撤回的三项均在 50-74%")
