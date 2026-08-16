"""Is the candidate point's precision real, or another maximum picked from a grid?

Relaxing B to 0.68 moves precision from 0.395 to 0.442 and sound-image boxes from
1.66 to 1.24. Three precision claims of exactly this shape died today under
holdout validation - the flip view, the router gate, and the brightness views -
because each was the best of many cells read off the same 45 images that chose it.

The candidate is not selected the same way: its threshold triple (0.15, 0.20,
0.12) follows from one stated decision, "let B fall to 0.68", rather than from a
search for the highest number. But that is an argument about provenance, and the
question is empirical. Two checks:

  paired bootstrap   candidate against the delivered point at their own fixed
                     thresholds, resampling the 45 images
  holdout            choose the B threshold on half the images under the B>=0.68
                     rule, score on the other half, paired against the delivered
                     point chosen the same way
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

CUR, CAND = (0.12, 0.20, 0.12), (0.15, 0.20, 0.12)
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
imgs = sorted(p for p in (DS/"test"/"images").iterdir()
              if p.suffix.lower() in {".jpg",".jpeg",".png"})
sizes, targets = {}, {}
for p in imgs:
    with Image.open(p) as h: sizes[p.name] = h.size
    targets[p.name] = read_targets(DS/"test"/"labels"/f"{p.stem}.txt", *sizes[p.name])
keys = [(m,v) for m in MEMBERS for v in RG.VIEWS]
w = [RG.RATIO[i]/len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
ft = RG.apply_gate(fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU),
                   RG.member_boxes(device, imgs, sizes), 0.5)
names = sorted(targets)
G = GRID

def tab(fused):
    tp=np.zeros((len(names),NC,len(G)),np.int32); fp=np.zeros_like(tp); fn=np.zeros_like(tp)
    for i,n in enumerate(names):
        for gi,t in enumerate(G):
            tot={c:{"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
            sel=[Prediction(cls=c,conf=s,xyxy=b) for c,s,b in fused[n] if s>=t]
            merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
            for c in range(NC):
                tp[i,c,gi]=tot[c]["tp"]; fp[i,c,gi]=tot[c]["fp"]; fn[i,c,gi]=tot[c]["fn"]
    return tp,fp,fn
T = tab(ft)

def prec(idx, thr):
    tp,fp,_=T; gi=[G.index(t) for t in thr]
    s=sum(tp[idx,c,gi[c]].sum() for c in range(NC)); f=sum(fp[idx,c,gi[c]].sum() for c in range(NC))
    return s/max(s+f,1)

print(f"全量: 现交付 {prec(np.arange(len(names)),CUR):.4f}  候补 {prec(np.arange(len(names)),CAND):.4f}")
rng=np.random.default_rng(20260816); d=[]
for _ in range(2000):
    idx=rng.integers(0,len(names),len(names))
    d.append(prec(idx,CAND)-prec(idx,CUR))
d=np.array(d); lo,hi=np.percentile(d,[2.5,97.5])
print(f"固定阈值配对自助 2000 次: 均值 {d.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
      f"候补更好 {np.mean(d>0):.1%}")

def best_under(idx, bfloor):
    tp,fp,fn=T; TP,FP,FN=tp[idx].sum(0),fp[idx].sum(0),fn[idx].sum(0)
    win=None
    for combo in itertools.product(range(len(G)), repeat=NC):
        t=np.array([TP[c,combo[c]] for c in range(NC)]); f=np.array([FP[c,combo[c]] for c in range(NC)])
        m=np.array([FN[c,combo[c]] for c in range(NC)])
        den=t+m; per=np.where(den>0,t/np.maximum(den,1),0.0)
        floors=[bfloor,0.70,0.70]
        if not all((den[c]==0) or per[c]>=floors[c] for c in range(NC)): continue
        p=t.sum()/max(t.sum()+f.sum(),1)
        if win is None or p>win[0]: win=(p,combo)
    return win

rng=np.random.default_rng(20260816); dd=[]
for _ in range(200):
    perm=rng.permutation(len(names)); a,b=perm[:len(perm)//2], perm[len(perm)//2:]
    wc, w0 = best_under(a,0.68), best_under(a,0.70)
    if not wc or not w0: continue
    dd.append(prec(b,tuple(G[i] for i in wc[1])) - prec(b,tuple(G[i] for i in w0[1])))
dd=np.array(dd); lo2,hi2=np.percentile(dd,[2.5,97.5])
print(f"  其中更差 {float((dd<0).mean()):.1%} 打平 {float((dd==0).mean()):.1%}")
print(f"留出验证 {len(dd)} 次配对(半分选阈值/半分评估): 均值 {dd.mean():+.4f}  "
      f"95% [{lo2:+.4f}, {hi2:+.4f}]  候补更好 {np.mean(dd>0):.1%}")
print(f"结论: {'两项检验均支持' if lo>0 and dd.mean()>0 else '需谨慎解读'}")
Path("/workspace/exp_cb/candidate_check.json").write_text(json.dumps(
  {"paired_mean":float(d.mean()),"paired_ci":[float(lo),float(hi)],
   "holdout_mean":float(dd.mean()),"holdout_ci":[float(lo2),float(hi2)],
   "holdout_n":len(dd)}, indent=2))
