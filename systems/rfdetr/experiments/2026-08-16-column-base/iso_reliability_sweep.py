"""Find the best operating point at a fixed promise reliability.

The candidate and the delivered point turned out to hold their respective B
promises on exactly 52% of resamples: lowering the promise from 0.70 to 0.68
raised the B threshold from 0.12 to 0.15, and the two effects cancelled. That
cancellation is a mechanism, not a coincidence, and it means the choice is not
confined to those two points. Every threshold triple sits somewhere on a
reliability-versus-precision surface, and the pair examined so far happen to lie
on the same contour.

So the question becomes: along the contour where the promise is kept as often as
the delivered configuration keeps its own -- 52% of resamples -- which point
reports the best numbers? And what does the whole curve look like, so the client
can see what a stricter or looser promise would actually buy?

For each threshold triple the promised B floor is taken to be its own measured B
recall on the full split, since that is what would be written into the delivery.
Reliability is then how often a resample still meets that floor, which makes the
comparison fair across points that promise different things.
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

NBOOT = 400
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
keys=[(m,v) for m in MEMBERS for v in RG.VIEWS]
w=[RG.RATIO[i]/len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
mt, ms = RG.member_boxes(device,imgs,sizes), RG.member_boxes(device,sound,ssz)
ft = RG.apply_gate(fuse(RG.detect(device,imgs,sizes),keys,w,sizes,RG.IOU), mt, 0.5)
fs = RG.apply_gate(fuse(RG.detect(device,sound,ssz),keys,w,ssz,RG.IOU), ms, 0.5)
names=sorted(targets); G=GRID

tp=np.zeros((len(names),NC,len(G)),np.int32); fp=np.zeros_like(tp); fn=np.zeros_like(tp)
for i,n in enumerate(names):
    for gi,t in enumerate(G):
        tot={c:{"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
        sel=[Prediction(cls=c,conf=s,xyxy=b) for c,s,b in ft[n] if s>=t]
        merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
        for c in range(NC):
            tp[i,c,gi]=tot[c]["tp"]; fp[i,c,gi]=tot[c]["fp"]; fn[i,c,gi]=tot[c]["fn"]

rng=np.random.default_rng(20260816)
idxs=[rng.integers(0,len(names),len(names)) for _ in range(NBOOT)]
full=np.arange(len(names))

def bpi(combo):
    thr=[G[i] for i in combo]
    return sum(len([x for x in d if x[1]>=thr[x[0]]]) for d in fs.values())/len(fs)

rows=[]
for combo in itertools.product(range(len(G)), repeat=NC):
    t,f,m = tp[full].sum(0), fp[full].sum(0), fn[full].sum(0)
    per=[t[c,combo[c]]/max(t[c,combo[c]]+m[c,combo[c]],1) for c in range(NC)]
    # C and D keep the delivered promise; only B is allowed to move.
    if per[1] < 0.70 or per[2] < 0.70: continue
    if per[0] < 0.60: continue
    st=sum(t[c,combo[c]] for c in range(NC)); sf=sum(f[c,combo[c]] for c in range(NC))
    sm=sum(m[c,combo[c]] for c in range(NC))
    if st/max(st+sm,1) < 0.60: continue
    P=st/max(st+sf,1)
    hold=0
    for idx in idxs:
        tb=tp[idx,0,combo[0]].sum(); mb=fn[idx,0,combo[0]].sum()
        hold += (tb/max(tb+mb,1)) >= per[0]
    rows.append({"thr":[G[i] for i in combo],"P":P,"B":per[0],"C":per[1],"D":per[2],
                 "rel":hold/NBOOT,"bpi":bpi(combo)})

print(f"共 {len(rows)} 个候选点(C/D 守 0.70,B >= 0.60)\n")
print(f"{'可靠性带':>10} {'点数':>5} {'最佳 P':>7} {'该点 B':>7} {'箱/张':>7} {'阈值':>18}")
for lo,hi in ((0.60,1.01),(0.55,0.60),(0.50,0.55),(0.45,0.50),(0.40,0.45)):
    band=[r for r in rows if lo <= r["rel"] < hi]
    if not band: continue
    b=max(band,key=lambda r:r["P"])
    print(f"{lo:.2f}-{hi:.2f} {len(band):5d} {b['P']:7.3f} {b['B']:7.3f} {b['bpi']:7.2f} "
          f"{str(b['thr']):>18}")
cur=[r for r in rows if r["thr"]==[0.12,0.20,0.12]]
cand=[r for r in rows if r["thr"]==[0.15,0.20,0.12]]
for lab,r in (("现交付",cur),("候补",cand)):
    if r: print(f"\n{lab}: P {r[0]['P']:.3f}  B {r[0]['B']:.3f}  可靠性 {r[0]['rel']:.0%}  "
                f"{r[0]['bpi']:.2f} 箱/张")
same=[r for r in rows if abs(r["rel"]-(cur[0]["rel"] if cur else 0.52))<0.03]
if same:
    b=max(same,key=lambda r:r["P"])
    print(f"\n同可靠性带内最佳: P {b['P']:.3f}  B {b['B']:.3f}  {b['bpi']:.2f} 箱/张  阈值 {b['thr']}")
Path("/workspace/exp_cb/iso_reliability.json").write_text(json.dumps(rows,indent=2))
