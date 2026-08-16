"""How stable is the candidate operating point across resamples of the test set?

The client is being asked to accept B recall at 0.681 in exchange for precision
0.455 and 1.03 boxes per sound image. Those figures come from 45 images and 29
sound photographs, and every delivery figure on this project carries the caveat
that four-target feasibility holds on only 46% of bootstrap resamples. The
candidate deliberately steps outside that constraint on B, so its own stability
is a different question and has not been asked.

Two things are measured on the same resamples, for the delivered point and the
candidate alike: the precision each reports, and how often B stays above 0.68 for
the candidate and above 0.70 for the delivered point. The second matters more --
if B slips below the promised floor on many resamples, the floor is not something
either configuration can be said to hold.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)
from PIL import Image

POINTS = {"现交付 B>=0.70": (0.12, 0.20, 0.12), "候补 B>=0.68": (0.15, 0.20, 0.12)}
FLOOR_OF = {"现交付 B>=0.70": 0.70, "候补 B>=0.68": 0.68}
NBOOT = 2000
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

def counts(thr):
    tp=np.zeros((len(names),NC),np.int32); fp=np.zeros_like(tp); fn=np.zeros_like(tp)
    for i,n in enumerate(names):
        tot={c:{"tp":0,"fp":0,"fn":0,"gt":0,"pred":0} for c in range(NC)}
        sel=[Prediction(cls=c,conf=s,xyxy=b) for c,s,b in ft[n] if s>=thr[c]]
        merge_counts(tot, match_counts(targets[n], sel, RG.MATCH_IOU, NC))
        for c in range(NC):
            tp[i,c]=tot[c]["tp"]; fp[i,c]=tot[c]["fp"]; fn[i,c]=tot[c]["fn"]
    return tp,fp,fn

T = {k: counts(v) for k, v in POINTS.items()}
rng = np.random.default_rng(20260816)
idxs = [rng.integers(0, len(names), len(names)) for _ in range(NBOOT)]
print(f"{'工作点':18s} {'P 中位':>8} {'P 90%区间':>18} {'B 守住下限':>10} {'四项全守住':>10}")
out={}
for k,(tp,fp,fn) in T.items():
    ps, holdB, hold4 = [], 0, 0
    fl = FLOOR_OF[k]
    for idx in idxs:
        t,f,m = tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0)
        st,sf,sm = t.sum(), f.sum(), m.sum()
        ps.append(st/max(st+sf,1))
        per = [t[c]/max(t[c]+m[c],1) for c in range(NC)]
        holdB += per[0] >= fl
        hold4 += (per[0] >= fl) and (per[1] >= 0.70) and (per[2] >= 0.70) \
                 and (st/max(st+sm,1) >= 0.70)
    ps=np.array(ps); lo,hi=np.percentile(ps,[5,95])
    print(f"{k:18s} {np.median(ps):8.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"{holdB/NBOOT:9.0%} {hold4/NBOOT:10.0%}")
    out[k]={"p_median":float(np.median(ps)),"p90":[float(lo),float(hi)],
            "hold_B":holdB/NBOOT,"hold_all":hold4/NBOOT}
Path("/workspace/exp_cb/candidate_robustness.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print("\n说明: '守住下限' = 该重采样上 B 仍达到各自承诺的下限(0.70 / 0.68)")
