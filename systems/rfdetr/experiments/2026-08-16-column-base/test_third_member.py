import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
"""Put the most divergent candidates in the third WBF slot, at a fixed epoch.

The screen ranked 38 earlier runs by how much they disagree with the shipped
pair, since WBF converts disagreement into coverage. The six most divergent go
into the ensemble here.

Each candidate contributes its *last* epoch, chosen in advance rather than by
score. Selecting the best epoch per candidate would repeat the mistake that
killed three precision claims today: a maximum over many cells read off the same
45 images that chose it. A fixed epoch costs some headroom and buys a number that
means what it says.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

CANDS = ["alldamage", "alldamage_bcd", "neg_rep28_v2", "neg_add28_v2", "syn25", "neg_rep56_v2"]
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

def unflip(b, v):
    if v != "hflip" or not len(b): return b
    o=b.copy(); o[:,0],o[:,2] = 1.0-b[:,2], 1.0-b[:,0]; return o

def det_views(ck, paths, szs):
    m = from_checkpoint_matched(str(ck), device=device, verbose=False)
    ctx=getattr(m,"model",None)
    if ctx is not None and hasattr(ctx,"device"): ctx.device=torch.device(device)
    out={}
    for view in RG.VIEWS:
        d={}
        for p in paths:
            with Image.open(p) as h: im=h.convert("RGB")
            if view=="hflip": im=im.transpose(Image.FLIP_LEFT_RIGHT)
            W,H=im.size
            dt=m.predict(im, threshold=RG.FLOOR)
            cls=np.asarray(dt.class_id).reshape(-1); keep=cls<NC
            bn=np.clip(np.asarray(dt.xyxy).reshape(-1,4)[keep]/np.array([W,H,W,H],np.float32),0,1)
            d[p.name]={"boxes":unflip(bn,view).tolist(),
                       "scores":np.asarray(dt.confidence).reshape(-1)[keep].tolist(),
                       "classes":cls[keep].tolist()}
        out[view]=d
    del m; torch.cuda.empty_cache(); return out

def fuse_all(stores, weights, szs):
    out={}
    for n in szs:
        bl,sl,ll=[],[],[]
        for st in stores:
            for v in RG.VIEWS:
                r=st[v][n]
                bl.append(np.asarray(r["boxes"],np.float32).reshape(-1,4).tolist())
                sl.append(list(r["scores"])); ll.append(list(r["classes"]))
        W,H=szs[n]
        if not any(len(b) for b in bl): out[n]=[]; continue
        b,s,l=weighted_boxes_fusion(bl,sl,ll,weights=weights,iou_thr=RG.IOU,
                                    skip_box_thr=0.0,conf_type=RG.CONF)
        out[n]=[(int(c),float(x),tuple((np.asarray(bb)*np.array([W,H,W,H])).tolist()))
                for bb,x,c in zip(b,s,l)]
    return out

def score(ft, fs):
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
            fired=sum(1 for d in fs.values() if any(x[1]>=combo[x[0]] for x in d))
            boxes=sum(len([x for x in d if x[1]>=combo[x[0]]]) for d in fs.values())
            best=(p,r,per,list(combo),fired/len(fs),boxes/len(fs))
    return best

mt=RG.member_boxes(device,imgs,sizes); ms=RG.member_boxes(device,sound,ssz)
ship_t=[det_views(c,imgs,sizes) for c in MEMBERS.values()]
ship_s=[det_views(c,sound,ssz) for c in MEMBERS.values()]
W2=[0.5,0.5,1.0,1.0]
b=score(RG.apply_gate(fuse_all(ship_t,W2,sizes),mt,0.5),
        RG.apply_gate(fuse_all(ship_s,W2,ssz),ms,0.5))
print(f"{'配置':28s} {'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'箱/张':>7}")
print(f"{'现交付(2 成员)':28s} {b[0]:7.3f} {b[1]:7.3f} {b[2][0]:6.3f} {b[2][1]:6.3f} "
      f"{b[2][2]:6.3f} {b[4]:5.0%} {b[5]:7.2f}")
rows=[]
W3=[0.5,0.5,1.0,1.0,1.0,1.0]
for name in CANDS:
    cks=sorted(Path(f"/workspace/exp_cb/{name}").glob("epoch_pth/checkpoint_epoch_*.pth"))
    if not cks: print(f"{name}: 无 ckpt"); continue
    ck=cks[-1]; ep=int(ck.stem.split("_")[-1])
    try:
        dt=det_views(ck,imgs,sizes); ds=det_views(ck,sound,ssz)
    except Exception as e:
        print(f"{name}: 载入失败 {type(e).__name__}"); continue
    s3=score(RG.apply_gate(fuse_all(ship_t+[dt],W3,sizes),mt,0.5),
             RG.apply_gate(fuse_all(ship_s+[ds],W3,ssz),ms,0.5))
    if not s3:
        print(f"{name+f' (ep{ep})':28s} 无四项达标点"); continue
    print(f"{name+f' (ep{ep})':28s} {s3[0]:7.3f} {s3[1]:7.3f} {s3[2][0]:6.3f} "
          f"{s3[2][1]:6.3f} {s3[2][2]:6.3f} {s3[4]:5.0%} {s3[5]:7.2f}")
    rows.append({"run":name,"epoch":ep,"precision":s3[0],"recall":s3[1],
                 "fire":s3[4],"bpi":s3[5]})
best=max(rows,key=lambda r:r["precision"]) if rows else None
print(f"\n基准 {b[0]:.3f};最好候选 " + (f"{best['run']} {best['precision']:.3f}" if best else "无"))
if best and best["precision"]>b[0]:
    print("-> 超过基准,须再经留出验证方可采纳")
else:
    print("-> 无候选超过基准:两成员集成仍是最优")
Path("/workspace/exp_cb/third_member_test.json").write_text(
    json.dumps({"baseline":b[0],"candidates":rows}, indent=2))
