import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
"""Does the flip view survive the gate on the same terms as the identity view?

The delivered configuration fuses four inputs: two checkpoints, each run on the
image and on its mirror. Boxes from the mirrored pass are un-flipped back into
original coordinates before fusion, and the gate is applied afterwards against a
member box computed on the original image. If the un-flip carried any systematic
offset, the flip view's boxes would sit slightly off the member and the gate
would cut them at a different rate than the identity view's - a silent asymmetry
that would show up nowhere except as a slightly worse gate.

Measured directly: for each view separately, what fraction of its boxes the gate
keeps, and how far their centres sit from the member box centre. Equal rates mean
the un-flip is sound; a gap means it is not.
"""
import sys
import numpy as np
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS
from PIL import Image

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
imgs = sorted(p for p in (DS/"test"/"images").iterdir()
              if p.suffix.lower() in {".jpg",".jpeg",".png"})
sizes = {}
for p in imgs:
    with Image.open(p) as h: sizes[p.name] = h.size
store = RG.detect(device, imgs, sizes)     # keyed (model, view), normalised boxes
mt = RG.member_boxes(device, imgs, sizes)

def frac_in(b, m):
    x1,y1 = max(b[0],m[0]), max(b[1],m[1]); x2,y2 = min(b[2],m[2]), min(b[3],m[3])
    i = max(0.0,x2-x1)*max(0.0,y2-y1); a = (b[2]-b[0])*(b[3]-b[1])
    return i/a if a>0 else 0.0

print(f"{'成员/视图':18s} {'框数':>6} {'门控保留率':>10} {'内含比例中位':>12} {'中心横向偏移':>12}")
for (mdl, view), d in store.items():
    kept = tot = 0; fr = []; dx = []
    for n, r in d.items():
        m = mt.get(n)
        W, H = sizes[n]
        for bn, s in zip(np.asarray(r["boxes"], np.float32).reshape(-1,4), r["scores"]):
            if s < 0.12: continue
            b = (bn * np.array([W,H,W,H])).tolist()
            tot += 1
            if m is None: kept += 1; continue
            f = frac_in(b, m); fr.append(f)
            if f >= 0.5: kept += 1
            # signed horizontal offset of the box centre from the member centre,
            # normalised by member width: a mirroring bug would skew this.
            dx.append((((b[0]+b[2])/2) - ((m[0]+m[2])/2)) / max(m[2]-m[0], 1))
    print(f"{mdl+'/'+view:18s} {tot:6d} {kept/max(tot,1):9.1%} "
          f"{np.median(fr) if fr else float('nan'):11.3f} "
          f"{np.median(dx) if dx else float('nan'):+11.3f}")
print("\n判读: 同一成员的 id 与 hflip 两行若保留率与偏移接近,反变换正确;"
      "\n      若 hflip 的横向偏移与 id 反号且量级相当,说明未正确反变换。")
