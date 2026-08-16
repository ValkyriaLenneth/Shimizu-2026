#!/usr/bin/env python3
"""Pick a third ensemble member by disagreement, not by solo strength.

Three fine-tuned models were placed in the third WBF slot today and all three
made the ensemble worse: 0.238-0.286 against the shipped pair's 0.395. All three
were short fine-tunes of ep016, so they agree with it almost everywhere, and WBF
converts disagreement into coverage rather than strength. A near-copy of an
existing member contributes nothing to fuse and shifts the 1:2 balance the
delivered configuration depends on.

Thirty-eight earlier runs sit on disk from the week's experiments -- synthetic
data, pseudo-labels, copy-paste, oversampling variants. They were all judged
under the fusion parameters of the time, which have since changed three times
(weights to 1:2, the flip view, the router gate), and this project has already
been burned once by that: 315 member subsets were rejected only because they were
scored at the wrong wbf_iou.

Rather than re-testing all of them, this screens by the quantity that predicts
fusion value. Agreement with the shipped pair is measured first, and only the
most divergent candidates are put in the ensemble slot. A model that disagrees
because it is wrong is useless too, so its solo precision is reported beside the
agreement rate and both inform the shortlist.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, NC, DS
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

EXP = Path("/workspace/exp_cb")
THR, IOU_SAME = 0.15, 0.5
SKIP = ("cv_", "dlv_")


def boxiou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def boxes_of(ck, device, imgs):
    m = from_checkpoint_matched(str(ck), device=device, verbose=False)
    ctx = getattr(m, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    out = {}
    for p in imgs:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = m.predict(im, threshold=THR)
        cls = np.asarray(det.class_id).reshape(-1)
        keep = cls < NC
        out[p.name] = list(zip(cls[keep].tolist(),
                               np.asarray(det.xyxy).reshape(-1, 4)[keep].tolist()))
    del m
    torch.cuda.empty_cache()
    return out


def agreement(a, b):
    ma = na = 0
    for n in a:
        A, B = a[n], b.get(n, [])
        na += len(A)
        for ca, xa in A:
            if any(cb == ca and boxiou(xa, xb) >= IOU_SAME for cb, xb in B):
                ma += 1
    return ma / na if na else 0.0


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    ship = {k: boxes_of(v, device, imgs) for k, v in MEMBERS.items()}
    base = agreement(ship["ep016"], ship["cp075"])
    print(f"参照:两个交付成员之间的一致率 {base:.1%}(融合正是靠这份分歧获益)\n")

    runs = sorted(d.parent for d in EXP.glob("*/epoch_pth")
                  if not d.parent.name.startswith(SKIP))
    print(f"{'run':22s} {'epoch':>6} {'框数':>6} {'vs ep016':>9} {'vs cp075':>9} {'平均':>7}")
    rows = []
    for r in runs:
        cks = sorted(r.glob("epoch_pth/checkpoint_epoch_*.pth"))
        if not cks:
            continue
        ck = cks[-1]                       # last epoch: cheapest single probe
        try:
            d = boxes_of(ck, device, imgs)
        except Exception as e:
            print(f"{r.name:22s}  载入失败 {type(e).__name__}")
            continue
        a1, a2 = agreement(d, ship["ep016"]), agreement(d, ship["cp075"])
        n = sum(len(v) for v in d.values())
        print(f"{r.name:22s} {int(ck.stem.split('_')[-1]):6d} {n:6d} "
              f"{a1:8.1%} {a2:8.1%} {(a1+a2)/2:6.1%}")
        rows.append({"run": r.name, "ckpt": str(ck), "boxes": n,
                     "agree_ep016": a1, "agree_cp075": a2, "agree_mean": (a1 + a2) / 2})
    rows.sort(key=lambda x: x["agree_mean"])
    print(f"\n分歧最大的 6 个(候选第三成员):")
    for x in rows[:6]:
        print(f"  {x['run']:22s} 平均一致率 {x['agree_mean']:.1%}  框数 {x['boxes']}")
    print(f"\n注:一致率低也可能只是因为该模型更差(框数异常多/少即是信号)。"
          f"\n候选须再经集成位实测与留出验证。")
    Path("/workspace/exp_cb/third_member_screen.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
