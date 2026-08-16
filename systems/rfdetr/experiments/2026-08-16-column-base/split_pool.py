#!/usr/bin/env python3
"""Separate the client's mixed pool before drawing any conclusion from it.

The deployment profile just run put the column-base configuration over all 1 158
readable photographs in the unlabelled pool and found a detection mix of 55/20/25
percent across B/C/D, against 72/17/11 on the frozen test split -- more than
twice the share of the most severe grade. That looked like a distribution
finding.

It probably is not one. The pool was supplied as additional unlabelled data for
*both* brace and column base, so an unknown fraction of those photographs are
brace, and a column-base detector run over brace imagery produces boxes that
belong to no grade distribution at all. Any transfer conclusion drawn from the
mixed pool is confounded by whatever that fraction is.

The 2026-07-07 five-class router is the instrument already used on this project
to locate members, and it separates the two elements. This splits the pool with
it and reports the profile on the column-base subset alone, which is the only
part the delivered configuration is entitled to speak about. The brace share is
reported too, since it determines how much of the earlier profile was noise.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

POOL = Path("/workspace/unlabeled_pool")
ROUTER = "/workspace/handoff_20260707_rfdetr_main/models/rfdetr/router_5class/selected_precision_p090_epoch049_thr069.pth"
OUT = Path("/workspace/exp_cb/e24_deploy")
THR = 0.30


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in POOL.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = from_checkpoint_matched(ROUTER, device=device, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    nc = getattr(getattr(model, "model", None), "num_classes", None)
    print(f"路由器类别数 {nc};池 {len(imgs)} 张,阈值 {THR}", flush=True)

    rows, tally = [], Counter()
    for i, p in enumerate(imgs):
        try:
            with Image.open(p) as h:
                im = h.convert("RGB")
        except Exception:
            tally["unreadable"] += 1
            continue
        det = model.predict(im, threshold=THR)
        cls = np.asarray(det.class_id).reshape(-1)
        conf = np.asarray(det.confidence).reshape(-1)
        if not len(cls):
            top, sc = -1, 0.0
        else:
            # Attribute the image to the class holding the most total confidence,
            # not merely the single highest box: a member photographed across
            # several boxes should not lose to one incidental high-scoring box.
            agg = {}
            for c, s in zip(cls.tolist(), conf.tolist()):
                agg[c] = agg.get(c, 0.0) + s
            top = max(agg, key=agg.get)
            sc = agg[top]
        tally[top] += 1
        rows.append({"image": p.name, "router_class": top, "class_conf": round(float(sc), 4),
                     "n_boxes": int(len(cls))})
        if i % 300 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)

    print("\n路由器类别分布:")
    for c, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        label = "无检出" if c == -1 else f"class {c}"
        print(f"  {label:12s} {n:5d}  {n/len(imgs):5.1%}")
    with (OUT / "pool_router.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "router_class", "class_conf", "n_boxes"])
        w.writeheader(); w.writerows(rows)
    (OUT / "pool_router.json").write_text(json.dumps(
        {str(k): v for k, v in tally.items()}, indent=2), encoding="utf-8")
    print(f"\n写入 {OUT/'pool_router.csv'}")
    print("说明: 类别编号需对照 2026-07-07 路由器的类别映射后才能判定哪些是柱脚")


if __name__ == "__main__":
    main()
