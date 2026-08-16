#!/usr/bin/env python3
"""Does the pool contain column bases at all, or only other elements?

The router assigned 86% of the client's 1 159 photographs to walls and RC
columns and only 1.5% to column base, which invalidated the deployment profile
built on that pool. But the router was asked the wrong question. It was scored by
which class held the most confidence in each image -- the *dominant* element --
and the pool is plausibly wide-scene site photography where a column base sits at
the foot of a frame otherwise filled by wall. Dominance would then be a fact
about framing, not about whether the target is present.

Presence is the question that decides whether the pool is usable: in how many
photographs does the router find a column base anywhere, at any confidence worth
acting on, regardless of what else is in frame. If presence is high while
dominance is low, the pool is wide-scene imagery containing the target and the
profile can be rebuilt on the subset. If presence is also low, the pool is simply
different data than assumed, and both the profile and the annotation queue stay
retracted.

Also reports the box area fraction of the located column bases, since a target
occupying a few percent of the frame is a different detection problem from the
member-centred crops the delivered model was trained and measured on -- which
would itself explain a confidence drop without any distribution shift.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

POOL = Path("/workspace/unlabeled_pool")
DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid/test/images")
ROUTER = "/workspace/handoff_20260707_rfdetr_main/models/rfdetr/router_5class/selected_precision_p090_epoch049_thr069.pth"
CB = 4          # calibrated on 45 known column-base images: 84.4% dominant
OUT = Path("/workspace/exp_cb/e24_deploy")
LEVELS = [0.20, 0.30, 0.50, 0.69]      # 0.69 is the router's own selected threshold


def scan(model, imgs, label):
    rows = []
    for i, p in enumerate(imgs):
        try:
            with Image.open(p) as h:
                im = h.convert("RGB"); W, H = im.size
        except Exception:
            continue
        det = model.predict(im, threshold=min(LEVELS))
        cls = np.asarray(det.class_id).reshape(-1)
        conf = np.asarray(det.confidence).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        m = cls == CB
        best = float(conf[m].max()) if m.any() else 0.0
        area = 0.0
        if m.any():
            j = int(np.argmax(conf[m]))
            b = xy[m][j]
            area = float(max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])) / (W * H))
        rows.append({"image": p.name, "cb_conf": round(best, 4),
                     "cb_area_frac": round(area, 4)})
        if i % 300 == 0:
            print(f"  {label} {i}/{len(imgs)}", flush=True)
    return rows


def report(rows, label):
    n = len(rows)
    print(f"\n{label} ({n} 张) —— 画面中是否存在柱脚:")
    for t in LEVELS:
        k = sum(1 for r in rows if r["cb_conf"] >= t)
        print(f"  置信度 >= {t:.2f}: {k:5d}  {k/max(n,1):5.1%}")
    a = np.array([r["cb_area_frac"] for r in rows if r["cb_conf"] >= 0.30])
    if len(a):
        print(f"  被定位柱脚的画面占比: 中位 {np.median(a):.1%}, "
              f"p10 {np.percentile(a,10):.1%}, p90 {np.percentile(a,90):.1%}")
    return {"n": n, **{f"present_{t}": sum(1 for r in rows if r["cb_conf"] >= t) / max(n, 1)
                       for t in LEVELS},
            "median_area": float(np.median(a)) if len(a) else None}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    model = from_checkpoint_matched(ROUTER, device=device, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)

    test = sorted(p for p in DS.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pool = sorted(p for p in POOL.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tr = scan(model, test, "test")
    pr = scan(model, pool, "pool")
    a = report(tr, "冻结测试集(已知全部为柱脚,作为参照)")
    b = report(pr, "客户无标注池")

    with (OUT / "pool_presence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "cb_conf", "cb_area_frac"])
        w.writeheader(); w.writerows(pr)
    (OUT / "pool_presence.json").write_text(json.dumps({"test": a, "pool": b}, indent=2),
                                            encoding="utf-8")
    k = sum(1 for r in pr if r["cb_conf"] >= 0.30)
    print(f"\n判定: 池中 {k} 张 ({k/max(len(pr),1):.1%}) 含柱脚 —— "
          f"{'可在该子集上重建部署画像' if k >= 100 else '数量不足,池与假设不符,画像与标注队列维持撤回'}")


if __name__ == "__main__":
    main()
