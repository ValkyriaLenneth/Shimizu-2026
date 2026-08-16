#!/usr/bin/env python3
"""Suppress boxes that fall outside the member the router locates.

The error breakdown found that 78% of the delivered configuration's false
positives are wrong-place -- boxes on something that is not damage at all --
against 15% wrong-grade. Threshold search cannot reach them: all 128 feasible
operating points were enumerated and the delivered one already minimises review
burden. Three training interventions aimed at the same failure either did nothing
or made it worse.

One lever remains untried, and it uses an asset the project already ships. The
five-class router locates the column base itself, and damage sits on the member.
A box drawn away from the located member is wrong-place almost by definition, so
the router's box is a spatial prior the detector never sees.

The risk is symmetric and has to be measured rather than assumed: the router
finds a column base in 86.7% of the frozen test images, so gating would discard
every detection in the remaining 13% and cost recall exactly where the delivery
claim has no slack. Images where the router finds nothing are therefore passed
through ungated -- the gate can only remove boxes it has a reason to remove.

Reported across gate strengths, from requiring the detection to sit inside the
member to merely touching it, with the four-target constraint re-solved under
each so the comparison is at matched recall rather than matched thresholds.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

SOUND = Path("/workspace/sound_20260807/column_base")
ROUTER = "/workspace/handoff_20260707_rfdetr_main/models/rfdetr/router_5class/selected_precision_p090_epoch049_thr069.pth"
CB_CLASS, ROUTER_THR = 4, 0.30
VIEWS, RATIO, IOU, CONF, FLOOR = ("id", "hflip"), (1.0, 2.0), 0.40, "max", 0.10
MATCH_IOU, TARGET, GRADES = 0.229, 0.70, "BCD"
# Fraction of the detection's own area that must lie inside the member box.
GATES = [None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
MARGIN = 0.10          # member box is dilated by this fraction before gating


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def detect(device, paths, sizes):
    store = {}
    for tag, ck in MEMBERS.items():
        m = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in VIEWS:
            d = {}
            for p in paths:
                with Image.open(p) as h:
                    im = h.convert("RGB")
                if view == "hflip":
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                W, H = im.size
                det = m.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
                bn = np.clip(np.asarray(det.xyxy).reshape(-1, 4)[keep]
                             / np.array([W, H, W, H], np.float32), 0, 1)
                d[p.name] = {"boxes": unflip(bn, view).tolist(),
                             "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist()}
            store[(tag, view)] = d
        del m; torch.cuda.empty_cache()
    return store


def member_boxes(device, paths, sizes):
    m = from_checkpoint_matched(ROUTER, device=device, verbose=False)
    ctx = getattr(m, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    out = {}
    for p in paths:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = m.predict(im, threshold=ROUTER_THR)
        cls = np.asarray(det.class_id).reshape(-1)
        conf = np.asarray(det.confidence).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        sel = cls == CB_CLASS
        if not sel.any():
            out[p.name] = None
            continue
        # Union of all located members, dilated: a column base spanning two boxes
        # should not have half of it gated away.
        b = xy[sel]
        W, H = sizes[p.name]
        x1, y1 = b[:, 0].min(), b[:, 1].min()
        x2, y2 = b[:, 2].max(), b[:, 3].max()
        dw, dh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        out[p.name] = [max(0, x1 - dw), max(0, y1 - dh), min(W, x2 + dw), min(H, y2 + dh)]
    del m; torch.cuda.empty_cache()
    return out


def inside_frac(box, member):
    x1, y1 = max(box[0], member[0]), max(box[1], member[1])
    x2, y2 = min(box[2], member[2]), min(box[3], member[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    return i / a if a > 0 else 0.0


def apply_gate(fused, members, g):
    if g is None:
        return fused
    out = {}
    for name, dets in fused.items():
        mb = members.get(name)
        # No member located: pass through. The gate only removes what it can
        # justify removing.
        out[name] = dets if mb is None else [d for d in dets if inside_frac(d[2], mb) >= g]
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sound = sorted(p for p in SOUND.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, ssz, targets = {}, {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    for p in sound:
        with Image.open(p) as h:
            ssz[p.name] = h.size
    keys = [(m, v) for m in MEMBERS for v in VIEWS]
    w = [RATIO[i] / len(VIEWS) for i in range(len(MEMBERS)) for _ in VIEWS]
    ft = fuse(detect(device, imgs, sizes), keys, w, sizes, IOU)
    fs = fuse(detect(device, sound, ssz), keys, w, ssz, IOU)
    mt = member_boxes(device, imgs, sizes)
    ms = member_boxes(device, sound, ssz)
    print(f"路由器定位到柱脚: 测试集 {sum(1 for v in mt.values() if v)}/{len(mt)}, "
          f"健全图 {sum(1 for v in ms.values() if v)}/{len(ms)}\n")

    print(f"{'门控':>8} {'最高P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} "
          f"{'阈值':>18} {'健全发火':>9} {'框/张':>7}")
    rows = []
    for g in GATES:
        gt_ = apply_gate(ft, mt, g)
        gs_ = apply_gate(fs, ms, g)
        best = None
        for combo in itertools.product(GRID, repeat=NC):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            for name, dets in gt_.items():
                sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dets if s >= combo[c]]
                merge_counts(tot, match_counts(targets[name], sel, MATCH_IOU, NC))
            tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
            fn = sum(v["fn"] for v in tot.values())
            p, r, _ = metric(tp, fp, fn)
            per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
            if r >= TARGET and all(v >= TARGET for v in per) and (best is None or p > best[0]):
                best = (p, r, per, combo)
        if best is None:
            print(f"{str(g):>8}   无四项达标点")
            continue
        p, r, per, combo = best
        fired = sum(1 for d in gs_.values() if any(x[1] >= combo[x[0]] for x in d))
        boxes = sum(len([x for x in d if x[1] >= combo[x[0]]]) for d in gs_.values())
        print(f"{str(g):>8} {p:7.3f} {r:7.3f} {per[0]:6.3f} {per[1]:6.3f} {per[2]:6.3f} "
              f"{str(list(combo)):>18} {fired/len(gs_):8.0%} {boxes/len(gs_):7.2f}")
        rows.append({"gate": g, "precision": p, "recall": r, "per": per,
                     "thr": list(combo), "fire": fired / len(gs_), "bpi": boxes / len(gs_)})
    Path("/workspace/exp_cb/router_gate.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    base = next((r for r in rows if r["gate"] is None), None)
    top = max((r for r in rows if r["gate"] is not None), key=lambda r: r["precision"], default=None)
    if base and top:
        print(f"\n无门控 {base['precision']:.3f} -> 最佳门控 {top['precision']:.3f} "
              f"(g={top['gate']})  差 {top['precision']-base['precision']:+.3f}")
        print(f"健全图 {base['bpi']:.2f} -> {top['bpi']:.2f} 框/张 "
              f"({top['bpi']-base['bpi']:+.2f})")


if __name__ == "__main__":
    main()
