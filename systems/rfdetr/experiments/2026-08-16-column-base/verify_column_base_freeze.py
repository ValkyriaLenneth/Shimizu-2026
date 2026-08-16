#!/usr/bin/env python3
"""Recompute every number in FREEZE.md from the frozen weights.

Nothing in the freeze document should be quoted from memory. This runs the
delivered configuration and the one it replaces against the frozen test split
and prints the figures the document claims, so any of them can be checked
independently of the notes that produced them.

It deliberately does not re-run the searches that *found* the configuration --
those live in the other scripts in this directory and take far longer. What it
verifies is the end state: the two members, the fusion settings, the thresholds,
and the resulting recalls and precision.

Usage:  python verify_freeze.py [cuda:0]
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
SOUND = Path("/workspace/sound_20260807/column_base")
MEMBERS = {"ep016": HERE / "checkpoints/column_base_negatives_v1_epoch_016.pth",
           "cp075": HERE / "checkpoints/column_base_copypaste_epoch_075.pth"}
MATCH_IOU, NC, GRADES, TARGET, FLOOR = 0.229, 3, "BCD", 0.70, 0.10

# The member weighting is part of each configuration, not a constant: the
# 2026-08-04 delivery fused the two members equally, and moving to 1:2 was one
# of the two changes that produced the 08-16 revision. Carrying 1:2 back onto
# the old row would score a configuration that was never delivered.
CONFIGS = {
    "2026-08-04 交付":   {"views": ("id",),          "iou": 0.40, "conf": "avg",
                          "w": (1.0, 1.0), "thr": (0.07, 0.15, 0.05), "gate": None},
    "08-16 参数修正":    {"views": ("id",),          "iou": 0.20, "conf": "max",
                          "w": (1.0, 2.0), "thr": (0.07, 0.15, 0.05), "gate": None},
    "08-16 +翻转":       {"views": ("id", "hflip"),  "iou": 0.40, "conf": "max",
                          "w": (1.0, 2.0), "thr": (0.12, 0.20, 0.12), "gate": None},
    "08-16 交付(+门控)": {"views": ("id", "hflip"),  "iou": 0.40, "conf": "max",
                          "w": (1.0, 2.0), "thr": (0.12, 0.20, 0.12), "gate": 0.5},
    # Pending the client's decision in results/14: B alone relaxed to 0.68, which
    # costs one of 47 B boxes and cuts what an inspector reviews by a quarter.
    # Kept here so that row can be recomputed rather than quoted, whether the
    # client accepts it or not.
    "候补 B>=0.68(待客户决定)": {"views": ("id", "hflip"), "iou": 0.40, "conf": "max",
                          "w": (1.0, 2.0), "thr": (0.15, 0.20, 0.12), "gate": 0.5},
}

# The router locates the column base itself; damage sits on the member, so a
# detection lying away from it is wrong-place almost by definition. Images where
# the router finds nothing are passed through ungated - the gate only removes
# what it has a reason to remove, which is why recall does not move.
ROUTER = ("/workspace/handoff_20260707_rfdetr_main/models/rfdetr/router_5class/"
          "selected_precision_p090_epoch049_thr069.pth")
CB_CLASS, ROUTER_THR, GATE_MARGIN = 4, 0.30, 0.10


def member_boxes(device, paths, sizes):
    model = from_checkpoint_matched(ROUTER, device=device, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    out = {}
    for p in paths:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = model.predict(im, threshold=ROUTER_THR)
        cls = np.asarray(det.class_id).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        sel = cls == CB_CLASS
        if not sel.any():
            out[p.name] = None
            continue
        b = xy[sel]
        W, H = sizes[p.name]
        x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        dw, dh = (x2 - x1) * GATE_MARGIN, (y2 - y1) * GATE_MARGIN
        out[p.name] = [max(0, x1 - dw), max(0, y1 - dh), min(W, x2 + dw), min(H, y2 + dh)]
    del model
    torch.cuda.empty_cache()
    return out


def apply_gate(fused, members, g):
    if g is None:
        return fused
    out = {}
    for name, dets in fused.items():
        mb = members.get(name)
        if mb is None:
            out[name] = dets
            continue
        keep = []
        for c, s, b in dets:
            ix1, iy1 = max(b[0], mb[0]), max(b[1], mb[1])
            ix2, iy2 = min(b[2], mb[2]), min(b[3], mb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > 0 and inter / area >= g:
                keep.append((c, s, b))
        out[name] = keep
    return out


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def predict(device, paths, sizes, views):
    store = {}
    for tag, ck in MEMBERS.items():
        model = from_checkpoint_matched(str(ck), device=device, verbose=False)
        ctx = getattr(model, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in views:
            d = {}
            for p in paths:
                with Image.open(p) as h:
                    im = h.convert("RGB")
                if view == "hflip":
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                W, H = im.size
                det = model.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
                bn = np.clip(np.asarray(det.xyxy).reshape(-1, 4)[keep]
                             / np.array([W, H, W, H], np.float32), 0, 1)
                d[p.name] = {"boxes": unflip(bn, view).tolist(),
                             "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist()}
            store[(tag, view)] = d
        del model
        torch.cuda.empty_cache()
    return store


def fuse(store, views, sizes, iou, conf, ratio):
    keys = [(m, v) for m in MEMBERS for v in views]
    w = [ratio[i] / len(views) for i in range(len(MEMBERS)) for _ in views]
    out = {}
    for name in sizes:
        bl = [np.asarray(store[k][name]["boxes"], np.float32).reshape(-1, 4).tolist() for k in keys]
        sl = [list(store[k][name]["scores"]) for k in keys]
        ll = [list(store[k][name]["classes"]) for k in keys]
        W, H = sizes[name]
        if not any(len(b) for b in bl):
            out[name] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=w, iou_thr=iou,
                                        skip_box_thr=0.0, conf_type=conf)
        out[name] = [(int(c), float(x), tuple((np.asarray(bb) * np.array([W, H, W, H])).tolist()))
                     for bb, x, c in zip(b, s, l)]
    return out


def score(fused, targets, thr):
    tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
    for name, dets in fused.items():
        sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dets if s >= thr[c]]
        merge_counts(tot, match_counts(targets[name], sel, MATCH_IOU, NC))
    tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
    fn = sum(v["fn"] for v in tot.values())
    p, r, _ = metric(tp, fp, fn)
    per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
    return p, r, per


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    ngt = sum(len(v) for v in targets.values())
    print(f"冻结测试集: {len(imgs)} 图 / {ngt} 框  (match_iou {MATCH_IOU})\n")

    store = predict(device, imgs, sizes, ("id", "hflip"))
    members_t = member_boxes(device, imgs, sizes)
    print(f"  路由器在 {sum(1 for v in members_t.values() if v)}/{len(members_t)} "
          f"张测试图上定位到柱脚\n")
    print(f"{'配置':26s} {'P':>7} {'R':>7} {'B':>7} {'C':>7} {'D':>7}  四项")
    for label, c in CONFIGS.items():
        f = apply_gate(fuse(store, c["views"], sizes, c["iou"], c["conf"], c["w"]),
                       members_t, c["gate"])
        p, r, per = score(f, targets, c["thr"])
        ok = r >= TARGET and all(v >= TARGET for v in per)
        note = "达标" if ok else ("B 低于 0.70(有意)" if "候补" in label else "未达标")
        print(f"{label:26s} {p:7.3f} {r:7.3f} {per[0]:7.3f} {per[1]:7.3f} {per[2]:7.3f}  {note}")

    sound = sorted(SOUND.glob("*.jpg"))
    if sound:
        ssz = {}
        for p in sound:
            with Image.open(p) as h:
                ssz[p.name] = h.size
        sstore = predict(device, sound, ssz, ("id", "hflip"))
        members_s = member_boxes(device, sound, ssz)
        print(f"\n健全图 {len(sound)} 张的误报:")
        for label, c in CONFIGS.items():
            f = apply_gate(fuse(sstore, c["views"], ssz, c["iou"], c["conf"], c["w"]),
                           members_s, c["gate"])
            fired = sum(1 for d in f.values() if any(x[1] >= c["thr"][x[0]] for x in d))
            boxes = sum(len([x for x in d if x[1] >= c["thr"][x[0]]]) for d in f.values())
            print(f"  {label:26s} 发火 {fired}/{len(sound)} = {fired/len(sound):.0%},"
                  f"{boxes/len(sound):.2f} 框/张")


if __name__ == "__main__":
    main()
