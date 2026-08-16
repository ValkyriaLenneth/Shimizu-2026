#!/usr/bin/env python3
"""Does the frozen-backbone recipe pay off under the delivery protocol?

Cross-validation measured +0.188 precision and -0.41 boxes per sound image for
freezing the DINOv2 backbone, the only positive result among five training-side
interventions. But those folds mix the frozen test split into training, so their
absolute values are inflated for every arm alike; a paired difference says the
recipe is better than its control, not what the delivery would report.

This trains nothing. It scores checkpoints that were trained on the 179 training
images alone, against the 45-image protocol, in the two roles that matter:

  single        the model on its own, against the delivered single model 0.159
  ensemble      the model as a third WBF member beside the two shipped
                checkpoints, against the delivered ensemble

Both seeds are reported separately. A training-dependent claim on this project
needs that: single-run reproducibility is 0.06-0.10, and three claims died this
week when a second seed was run. If the two seeds disagree in direction, the
recipe has not been shown to help the delivery whatever the CV said.

The router gate is applied exactly as the delivered configuration applies it.
"""
from __future__ import annotations
import os as _os, sys as _sys
# Resolve sibling modules from wherever this package was extracted,
# falling back to the authoring location if it happens to exist.
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here, "/workspace/scripts_exp"):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import itertools, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
import router_gate as RG
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
SOUND = Path("/workspace/sound_20260807/column_base")
SHIPPED = {"ep016": "/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth",
           "cp075": "/workspace/handoff_20260804/checkpoints/column_base_copypaste_epoch_075.pth"}
VIEWS, IOU, CONF, FLOOR, GATE = ("id", "hflip"), 0.40, "max", 0.10, 0.5
GRID = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MATCH_IOU, NC, TARGET, GRADES = 0.229, 3, 0.70, "BCD"


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def detect_one(ck, device, paths, sizes):
    m = from_checkpoint_matched(str(ck), device=device, verbose=False)
    ctx = getattr(m, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    out = {}
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
        out[view] = d
    del m; torch.cuda.empty_cache()
    return out


def fuse_named(per_model, weights, sizes):
    names = list(sizes)
    out = {}
    for n in names:
        bl, sl, ll = [], [], []
        for store in per_model:
            for view in VIEWS:
                r = store[view][n]
                bl.append(np.asarray(r["boxes"], np.float32).reshape(-1, 4).tolist())
                sl.append(list(r["scores"])); ll.append(list(r["classes"]))
        W, H = sizes[n]
        if not any(len(b) for b in bl):
            out[n] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=IOU,
                                        skip_box_thr=0.0, conf_type=CONF)
        out[n] = [(int(c), float(x), tuple((np.asarray(bb) * np.array([W, H, W, H])).tolist()))
                  for bb, x, c in zip(b, s, l)]
    return out


def score(fused, targets, sound_fused, n_sound):
    best = None
    for combo in itertools.product(GRID, repeat=NC):
        tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
        for n, dets in fused.items():
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dets if s >= combo[c]]
            merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
        tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
        fn = sum(v["fn"] for v in tot.values())
        p, r, _ = metric(tp, fp, fn)
        per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
        if r >= TARGET and all(v >= TARGET for v in per) and (best is None or p > best[0]):
            fired = sum(1 for d in sound_fused.values() if any(x[1] >= combo[x[0]] for x in d))
            boxes = sum(len([x for x in d if x[1] >= combo[x[0]]]) for d in sound_fused.values())
            best = (p, r, per, list(combo), fired / n_sound, boxes / n_sound)
    return best


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    cands = sys.argv[2:] or ["/workspace/exp_cb/dlv_frz_s1", "/workspace/exp_cb/dlv_frz_s2"]
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
    mt = RG.member_boxes(device, imgs, sizes)
    ms = RG.member_boxes(device, sound, ssz)

    shipped_t = [detect_one(c, device, imgs, sizes) for c in SHIPPED.values()]
    shipped_s = [detect_one(c, device, sound, ssz) for c in SHIPPED.values()]
    W2 = [0.5, 0.5, 1.0, 1.0]

    base = score(RG.apply_gate(fuse_named(shipped_t, W2, sizes), mt, GATE), targets,
                 RG.apply_gate(fuse_named(shipped_s, W2, ssz), ms, GATE), len(ssz))
    print(f"{'配置':34s} {'P':>7} {'R':>7} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} {'框/张':>7}")
    print(f"{'现交付 (2 成员 + 门控)':34s} {base[0]:7.3f} {base[1]:7.3f} "
          f"{base[2][0]:6.3f} {base[2][1]:6.3f} {base[2][2]:6.3f} {base[4]:5.0%} {base[5]:7.2f}")

    rows = []
    for run in cands:
        cks = sorted(Path(run).glob("epoch_pth/checkpoint_epoch_*.pth"))
        if not cks:
            print(f"{run}: 无 checkpoint,跳过"); continue
        tag = Path(run).name
        best_single, best_ens = None, None
        for ck in cks:
            ep = int(ck.stem.split("_")[-1])
            dt = detect_one(ck, device, imgs, sizes)
            dsd = detect_one(ck, device, sound, ssz)
            s1 = score(RG.apply_gate(fuse_named([dt], [1.0, 1.0], sizes), mt, GATE), targets,
                       RG.apply_gate(fuse_named([dsd], [1.0, 1.0], ssz), ms, GATE), len(ssz))
            W3 = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
            s3 = score(RG.apply_gate(fuse_named(shipped_t + [dt], W3, sizes), mt, GATE), targets,
                       RG.apply_gate(fuse_named(shipped_s + [dsd], W3, ssz), ms, GATE), len(ssz))
            if s1 and (best_single is None or s1[0] > best_single[1][0]):
                best_single = (ep, s1)
            if s3 and (best_ens is None or s3[0] > best_ens[1][0]):
                best_ens = (ep, s3)
        for label, b in (("单模型", best_single), ("+入 WBF 成为第三成员", best_ens)):
            if not b:
                print(f"{tag} {label}: 无四项达标点"); continue
            ep, s = b
            print(f"{tag+' '+label+f' (ep{ep})':34s} {s[0]:7.3f} {s[1]:7.3f} "
                  f"{s[2][0]:6.3f} {s[2][1]:6.3f} {s[2][2]:6.3f} {s[4]:5.0%} {s[5]:7.2f}")
            rows.append({"run": tag, "role": label, "epoch": ep, "precision": s[0],
                         "recall": s[1], "per": s[2], "thr": s[3],
                         "fire": s[4], "bpi": s[5]})
    print(f"\n对照: 交付单模型 0.159 / 现交付集成 {base[0]:.3f}")
    print("注意: epoch 在同一测试集上选,与交付协议同为上界;两个种子分别报,"
          "方向不一致则不成立。")
    Path("/workspace/exp_cb/delivery_frz_eval.json").write_text(json.dumps(
        {"baseline": {"precision": base[0], "recall": base[1], "fire": base[4], "bpi": base[5]},
         "candidates": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
