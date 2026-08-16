#!/usr/bin/env python3
"""What are the false positives actually made of?

Precision on the frozen split is 0.383, so roughly three of every five boxes the
delivered configuration draws are counted wrong. "Wrong" covers two very
different failures, and the delivery protocol scores them identically:

  wrong place   a box on something that is not damage at all
  wrong grade   a box correctly on real damage, assigned the wrong severity

The second is bounded by the label set rather than by the model. The baseline
configuration records eleven grade contradictions in the annotations, six of them
柱脚 B-versus-C pairs, so a B/C confusion in the results may be the corpus
disagreeing with itself. If most false positives turn out to be right-place
wrong-grade, then detection is not the ceiling and further detection work cannot
move the number; if most are wrong-place, the reverse.

Nothing here changes the delivered figure. It says which kind of work could.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
from tta_fusion import MEMBERS, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import read_targets

VIEWS, RATIO, IOU, CONF, FLOOR = ("id", "hflip"), (1.0, 2.0), 0.40, "max", 0.10
THR = {0: 0.12, 1: 0.20, 2: 0.12}
MATCH_IOU, GRADES = 0.229, "BCD"


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def boxiou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    store = {}
    for tag, ck in MEMBERS.items():
        m = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in VIEWS:
            d = {}
            for p in imgs:
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
    keys = [(m, v) for m in MEMBERS for v in VIEWS]
    w = [RATIO[i] / len(VIEWS) for i in range(len(MEMBERS)) for _ in VIEWS]
    fused = fuse(store, keys, w, sizes, IOU)

    kinds = Counter(); pairs = Counter(); tp = 0
    for name, dets in fused.items():
        gts = [(t.cls, t.xyxy) for t in targets[name]]
        used = set()
        for c, s, b in sorted(dets, key=lambda d: -d[1]):
            if s < THR[c]:
                continue
            same = [i for i, (gc, gb) in enumerate(gts)
                    if gc == c and i not in used and boxiou(b, gb) >= MATCH_IOU]
            if same:
                used.add(same[0]); tp += 1; continue
            other = [(i, gc) for i, (gc, gb) in enumerate(gts)
                     if gc != c and boxiou(b, gb) >= MATCH_IOU]
            if other:
                kinds["位置对、等级错"] += 1
                pairs[f"预测{GRADES[c]} 实为{GRADES[other[0][1]]}"] += 1
            else:
                dup = [i for i, (gc, gb) in enumerate(gts)
                       if gc == c and i in used and boxiou(b, gb) >= MATCH_IOU]
                kinds["同一处重复框" if dup else "位置错(无对应真值)"] += 1
    fp = sum(kinds.values())
    print(f"交付配置在冻结集上: 命中 {tp} 个,误报 {fp} 个,precision {tp/(tp+fp):.3f}\n")
    print("误报构成:")
    for k, v in kinds.most_common():
        print(f"  {k:22s} {v:4d}  {v/fp:5.1%}")
    if pairs:
        print("\n等级混淆方向:")
        for k, v in pairs.most_common():
            print(f"  {k:20s} {v:4d}")
    grade_err = kinds["位置对、等级错"]
    print(f"\n若等级判对即算命中,precision 将为 {(tp+grade_err)/(tp+fp):.3f} "
          f"(现 {tp/(tp+fp):.3f},差 {grade_err/(tp+fp):+.3f})")
    Path("/workspace/exp_cb/fp_breakdown.json").write_text(json.dumps(
        {"tp": tp, "fp": fp, "kinds": dict(kinds), "pairs": dict(pairs)},
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
