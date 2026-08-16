#!/usr/bin/env python3
"""False-alarm rate at the operating point the damaged side actually selects.

A false-alarm rate is meaningless without the thresholds it was measured at, and
the thresholds are not free parameters here -- they are whatever the damaged-side
aggregation picks for that arm and epoch under the four-target constraint. So the
two sides are joined: pool the damaged folds, find the threshold triple, then
apply that same triple to the pooled held-out sound detections.

Each fold contributes only the sound photographs it did not train on, so the
five fifths together score all 29 exactly once. The control arm trains on no
sound imagery at all, but is scored on the same per-fold fifths, because the arms
are compared fold by fold and a different evaluation set would break the pairing.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts_exp")
from cv_aggregate import EXP, GRID, NC, best, fold_targets, tabulate


def arm(name: str):
    pooled, sound = {}, {}
    for f in range(5):
        dd = EXP / f"cv_{name}_f{f}" / "cv_dets.json"
        sd = EXP / f"cv_{name}_f{f}" / "cv_sound.json"
        if not dd.exists() or not sd.exists():
            return None, None
        dets = json.loads(dd.read_text())
        snd = json.loads(sd.read_text())
        tg = fold_targets(f)
        names = sorted(tg)
        for ep, per_img in dets.items():
            acc = pooled.setdefault(int(ep), [np.zeros((NC, len(GRID)), np.int64) for _ in range(3)])
            for k, t in enumerate(tabulate(per_img, tg, names)):
                acc[k] += t.sum(0)
        for ep, per_img in snd.items():
            sound.setdefault(int(ep), []).extend(
                [(int(c), float(s)) for r in per_img.values()
                 for c, s in zip(r["classes"], r["scores"])] and
                [[(int(c), float(s)) for c, s in zip(r["classes"], r["scores"])]
                 for r in per_img.values()] or [])
    return pooled, sound


def fa(sound_epoch, thr):
    imgs = sound_epoch
    fired = sum(1 for d in imgs if any(s >= thr[c] for c, s in d))
    boxes = sum(sum(1 for c, s in d if s >= thr[c]) for d in imgs)
    return fired / max(len(imgs), 1), boxes / max(len(imgs), 1), len(imgs)


def main():
    arms = sys.argv[1:] or ["ctrlA", "ctrlB"]
    out = {}
    for a in arms:
        pooled, sound = arm(a)
        if pooled is None:
            print(f"{a}: 数据不全,跳过")
            continue
        rows = []
        for ep in sorted(pooled):
            w = best(*pooled[ep])
            if not w or ep not in sound:
                continue
            thr = [GRID[i] for i in w[1]]
            r, b, n = fa(sound[ep], thr)
            rows.append({"epoch": ep, "precision": w[0], "recall": w[2],
                         "thr": thr, "fire": r, "bpi": b, "n_sound": n})
        out[a] = rows
        print(f"\n{a} —— 阈值取该 epoch 汇总损伤侧的四项达标最优点:")
        print(f"{'epoch':>6} {'P':>7} {'R':>7} {'阈值':>18} {'健全发火':>9} {'框/张':>7}")
        for r in rows:
            print(f"{r['epoch']:6d} {r['precision']:7.3f} {r['recall']:7.3f} "
                  f"{str(r['thr']):>18} {r['fire']:8.0%} {r['bpi']:7.2f}")
        if rows:
            print(f"  跨 epoch 中位: 发火 {np.median([r['fire'] for r in rows]):.0%}, "
                  f"{np.median([r['bpi'] for r in rows]):.2f} 框/张 "
                  f"({rows[0]['n_sound']} 张留出健全图)")
    if len(out) >= 2:
        a, b = arms[0], arms[1]
        common = sorted({r["epoch"] for r in out[a]} & {r["epoch"] for r in out[b]})
        da = {r["epoch"]: r for r in out[a]}
        db = {r["epoch"]: r for r in out[b]}
        d = np.array([da[e]["bpi"] - db[e]["bpi"] for e in common])
        print(f"\n{a} - {b} 每张框数配对差: 均值 {d.mean():+.3f}  "
              f"95% [{d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d)):+.3f}, "
              f"{d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)):+.3f}]  n={len(d)}")
    Path("/workspace/exp_cb/cv_falsealarm.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
