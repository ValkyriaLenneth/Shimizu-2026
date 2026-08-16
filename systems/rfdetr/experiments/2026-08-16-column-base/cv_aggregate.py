#!/usr/bin/env python3
"""Pool the folds and measure what the noise floor actually is.

Each fold wrote raw detections rather than a score, because averaging five
per-fold precisions would preserve exactly the small-sample noise the folds were
built to remove. Pooling first and searching thresholds once over the union is
the whole point: 320 boxes instead of 72, and 39 grade-D boxes instead of 10.

What this reports, in order of what it settles:

  noise floor   the same arm run under two seeds. Whatever gap appears between
                them is measurement noise, since nothing else differs. This is
                the number that decides whether any training-side claim can be
                made on this project at all -- if two identical arms differ by
                more than the effects being chased, they cannot.
  comparison    the two arms per epoch, so a difference can be read at a fixed
                epoch rather than at each arm's own best, which would compare two
                maxima and inflate the gap the way five retracted claims did.
  vs per-fold   the identical detections scored fold by fold instead of pooled.
                Each fold holds about 64 boxes, close to the frozen split's 72,
                so the five per-fold seed gaps sample the small-sample noise the
                project has been living with, and the pooled gap is the same
                quantity measured once on 320. Comparing the two needs no
                historical figure and no second evaluation set -- it is the same
                predictions summed differently, which is the cleanest form the
                question can take.

Nothing here is a delivery number: these folds mix the frozen test split into
training. Absolute values are inflated for every arm equally; only differences
between arms are meaningful.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)
from PIL import Image

EXP = Path("/workspace/exp_cb")
CV = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_cv5")
FROZEN = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
GRID = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MATCH_IOU, NC, GRADES, TARGET = 0.229, 3, "BCD", 0.70


def fold_targets(f: int):
    d = CV / f"fold{f}" / "test"
    out = {}
    for p in sorted((d / "images").iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(p) as h:
            W, H = h.size
        out[p.name] = read_targets(d / "labels" / f"{p.stem}.txt", W, H)
    return out


def tabulate(dets, targets, names):
    """(image, class, threshold) counts, so pooling is addition."""
    G = len(GRID)
    tp = np.zeros((len(names), NC, G), np.int32); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for i, n in enumerate(names):
        rec = dets.get(n, {"boxes": [], "scores": [], "classes": []})
        trio = list(zip(rec["classes"], rec["scores"], rec["boxes"]))
        for gi, t in enumerate(GRID):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            sel = [Prediction(cls=int(c), conf=float(s), xyxy=tuple(b))
                   for c, s, b in trio if s >= t]
            merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
            for c in range(NC):
                tp[i, c, gi] = tot[c]["tp"]; fp[i, c, gi] = tot[c]["fp"]; fn[i, c, gi] = tot[c]["fn"]
    return tp, fp, fn


def best(TP, FP, FN):
    win = None
    for combo in itertools.product(range(len(GRID)), repeat=NC):
        t = np.array([TP[c, combo[c]] for c in range(NC)])
        f = np.array([FP[c, combo[c]] for c in range(NC)])
        m = np.array([FN[c, combo[c]] for c in range(NC)])
        den = t + m
        per = np.where(den > 0, t / np.maximum(den, 1), 0.0)
        if not np.all((den == 0) | (per >= TARGET)):
            continue
        st, sf, sm = t.sum(), f.sum(), m.sum()
        if st / max(st + sm, 1) < TARGET:
            continue
        p = st / max(st + sf, 1)
        if win is None or p > win[0]:
            win = (p, combo, st / max(st + sm, 1), per)
    return win


def arm_tables(arm: str):
    """Per-epoch tables, pooled across folds and kept separately per fold."""
    pooled, per_fold = {}, {}
    for f in range(5):
        d = EXP / f"cv_{arm}_f{f}" / "cv_dets.json"
        if not d.exists():
            return None, None
        dets = json.loads(d.read_text())
        tg = fold_targets(f)
        names = sorted(tg)
        for ep, per_img in dets.items():
            tab = tabulate(per_img, tg, names)
            summed = [t.sum(0) for t in tab]
            acc = pooled.setdefault(int(ep), [np.zeros((NC, len(GRID)), np.int64) for _ in range(3)])
            for k in range(3):
                acc[k] += summed[k]
            per_fold.setdefault(f, {})[int(ep)] = summed
    return pooled, per_fold


def main():
    arms = sys.argv[1:] or ["ctrlA", "ctrlB"]
    pooled, per_fold = {}, {}
    for a in arms:
        p, pf = arm_tables(a)
        if p is None:
            print(f"{a}: 尚未完成五折,跳过")
            continue
        pooled[a], per_fold[a] = p, pf
        print(f"{a}: {len(p)} epoch 已汇总")
    if len(pooled) < 2:
        return
    a, b = arms[0], arms[1]

    nbox = sum(sum(len(v) for v in fold_targets(f).values()) for f in range(5))
    print(f"\n汇总评测 {nbox} 框 —— 逐 epoch 四项达标最高 precision:")
    print(f"{'epoch':>6} {a:>10} {b:>10} {'差':>9}")
    pooled_d = []
    for ep in sorted(pooled[a]):
        x, y = best(*pooled[a][ep]), best(*pooled[b].get(ep, pooled[a][ep]))
        if x and y:
            pooled_d.append(x[0] - y[0])
        g = lambda w: f"{w[0]:.3f}" if w else "--"
        d = f"{x[0]-y[0]:+.3f}" if x and y else "--"
        print(f"{ep:6d} {g(x):>10} {g(y):>10} {d:>9}")

    # Same detections, scored inside each fold instead of pooled. Each fold holds
    # roughly the frozen split's box count, so these gaps are what the project's
    # existing protocol would have reported.
    fold_d = []
    for f in range(5):
        for ep in sorted(per_fold[a][f]):
            x, y = best(*per_fold[a][f][ep]), best(*per_fold[b][f][ep])
            if x and y:
                fold_d.append(x[0] - y[0])

    def stat(d, label, n):
        d = np.array(d)
        print(f"  {label:22s} n={len(d):3d}  |差| 中位 {np.median(np.abs(d)):.3f}  "
              f"p90 {np.percentile(np.abs(d),90):.3f}  最大 {np.abs(d).max():.3f}   ({n} 框/次)")
        return float(np.median(np.abs(d)))

    print(f"\n噪声底(同一配置,仅换种子 —— 差应为 0):")
    mp = stat(pooled_d, "汇总 (5 折合并)", nbox)
    mf = stat(fold_d, "折内单独 (现行协议)", nbox // 5)
    ratio = mp / max(mf, 1e-9)
    print(f"\n  汇总把噪声降到折内的 {ratio:.0%}")
    print(f"  {'-> 训练类实验现在可测' if ratio < 0.7 else '-> 噪声未显著下降,训练类改进在本项目数据下仍不可测'}")

    # The statistic the project actually reports is the maximum over epochs, not
    # a per-epoch value, so it gets its own line: a maximum has a longer upper
    # tail than the values it is drawn from and behaves differently under pooling.
    def best_over_epochs(tabs):
        v = [best(*t) for t in tabs.values()]
        v = [w[0] for w in v if w]
        return max(v) if v else None
    ma, mb = best_over_epochs(pooled[a]), best_over_epochs(pooled[b])
    if ma and mb:
        print(f"\n  最佳 epoch 统计量(项目现行报法): {a} {ma:.3f} / {b} {mb:.3f}  差 {ma-mb:+.3f}")

    Path("/workspace/exp_cb/cv_noise.json").write_text(json.dumps(
        {"pooled_abs_median": mp, "perfold_abs_median": mf, "ratio": ratio,
         "n_pooled_epochs": len(pooled_d), "n_fold_epochs": len(fold_d),
         "boxes_pooled": nbox,
         "best_epoch": {a: ma, b: mb}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
