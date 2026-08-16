#!/usr/bin/env python3
"""Judge the arms on the criterion the delivery actually turns on.

Horizontal-flip TTA was adopted this morning not because it raised precision --
that gain did not survive holdout validation -- but because it nearly doubled how
often the four-target constraint admits any solution at all, 26.6% to 46.4%, with
zero reversals across a thousand resamples. Feasibility, not precision, is what
the delivery claim rests on: "all four grades at 0.70" holds on only 46% of
resamples of the frozen split, and that is its weakest point.

The three training arms were scored on precision and false alarms and neither was
this. An arm that lifts grade-D recall would widen the feasible region even while
costing precision, and would then be worth adopting on the same grounds the flip
was. Grade D carries 30 training boxes against B's 165 and decides feasibility
more than anything else, so this is not a remote possibility.

Measured the same way as the flip: bootstrap the pooled folds, ask whether any
threshold triple meets all four targets, and compare arms on the same resamples
so the discordant cases carry the evidence.
"""
from __future__ import annotations
import itertools, json, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts_exp")
from cv_aggregate import EXP, GRID, NC, TARGET, fold_targets, tabulate

NBOOT = 400
GRADES = "BCD"


def per_image_tables(arm: str):
    """Per-image tables kept unpooled, so resampling images is possible."""
    rows, per_ep = [], {}
    for f in range(5):
        d = EXP / f"cv_{arm}_f{f}" / "cv_dets.json"
        if not d.exists():
            return None
        dets = json.loads(d.read_text())
        tg = fold_targets(f)
        names = sorted(tg)
        for ep, per_img in dets.items():
            tp, fp, fn = tabulate(per_img, tg, names)
            acc = per_ep.setdefault(int(ep), [[], [], []])
            acc[0].append(tp); acc[1].append(fp); acc[2].append(fn)
    return {ep: [np.concatenate(v[k], 0) for k in range(3)] for ep, v in per_ep.items()}


def feasible(tabs, idx):
    tp, fp, fn = tabs
    TP, FN = tp[idx].sum(0), fn[idx].sum(0)
    for combo in itertools.product(range(len(GRID)), repeat=NC):
        t = np.array([TP[c, combo[c]] for c in range(NC)])
        m = np.array([FN[c, combo[c]] for c in range(NC)])
        den = t + m
        per = np.where(den > 0, t / np.maximum(den, 1), 0.0)
        if not np.all((den == 0) | (per >= TARGET)):
            continue
        if t.sum() / max(t.sum() + m.sum(), 1) >= TARGET:
            return True
    return False


def main():
    arms = sys.argv[1:] or ["negA", "ctrlA"]
    tabs = {a: per_image_tables(a) for a in arms}
    if any(v is None for v in tabs.values()):
        print("数据不全")
        return
    a, b = arms[0], arms[1]
    eps = sorted(set(tabs[a]) & set(tabs[b]))
    n = tabs[a][eps[0]][0].shape[0]
    print(f"{a} vs {b} —— {n} 图 / {len(eps)} epoch,每 epoch {NBOOT} 次自助\n")
    print(f"{'epoch':>5} {a+' 可行率':>12} {b+' 可行率':>12} {'仅前者':>7} {'仅后者':>7}")
    tot_a = tot_b = 0
    for ep in eps:
        rng = np.random.default_rng(20260816)
        both = oa = ob = 0
        for _ in range(NBOOT):
            idx = rng.integers(0, n, n)
            fa = feasible(tabs[a][ep], idx)
            fb = feasible(tabs[b][ep], idx)
            both += fa and fb; oa += fa and not fb; ob += fb and not fa
        tot_a += oa; tot_b += ob
        print(f"{ep:5d} {(both+oa)/NBOOT:11.1%} {(both+ob)/NBOOT:11.1%} {oa:7d} {ob:7d}")
    disc = tot_a + tot_b
    p = (sum(comb(disc, i) for i in range(tot_a, disc + 1)) / 2 ** disc) if disc else 1.0
    print(f"\n跨全部 epoch 的不一致 {disc} 次: 仅 {a} 可行 {tot_a}, 仅 {b} 可行 {tot_b}")
    print(f"精确二项 p = {p:.2e}  -> "
          f"{a+' 可行区域更大' if tot_a > tot_b and p < 0.05 else (b+' 可行区域更大' if p < 0.05 else '无显著差异')}")
    Path(f"/workspace/exp_cb/cv_feas_{a}_vs_{b}.json").write_text(json.dumps(
        {"only_a": tot_a, "only_b": tot_b, "p": float(p)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
