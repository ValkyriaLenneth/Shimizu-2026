#!/usr/bin/env python3
"""The recall-versus-false-alarm curve, in the unit the client feels.

Two facts from the day meet here. Inspector burden is dominated by false alarms:
at a 5% damage prevalence, 92% of the boxes a person reviews come from sound
photographs. And the surviving false alarms are low-scoring -- median 0.207
against 0.439 for true positives -- so they sit just above the decision
thresholds and a small relaxation clears many of them.

The delivered operating point maximises precision subject to all four recalls
reaching 0.70, and 128 feasible points were enumerated under that constraint, so
nothing more is available *while the constraint holds*. What has never been put
to the client is what lies just outside it: how much false-alarm relief a small
recall concession buys, per grade.

This enumerates the same threshold grid without the four-target constraint and
reports, for each recall floor, the operating point that minimises boxes reviewed
per 100 photographs. The client knows their own damage prevalence and their own
tolerance for a missed grade-D; this turns the decision into those two facts.

Per-grade columns are kept separate because a concession on B is not the same
promise as a concession on D: B carries 47 of the 72 test boxes and D only 10.
"""
from __future__ import annotations
import csv, itertools, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/workspace/scripts_exp")
import router_gate as RG
from tta_fusion import MEMBERS, GRID, NC, DS, fuse
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

GATE, GRADES, MATCH_IOU = 0.5, "BCD", 0.229
PREV = 0.05
OUT = Path("/workspace/exp_cb/client_tradeoff")


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sound = sorted(p for p in RG.SOUND.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, ssz, targets = {}, {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    for p in sound:
        with Image.open(p) as h:
            ssz[p.name] = h.size
    keys = [(m, v) for m in MEMBERS for v in RG.VIEWS]
    w = [RG.RATIO[i] / len(RG.VIEWS) for i in range(len(MEMBERS)) for _ in RG.VIEWS]
    ft = RG.apply_gate(fuse(RG.detect(device, imgs, sizes), keys, w, sizes, RG.IOU),
                       RG.member_boxes(device, imgs, sizes), GATE)
    fs = RG.apply_gate(fuse(RG.detect(device, sound, ssz), keys, w, ssz, RG.IOU),
                       RG.member_boxes(device, sound, ssz), GATE)
    ngt = sum(len(v) for v in targets.values())
    gt_per = ngt / len(imgs)

    rows = []
    for combo in itertools.product(GRID, repeat=NC):
        tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
        for n, dets in ft.items():
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in dets if s >= combo[c]]
            merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
        tp = sum(v["tp"] for v in tot.values()); fp = sum(v["fp"] for v in tot.values())
        fn = sum(v["fn"] for v in tot.values())
        p_, r_, _ = metric(tp, fp, fn)
        per = [metric(tot[c]["tp"], tot[c]["fp"], tot[c]["fn"])[1] for c in range(NC)]
        fired = sum(1 for d in fs.values() if any(x[1] >= combo[x[0]] for x in d))
        boxes = sum(len([x for x in d if x[1] >= combo[x[0]]]) for d in fs.values())
        bpi = boxes / len(fs)
        cost = (100 * PREV * gt_per * r_ / max(p_, 1e-9) + 100 * (1 - PREV) * bpi) if p_ > 0 else 9e9
        rows.append({"thr": list(combo), "precision": p_, "recall": r_,
                     **{f"recall_{GRADES[c]}": per[c] for c in range(NC)},
                     "fire": fired / len(fs), "bpi": bpi, "cost100": cost})

    with (OUT / "curve.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    cur = [r for r in rows if r["thr"] == [0.12, 0.20, 0.12]][0]
    print(f"冻结测试集 {len(imgs)} 图 / {ngt} 框 · 健全图 {len(fs)} 张 · "
          f"损伤占比假设 {PREV:.0%}\n")
    print(f"现交付点: 阈值 {cur['thr']}  全体R {cur['recall']:.3f}  "
          f"B {cur['recall_B']:.3f} C {cur['recall_C']:.3f} D {cur['recall_D']:.3f}  "
          f"P {cur['precision']:.3f}  发火 {cur['fire']:.0%}  {cur['bpi']:.2f} 箱/张  "
          f"过目 {cur['cost100']:.0f} 箱/100 张\n")

    print("放宽「全部三级均须达到」的下限,换取误报下降:")
    print(f"{'各级R下限':>9} {'最优阈值':>18} {'全体R':>7} {'B':>6} {'C':>6} {'D':>6} "
          f"{'发火':>6} {'箱/张':>7} {'过目/100张':>10} {'较现点':>8}")
    out = []
    for floor in (0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40):
        ok = [r for r in rows if all(r[f"recall_{g}"] >= floor for g in GRADES)]
        if not ok:
            continue
        b = min(ok, key=lambda r: r["cost100"])
        print(f"{floor:9.2f} {str(b['thr']):>18} {b['recall']:7.3f} {b['recall_B']:6.3f} "
              f"{b['recall_C']:6.3f} {b['recall_D']:6.3f} {b['fire']:5.0%} {b['bpi']:7.2f} "
              f"{b['cost100']:10.0f} {b['cost100']-cur['cost100']:+8.0f}")
        out.append({"floor": floor, **b})

    print("\n只放宽 B 级(误报的 83% 是 B),C/D 仍保持 0.70:")
    print(f"{'B 下限':>7} {'最优阈值':>18} {'B':>6} {'C':>6} {'D':>6} {'发火':>6} "
          f"{'箱/张':>7} {'过目/100张':>10} {'较现点':>8}")
    out_b = []
    for floor in (0.70, 0.65, 0.60, 0.55, 0.50):
        ok = [r for r in rows if r["recall_B"] >= floor
              and r["recall_C"] >= 0.70 and r["recall_D"] >= 0.70]
        if not ok:
            continue
        b = min(ok, key=lambda r: r["cost100"])
        print(f"{floor:7.2f} {str(b['thr']):>18} {b['recall_B']:6.3f} {b['recall_C']:6.3f} "
              f"{b['recall_D']:6.3f} {b['fire']:5.0%} {b['bpi']:7.2f} "
              f"{b['cost100']:10.0f} {b['cost100']-cur['cost100']:+8.0f}")
        out_b.append({"b_floor": floor, **b})
    (OUT / "tradeoff.json").write_text(json.dumps(
        {"current": cur, "all_grades": out, "b_only": out_b}, indent=2), encoding="utf-8")
    print(f"\n完整网格写入 {OUT/'curve.csv'}")


if __name__ == "__main__":
    main()
