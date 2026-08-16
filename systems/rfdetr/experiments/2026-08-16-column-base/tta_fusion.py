#!/usr/bin/env python3
"""Buy fusion diversity without training anything.

Two facts from this week point at the same experiment. First, WBF pays for
disagreement rather than strength -- measured directly: the two shipped members
agree on only 55.3% of their boxes, and that is why fusing them nearly doubles
the better one's precision, while swapping in a stronger but more agreeable
member never helped across 315 subsets. Second, every attempt to add a *third*
member by training one collapsed under seed variation: 0.389 became 0.290 when
the same recipe was retrained, and the shipped-checkpoint pool is exhausted at
two.

So the remaining way to add a disagreeing member is to stop asking for a new
model and take a new *view*. A horizontal flip or a rescale runs the same weights
over a different input and produces boxes that differ where the model is
uncertain -- which is exactly the disagreement WBF converts into recall. Nothing
is trained, so seed variance cannot touch the result, and it can be checked with
the paired bootstrap that settled the wbf_iou change.

Views are weighted to preserve the 1:2 balance between ep016 and cp075 that the
current deliverable depends on; each member's views split that member's weight
rather than adding to it, so a member does not gain influence merely by being
viewed more ways.

Every configuration is scored under the delivery protocol -- match_iou 0.229,
four-target feasibility, precision maximised subject to it -- and the winner is
then paired-bootstrapped against the current 0.329 at fixed thresholds.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import (
    Prediction, match_counts, merge_counts, metric, read_targets)

DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
SOUND = Path("/workspace/sound_20260807/column_base")
OUT = Path("/workspace/exp_cb/e20_tta")
MEMBERS = {"ep016": "/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth",
           "cp075": "/workspace/handoff_20260804/checkpoints/column_base_copypaste_epoch_075.pth"}
BASE_W = {"ep016": 1.0, "cp075": 2.0}
VIEWS = ["id", "hflip", "s085", "s115"]
VIEW_SETS = [("id",), ("id", "hflip"), ("id", "s085", "s115"), ("id", "hflip", "s085", "s115")]
WBF_IOU = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
GRID = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
FLOOR, MATCH_IOU, NC, GRADES, TARGET = 0.10, 0.229, 3, "BCD", 0.70
CONF = "max"
CUR = 0.329


def render(im: Image.Image, view: str) -> Image.Image:
    if view == "id":
        return im
    if view == "hflip":
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    s = float(view[1:]) / 100.0
    return im.resize((max(32, int(im.width * s)), max(32, int(im.height * s))), Image.BICUBIC)


def unrender(boxes_n: np.ndarray, view: str) -> np.ndarray:
    """Map normalised boxes from a view back onto the original frame.

    Scaling is a no-op in normalised coordinates -- the box occupies the same
    fraction of a smaller image -- so only the flip needs undoing.
    """
    if view != "hflip" or not len(boxes_n):
        return boxes_n
    out = boxes_n.copy()
    out[:, 0], out[:, 2] = 1.0 - boxes_n[:, 2], 1.0 - boxes_n[:, 0]
    return out


def predict_views(device: str, paths, sizes) -> dict:
    store = {}
    for tag, ck in MEMBERS.items():
        model = from_checkpoint_matched(ck, device=device, verbose=False)
        ctx = getattr(model, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in VIEWS:
            d = {}
            for p in paths:
                with Image.open(p) as h:
                    im = render(h.convert("RGB"), view)
                W, H = im.size
                det = model.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1)
                keep = cls < NC
                xy = np.asarray(det.xyxy).reshape(-1, 4)[keep]
                bn = np.clip(xy / np.array([W, H, W, H], np.float32), 0, 1)
                d[p.name] = {"boxes": unrender(bn, view).tolist(),
                             "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist()}
            store[(tag, view)] = d
        del model
        torch.cuda.empty_cache()
    return store


def fuse(store, keys, weights, sizes, iou):
    out = {}
    for name in sizes:
        bl, sl, ll = [], [], []
        for k in keys:
            r = store[k][name]
            bl.append(np.asarray(r["boxes"], np.float32).reshape(-1, 4).tolist())
            sl.append(list(r["scores"])); ll.append(list(r["classes"]))
        W, H = sizes[name]
        if not any(len(b) for b in bl):
            out[name] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=list(weights),
                                        iou_thr=iou, skip_box_thr=0.0, conf_type=CONF)
        out[name] = [(int(c), float(x), tuple((np.asarray(bb) * np.array([W, H, W, H])).tolist()))
                     for bb, x, c in zip(b, s, l)]
    return out


def tabulate(fused, targets, names):
    G = len(GRID)
    tp = np.zeros((len(names), NC, G), np.int32); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for i, n in enumerate(names):
        for gi, t in enumerate(GRID):
            tot = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(NC)}
            sel = [Prediction(cls=c, conf=s, xyxy=b) for c, s, b in fused[n] if s >= t]
            merge_counts(tot, match_counts(targets[n], sel, MATCH_IOU, NC))
            for c in range(NC):
                tp[i, c, gi] = tot[c]["tp"]; fp[i, c, gi] = tot[c]["fp"]; fn[i, c, gi] = tot[c]["fn"]
    return tp, fp, fn


def best(tab, idx):
    tp, fp, fn = tab
    TP, FP, FN = tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0)
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


def at(tab, idx, combo):
    tp, fp, fn = tab
    t = np.array([tp[idx, c, combo[c]].sum() for c in range(NC)])
    f = np.array([fp[idx, c, combo[c]].sum() for c in range(NC)])
    st, sf = t.sum(), f.sum()
    return st / max(st + sf, 1)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, targets = {}, {}
    for p in imgs:
        with Image.open(p) as h:
            sizes[p.name] = h.size
        targets[p.name] = read_targets(DS / "test" / "labels" / f"{p.stem}.txt", *sizes[p.name])
    cache = OUT / "views.pt"
    if cache.exists():
        store = {tuple(k.split("|")): v for k, v in torch.load(cache, weights_only=False).items()}
        print(f"复用缓存的 {len(store)} 组预测", flush=True)
    else:
        print(f"两个已交付权重 x {len(VIEWS)} 个视图,共 {2*len(VIEWS)} 组预测", flush=True)
        store = predict_views(device, imgs, sizes)
        torch.save({"|".join(k): v for k, v in store.items()}, cache)
    names = [p.name for p in imgs]

    rows, tabs = [], {}
    for vs in VIEW_SETS:
        keys = [(m, v) for m in MEMBERS for v in vs]
        w = [BASE_W[m] / len(vs) for m in MEMBERS for _ in vs]
        for iou in WBF_IOU:
            tab = tabulate(fuse(store, keys, w, sizes, iou), targets, names)
            win = best(tab, np.arange(len(names)))
            tag = f"{'+'.join(vs)}@{iou}"
            tabs[tag] = (tab, win)
            rows.append({"views": "+".join(vs), "wbf_iou": iou,
                         "precision": None if not win else round(win[0], 4),
                         "recall": None if not win else round(win[2], 4),
                         "thresholds": None if not win else ",".join(str(GRID[i]) for i in win[1])})
            print(f"  {tag:28s} P = {'--' if not win else f'{win[0]:.3f}'}", flush=True)

    ok = [r for r in rows if r["precision"]]
    ok.sort(key=lambda r: -r["precision"])
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n当前交付 = {CUR:.3f} (id@0.20,不在本次搜索范围内)")
    if not ok or ok[0]["precision"] <= CUR + 1e-9:
        print(f"最好 {ok[0]['precision']:.3f} ({ok[0]['views']}@{ok[0]['wbf_iou']}) 未超过交付配置 -> TTA 无效,不采纳")
        return
    top = ok[0]
    tag = f"{top['views']}@{top['wbf_iou']}"
    print(f"最好 {top['precision']:.3f} ({tag}) 超过交付配置,进入配对自助检验")

    base_tab = tabulate(fuse(store, [(m, "id") for m in MEMBERS],
                             [BASE_W[m] for m in MEMBERS], sizes, 0.20), targets, names)
    base_win = best(base_tab, np.arange(len(names)))
    cand_tab, cand_win = tabs[tag]
    rng = np.random.default_rng(20260816)
    d = []
    for _ in range(2000):
        idx = rng.integers(0, len(names), len(names))
        d.append(at(cand_tab, idx, cand_win[1]) - at(base_tab, idx, base_win[1]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n固定阈值配对自助 2000 次 vs 交付配置:")
    print(f"  配对差均值 {d.mean():+.4f}  95% 区间 [{lo:+.4f}, {hi:+.4f}]  更好 {np.mean(d>0):.1%}")
    print(f"  结论: {'站得住,可采纳' if lo > 0 else '落在噪声内,不采纳'}")
    (OUT / "paired.json").write_text(json.dumps(
        {"candidate": tag, "precision": top["precision"], "mean_diff": float(d.mean()),
         "ci95": [float(lo), float(hi)], "p_better": float(np.mean(d > 0)),
         "adopt": bool(lo > 0)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
