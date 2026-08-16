#!/usr/bin/env python3
"""What the delivered configuration does on the client's real incoming photographs.

Every number in the freeze document comes from 45 images. The thresholds
B 0.12 / C 0.20 / D 0.12 were tuned on them, and the previous rounds established
that even the four-target claim holds on only 46% of resamples of that split.
The client also supplied 1 159 unlabelled photographs, which have been sitting
unused because without ground truth they cannot measure recall.

They can answer two questions that matter for the delivery decision and need no
labels at all.

First, transfer. A threshold is a cut through a score distribution. If the scores
the model produces on the client's 1 159 photographs are distributed differently
from the 45 it was tuned on, the cut lands somewhere else and the tuned operating
point does not carry over -- a deployment risk entirely separate from the recall
estimate, and invisible to every experiment run this week. Comparing the two
distributions directly is the check.

Second, which photographs to annotate. The binding constraint on this project is
that C has 15 boxes and D has 10, too few for any real improvement to be
measurable. That request has been made twice as a sentence; it is far more
useful as a list. Detections that sit near the decision threshold in the two
scarce grades are the ones whose true labels would most change the recall
estimate, so the pool is ranked by that and written out as an annotation queue.

Runs the frozen delivery configuration exactly as shipped -- both members, both
views, the same fusion settings and thresholds.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched
from evaluate_rfdetr_threshold_sweep import read_targets

POOL = Path("/workspace/unlabeled_pool")
DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
FRZ = Path("/workspace/handoff_20260816_column_base_freeze")
MEMBERS = {"ep016": FRZ / "checkpoints/column_base_negatives_v1_epoch_016.pth",
           "cp075": FRZ / "checkpoints/column_base_copypaste_epoch_075.pth"}
RATIO, VIEWS, IOU, CONF, FLOOR = (1.0, 2.0), ("id", "hflip"), 0.40, "max", 0.10
THR = {0: 0.12, 1: 0.20, 2: 0.12}
GRADES = "BCD"
SCARCE = (1, 2)          # C and D: 15 and 10 boxes in the whole test split
OUT = Path("/workspace/exp_cb/e24_deploy")


def unflip(b, view):
    if view != "hflip" or not len(b):
        return b
    o = b.copy(); o[:, 0], o[:, 2] = 1.0 - b[:, 2], 1.0 - b[:, 0]
    return o


def run(device, paths):
    per = {}
    for tag, ck in MEMBERS.items():
        m = from_checkpoint_matched(str(ck), device=device, verbose=False)
        ctx = getattr(m, "model", None)
        if ctx is not None and hasattr(ctx, "device"):
            ctx.device = torch.device(device)
        for view in VIEWS:
            d = {}
            for i, p in enumerate(paths):
                try:
                    with Image.open(p) as h:
                        im = h.convert("RGB")
                except Exception:
                    continue
                if view == "hflip":
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                W, H = im.size
                det = m.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1); keep = cls < 3
                bn = np.clip(np.asarray(det.xyxy).reshape(-1, 4)[keep]
                             / np.array([W, H, W, H], np.float32), 0, 1)
                d[p.name] = {"boxes": unflip(bn, view).tolist(),
                             "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist(), "size": [W, H]}
                if i % 300 == 0:
                    print(f"    {tag}/{view} {i}/{len(paths)}", flush=True)
            per[(tag, view)] = d
        del m; torch.cuda.empty_cache()
    return per


def fuse(per, names):
    keys = [(m, v) for m in MEMBERS for v in VIEWS]
    w = [RATIO[i] / len(VIEWS) for i in range(len(MEMBERS)) for _ in VIEWS]
    out = {}
    for name in names:
        if not all(name in per[k] for k in keys):
            continue
        bl = [np.asarray(per[k][name]["boxes"], np.float32).reshape(-1, 4).tolist() for k in keys]
        sl = [list(per[k][name]["scores"]) for k in keys]
        ll = [list(per[k][name]["classes"]) for k in keys]
        if not any(len(b) for b in bl):
            out[name] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=w, iou_thr=IOU,
                                        skip_box_thr=0.0, conf_type=CONF)
        out[name] = [(int(c), float(x)) for x, c in zip(s, l)]
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    pool = sorted(p for p in POOL.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    test = sorted(p for p in (DS / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"客户无标注池 {len(pool)} 张 · 冻结测试集 {len(test)} 张", flush=True)

    print("  测试集推理...", flush=True)
    ftest = fuse(run(device, test), [p.name for p in test])
    print("  无标注池推理...", flush=True)
    fpool = fuse(run(device, pool), [p.name for p in pool])

    def profile(f, label):
        fired = sum(1 for d in f.values() if any(s >= THR[c] for c, s in d))
        kept = [(c, s) for d in f.values() for c, s in d if s >= THR[c]]
        cnt = {c: sum(1 for k, _ in kept if k == c) for c in range(3)}
        print(f"\n{label} ({len(f)} 张):")
        print(f"  发火 {fired}/{len(f)} = {fired/max(len(f),1):.0%}  "
              f"{len(kept)/max(len(f),1):.2f} 框/张")
        print(f"  各级框数 " + "  ".join(f"{GRADES[c]} {cnt[c]}" for c in range(3)) +
              f"  (占比 " + " / ".join(f"{cnt[c]/max(len(kept),1):.0%}" for c in range(3)) + ")")
        return fired / max(len(f), 1), len(kept) / max(len(f), 1), cnt, kept

    rt = profile(ftest, "冻结测试集(阈值调优所在)")
    rp = profile(fpool, "客户无标注池")

    # Transfer check: the tuned cut only carries over if the score distribution
    # it cuts through has the same shape on the incoming data.
    print("\n分数分布对比(超过各级阈值的框):")
    print(f"{'级':>3} {'测试集中位':>10} {'池中位':>9} {'测试集 p90':>11} {'池 p90':>9} {'KS':>7}")
    ks_out = {}
    for c in range(3):
        a = np.array([s for k, s in rt[3] if k == c])
        b = np.array([s for k, s in rp[3] if k == c])
        if len(a) < 3 or len(b) < 3:
            print(f"{GRADES[c]:>3} {'样本太少':>10}")
            continue
        grid = np.linspace(0, 1, 501)
        ks = np.max(np.abs(np.searchsorted(np.sort(a), grid) / len(a)
                           - np.searchsorted(np.sort(b), grid) / len(b)))
        print(f"{GRADES[c]:>3} {np.median(a):10.3f} {np.median(b):9.3f} "
              f"{np.percentile(a,90):11.3f} {np.percentile(b,90):9.3f} {ks:7.3f}")
        ks_out[GRADES[c]] = {"test_median": float(np.median(a)), "pool_median": float(np.median(b)),
                             "ks": float(ks), "n_test": len(a), "n_pool": len(b)}

    # Annotation queue: rank by how much a label would move the scarce-grade
    # recall estimate. A detection just under threshold is the ambiguous case;
    # one just over is what the delivered system would report and is worth
    # confirming. Both sit inside a band around the cut.
    rows = []
    for name, dets in fpool.items():
        score = 0.0
        detail = []
        for c, s in dets:
            band = abs(s - THR[c])
            if c in SCARCE and band <= 0.10:
                score += (0.10 - band) * (2.0 if c == 2 else 1.5)   # D scarcer than C
                detail.append(f"{GRADES[c]}:{s:.2f}")
        if score > 0:
            rows.append({"image": name, "priority": round(score, 4),
                         "near_threshold": ";".join(detail[:6]), "n_boxes": len(dets)})
    rows.sort(key=lambda r: -r["priority"])
    with (OUT / "annotation_queue.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "priority", "near_threshold", "n_boxes"])
        w.writeheader(); w.writerows(rows)
    print(f"\n标注优先队列: {len(rows)} 张图在 C/D 阈值 +-0.10 带内,已按信息量排序")
    print(f"  写入 {OUT/'annotation_queue.csv'}")
    print(f"  前 100 张覆盖 {sum(r['n_boxes'] for r in rows[:100])} 个候选框")
    (OUT / "profile.json").write_text(json.dumps(
        {"pool_n": len(fpool), "test_n": len(ftest),
         "test_fire": rt[0], "pool_fire": rp[0],
         "test_bpi": rt[1], "pool_bpi": rp[1],
         "test_counts": {GRADES[c]: rt[2][c] for c in range(3)},
         "pool_counts": {GRADES[c]: rp[2][c] for c in range(3)},
         "score_dist": ks_out, "queue_len": len(rows)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
