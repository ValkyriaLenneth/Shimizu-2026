#!/usr/bin/env python3
"""Score false alarms on sound photographs each fold never trained on.

The negatives arm exists to quiet the model on undamaged column bases -- the
failure a site inspector actually experiences, and the one the delivered
configuration has never improved: 93% of sound photographs fire at 1.86 boxes
each. Measuring that honestly requires sound imagery the model did not see, so
the 29 available photographs are split the same five ways as the damaged ones,
and each fold is scored only on its own held-out fifth.

The control arm trains on no sound imagery at all, so every photograph is
held out for it. It is nonetheless scored on the same per-fold fifths rather than
on all 29, because the two arms are compared fold by fold and giving one arm a
different evaluation set would break that pairing.

Detections are dumped rather than scored here. The thresholds that matter are
whichever the damaged-side aggregation selects, and those are not known until the
folds are pooled, so the decision is deferred to the aggregation step and this
writes raw scores.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

EXP = Path("/workspace/exp_cb")
SOUND = Path("/workspace/sound_20260807/column_base")
LISTS = Path("/workspace/exp_cb/cv_sound_folds")
FLOOR, NC = 0.05, 3


def main():
    arm, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
    for f in range(5):
        run = EXP / f"cv_{arm}_f{f}"
        lst = LISTS / f"fold{f}_heldout.txt"
        if not run.is_dir() or not lst.exists():
            print(f"fold{f}: 缺 {'run' if not run.is_dir() else '留出清单'},跳过")
            continue
        out_path = run / "cv_sound.json"
        if out_path.exists():
            print(f"fold{f}: 已存在,跳过")
            continue
        names = [n for n in lst.read_text().split() if n]
        imgs = [SOUND / n for n in names if (SOUND / n).exists()]
        store = {}
        for ck in sorted((run / "epoch_pth").glob("checkpoint_epoch_*.pth")):
            ep = int(ck.stem.split("_")[-1])
            m = from_checkpoint_matched(str(ck), device=device, verbose=False)
            ctx = getattr(m, "model", None)
            if ctx is not None and hasattr(ctx, "device"):
                ctx.device = torch.device(device)
            d = {}
            for p in imgs:
                with Image.open(p) as h:
                    im = h.convert("RGB")
                det = m.predict(im, threshold=FLOOR)
                cls = np.asarray(det.class_id).reshape(-1)
                keep = cls < NC
                d[p.name] = {"scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                             "classes": cls[keep].tolist()}
            store[str(ep)] = d
            del m
            torch.cuda.empty_cache()
        out_path.write_text(json.dumps(store), encoding="utf-8")
        print(f"fold{f}: {len(store)} epoch x {len(imgs)} 张健全图 -> {out_path.name}")


if __name__ == "__main__":
    main()
