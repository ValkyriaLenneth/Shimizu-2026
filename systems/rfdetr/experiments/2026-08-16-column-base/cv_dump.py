#!/usr/bin/env python3
"""Dump one fold's per-epoch detections so folds can be pooled before scoring.

The point of cross-validation here is to threshold-search once over all 320 boxes
rather than five times over 64. That requires the raw detections, not per-fold
scores: a precision computed inside a fold and then averaged is not the precision
of the pooled set, and it would carry exactly the small-sample noise the folds
exist to remove.

So each fold writes its detections for every epoch, and the aggregation step
pools them and searches thresholds on the union.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

FLOOR, NC = 0.05, 3


def main():
    run, fold_dir, device = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    imgs = sorted(p for p in (fold_dir / "test" / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = {}
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
            cls = np.asarray(det.class_id).reshape(-1); keep = cls < NC
            d[p.name] = {"boxes": np.asarray(det.xyxy).reshape(-1, 4)[keep].tolist(),
                         "scores": np.asarray(det.confidence).reshape(-1)[keep].tolist(),
                         "classes": cls[keep].tolist()}
        out[str(ep)] = d
        del m; torch.cuda.empty_cache()
        print(f"  epoch {ep} 完成", flush=True)
    (run / "cv_dets.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"写入 {run/'cv_dets.json'} ({len(out)} epoch x {len(imgs)} 图)")


if __name__ == "__main__":
    main()
