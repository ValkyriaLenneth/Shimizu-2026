#!/usr/bin/env python3
"""Establish the router's class numbering by measuring it, not reading it.

The router assigned the client's unlabelled pool to 壁类 (49%) and RC柱 (37%),
with 柱脚 at 1.5% -- for a pool the client supplied as additional brace and
column-base imagery. Either the pool is not what it was taken to be, or the
integer class ids the detector emits do not index the class_names list directly.
Detector checkpoints frequently offset those ids, and reading the list as if
they did would silently mislabel every downstream conclusion.

The mapping is measurable rather than guessable. Two frozen test splits exist
whose element is known by construction: 45 column-base images and 58 brace
images. Running the router over each and reporting which id dominates fixes the
numbering directly. Whatever id wins on the column-base split is the column-base
id, whatever wins on the brace split is the brace id, and if the same id wins on
both, or neither wins cleanly, then the router cannot separate these two elements
and the pool split has to be abandoned rather than reinterpreted.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

ROUTER = "/workspace/handoff_20260707_rfdetr_main/models/rfdetr/router_5class/selected_precision_p090_epoch049_thr069.pth"
NAMES = ["天井", "壁类", "RC柱", "ブレース", "柱脚"]
SPLITS = {
    "柱脚 (45 图)": "/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid/test/images",
    "ブレース (58 图)": "/workspace/Shimizu-2026/data/rfdetr_brace_bcd_20260725_test_as_valid/test/images",
}
THR = 0.30


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    model = from_checkpoint_matched(ROUTER, device=device, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    for label, d in SPLITS.items():
        imgs = sorted(p for p in Path(d).iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        tally = Counter()
        for p in imgs:
            with Image.open(p) as h:
                im = h.convert("RGB")
            det = model.predict(im, threshold=THR)
            cls = np.asarray(det.class_id).reshape(-1)
            conf = np.asarray(det.confidence).reshape(-1)
            if not len(cls):
                tally[-1] += 1
                continue
            agg = {}
            for c, s in zip(cls.tolist(), conf.tolist()):
                agg[c] = agg.get(c, 0.0) + s
            tally[max(agg, key=agg.get)] += 1
        print(f"\n{label} —— 已知构件,用于定标:")
        for c, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            name = "无检出" if c == -1 else (NAMES[c] if 0 <= c < len(NAMES) else "越界")
            print(f"  id {c:>2} ({name:>6}) {n:4d}  {n/len(imgs):5.1%}")


if __name__ == "__main__":
    main()
