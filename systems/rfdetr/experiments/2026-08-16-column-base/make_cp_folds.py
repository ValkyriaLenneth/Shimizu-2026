#!/usr/bin/env python3
"""Copy-paste positives, generated per fold so provenance is clean by construction.

Copy-paste is the technique behind one of the two delivered ensemble members, and
its contribution has never been measured under a protocol able to resolve it. It
pastes a real damage crop onto a real sound column base: the pixels are all
photographed, so nothing is hallucinated, and it breaks the corpus's defining
correlation -- every one of the 179 training images contains damage, so "element
present" and "damage present" have never been separable.

The leakage vector is the crop, not the paste target. A damage crop lifted from an
image that lands in the evaluation fold would put that exact damage instance into
training, and the fold would then score a memorised answer. So crops are taken
only from the fold's own training images, and sound targets only from the sound
photographs that fold trains on. Both restrictions are structural rather than
checked afterwards, which is what makes this arm runnable when the Gemini
synthetic set is not: that set's originals could not be traced to any labelled
image by byte hash, by pHash, or by pHash after matching the generator's 768x768
stretch, so its leakage exposure is unknown.

Paste sites come from the sound image's own column base rather than anywhere in
frame. The router locates it; damage sits on the member, and a crop dropped on
floor or sky teaches the model that context, not the damage, is the signal.
"""
from __future__ import annotations
import json, random, shutil, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/workspace/Shimizu-2026/systems/rfdetr/scripts")
from checkpoint_resolution import from_checkpoint_matched

CV = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_cv5")
CP = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_cv5_cp")
SOUND = Path("/workspace/sound_20260807/column_base")
LISTS = Path("/workspace/exp_cb/cv_sound_folds")
ROUTER = "/workspace/handoff_20260707_rfdetr_main/models/rfdetr/router_5class/selected_precision_p090_epoch049_thr069.pth"
CB_CLASS, ROUTER_THR = 4, 0.30
PER_IMAGE, VARIANTS, FEATHER = 2, 2, 0.12
MIN_CROP = 24


def load_boxes(img: Path, lab: Path):
    with Image.open(img) as h:
        W, H = h.size
    out = []
    if lab.exists():
        for line in lab.read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, w, h_ = (float(v) for v in line.split()[:5])
            bw, bh = w * W, h_ * H
            if bw >= MIN_CROP and bh >= MIN_CROP:
                out.append((int(c), x * W - bw / 2, y * H - bh / 2, bw, bh))
    return out, (W, H)


def paste(base: Image.Image, crop: Image.Image, cx: int, cy: int) -> Image.Image:
    """Feathered alpha composite; matches the crop's median to the site's.

    Without the photometric match the pasted region carries its source image's
    exposure, which is a low-level cue the detector can learn instead of the
    damage itself.
    """
    w, h = crop.size
    site = base.crop((cx, cy, cx + w, cy + h))
    a = np.asarray(crop, np.float32)
    b = np.asarray(site, np.float32)
    if b.size:
        a = np.clip(a + (np.median(b, (0, 1)) - np.median(a, (0, 1))), 0, 255)
    f = max(1, int(min(w, h) * FEATHER))
    m = np.ones((h, w), np.float32)
    ramp = np.linspace(0, 1, f)
    m[:f, :] *= ramp[:, None]; m[-f:, :] *= ramp[::-1, None]
    m[:, :f] *= ramp[None, :]; m[:, -f:] *= ramp[::-1][None, :]
    out = base.copy()
    out.paste(Image.fromarray((a * m[..., None] + b * (1 - m[..., None])).astype(np.uint8)),
              (cx, cy))
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    rng = random.Random(20260816)
    sound = sorted(p for p in SOUND.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sfolds = {f: {n for n in (LISTS / f"fold{f}_heldout.txt").read_text().split() if n}
              for f in range(5)}

    model = from_checkpoint_matched(ROUTER, device=device, verbose=False)
    ctx = getattr(model, "model", None)
    if ctx is not None and hasattr(ctx, "device"):
        ctx.device = torch.device(device)
    sites = {}
    for p in sound:
        with Image.open(p) as h:
            im = h.convert("RGB")
        det = model.predict(im, threshold=ROUTER_THR)
        cls = np.asarray(det.class_id).reshape(-1)
        conf = np.asarray(det.confidence).reshape(-1)
        xy = np.asarray(det.xyxy).reshape(-1, 4)
        m = cls == CB_CLASS
        sites[p.name] = xy[m][int(np.argmax(conf[m]))].tolist() if m.any() else None
    del model
    torch.cuda.empty_cache()
    located = sum(1 for v in sites.values() if v)
    print(f"路由器在 {located}/{len(sound)} 张健全图上定位到柱脚", flush=True)

    if CP.exists():
        shutil.rmtree(CP)
    summary = []
    for f in range(5):
        src, dst = CV / f"fold{f}", CP / f"fold{f}"
        for split in ("train", "valid", "test"):
            for kind in ("images", "labels"):
                (dst / split / kind).mkdir(parents=True, exist_ok=True)
                for p in (src / split / kind).iterdir():
                    t = dst / split / kind / p.name
                    if not t.exists():
                        t.symlink_to(p.resolve() if p.is_symlink() else p)
        # Crops: this fold's training images only.
        crops = []
        for p in sorted((src / "train" / "images").iterdir()):
            lab = src / "train" / "labels" / f"{p.stem}.txt"
            boxes, _ = load_boxes(p, lab)
            for c, x, y, w, h in boxes:
                crops.append((p, c, int(x), int(y), int(w), int(h)))
        targets = [p for p in sound if p.name not in sfolds[f] and sites[p.name]]
        made = 0
        for p in targets:
            sx1, sy1, sx2, sy2 = (int(v) for v in sites[p.name])
            for v in range(VARIANTS):
                with Image.open(p) as h:
                    im = h.convert("RGB")
                lines = []
                for _ in range(PER_IMAGE):
                    if not crops:
                        break
                    cp_, c, x, y, w, h_ = rng.choice(crops)
                    with Image.open(cp_) as ch:
                        crop = ch.convert("RGB").crop((x, y, x + w, y + h_))
                    # Scale the crop to the target member's size so a crop from a
                    # close-up does not land as a giant patch on a wide shot.
                    k = max(0.25, min(2.0, (sx2 - sx1) / max(w * 3.0, 1)))
                    nw, nh = max(MIN_CROP, int(w * k)), max(MIN_CROP, int(h_ * k))
                    if nw >= (sx2 - sx1) or nh >= (sy2 - sy1):
                        continue
                    crop = crop.resize((nw, nh), Image.BICUBIC)
                    cx = rng.randint(sx1, max(sx1, sx2 - nw))
                    cy = rng.randint(sy1, max(sy1, sy2 - nh))
                    im = paste(im, crop, cx, cy)
                    W, H = im.size
                    lines.append(f"{c} {(cx+nw/2)/W:.6f} {(cy+nh/2)/H:.6f} {nw/W:.6f} {nh/H:.6f}")
                if not lines:
                    continue
                name = f"cp__f{f}__{p.stem}__v{v}.jpg"
                im.save(dst / "train" / "images" / name, quality=92)
                (dst / "train" / "labels" / f"cp__f{f}__{p.stem}__v{v}.txt").write_text(
                    "\n".join(lines) + "\n")
                made += 1
        (dst / "data.yaml").write_text(
            f"path: {dst}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 3\n"
            "names:\n  0: 柱脚の損傷程度B\n  1: 柱脚の損傷程度C\n  2: 柱脚の損傷程度D\n")
        n_tr = len(list((dst / "train" / "images").iterdir()))
        summary.append({"fold": f, "crops": len(crops), "targets": len(targets),
                        "made": made, "train_total": n_tr})
        print(f"fold{f}: 裁片 {len(crops)} (仅本折训练图) / 目标健全图 {len(targets)} "
              f"-> 合成 {made} 张,训练集 {n_tr}")

    same = all({p.name for p in (CP / f"fold{f}" / "test" / "images").iterdir()} ==
               {p.name for p in (CV / f"fold{f}" / "test" / "images").iterdir()} for f in range(5))
    print(f"\n评测集与对照组逐张一致: {same}")
    Path("/workspace/exp_cb/cv_cp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
