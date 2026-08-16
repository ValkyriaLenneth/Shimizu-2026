#!/usr/bin/env python3
"""Does the frozen test split share imagery with what the models were trained on?

Every delivered number rests on 45 held-out photographs. Their integrity has been
checked for the obvious failures -- no byte-identical duplicates, no missing
labels, 72 boxes matching the record -- and the split passes all of them. But
byte equality is the weakest possible test. Site photographs of the same column
base taken seconds apart, or the same photograph resaved at a different quality,
are different files and identical evidence, and a held-out set containing them
reports a recall the model has not earned.

This has never been checked on this project, and it matters more than any
remaining tuning. If the test split overlaps the training set, the delivered
recall of 0.708 is optimistic by an unknown amount and the freeze document needs
that stated. If it does not, every number in the document is standing on ground
that has now been examined rather than assumed.

Two independent similarity measures, because each fails differently:

  pHash    a 64-bit perceptual hash from the low-frequency DCT of a 32x32
           greyscale reduction. Invariant to rescaling, JPEG quality, and small
           brightness shifts; catches the same photograph stored twice.
  pixel    cosine similarity of contrast-normalised 64x64 greyscale. Sensitive to
           framing rather than encoding; catches consecutive frames of the same
           member, which pHash can miss when the camera moved slightly.

Both are compared against the 179 training images and the 45 validation images,
and any test image scoring above threshold on either is reported with its match
so it can be inspected by eye rather than trusted to a number.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

DS = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
OUT = Path("/workspace/exp_cb/e27_leakage")
PHASH_BITS = 10          # 10x10 low-frequency block, minus DC -> 99 bits
HAMMING_FLAG = 12        # out of 99; below this two images are visually the same shot
PIXEL_FLAG = 0.92


def phash(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), np.float64)
    # 2-D DCT-II via the orthonormal basis; scipy is not a dependency here.
    n = 32
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[:, 0] *= 1 / np.sqrt(2)
    d = basis.T @ g @ basis
    low = d[:PHASH_BITS, :PHASH_BITS].flatten()[1:]     # drop DC
    return low > np.median(low)


def pixvec(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS), np.float64).flatten()
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def load(split: str):
    d = DS / split / "images"
    if not d.is_dir():
        return [], None, None
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    hs, vs = [], []
    for p in paths:
        with Image.open(p) as h:
            im = h.convert("RGB")
        hs.append(phash(im)); vs.append(pixvec(im))
    return [p.name for p in paths], np.array(hs), np.array(vs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tn, th, tv = load("test")
    print(f"test {len(tn)} 张", flush=True)
    findings, summary = [], {}
    for split in ("train", "valid"):
        rn, rh, rv = load(split)
        if not rn:
            continue
        print(f"{split} {len(rn)} 张 —— 比对中", flush=True)
        ham = (th[:, None, :] != rh[None, :, :]).sum(2)          # (T, R)
        cos = tv @ rv.T
        flagged = 0
        for i, name in enumerate(tn):
            j_h = int(np.argmin(ham[i])); j_c = int(np.argmax(cos[i]))
            hit_h, hit_c = ham[i, j_h] <= HAMMING_FLAG, cos[i, j_c] >= PIXEL_FLAG
            if hit_h or hit_c:
                flagged += 1
                findings.append({"test": name, "split": split,
                                 "phash_match": rn[j_h], "hamming": int(ham[i, j_h]),
                                 "pixel_match": rn[j_c], "cosine": round(float(cos[i, j_c]), 4),
                                 "by": ("phash" if hit_h else "") + ("+pixel" if hit_c else "")})
        summary[split] = {"n_ref": len(rn), "flagged": flagged,
                          "min_hamming": int(ham.min()), "max_cosine": round(float(cos.max()), 4),
                          "median_min_hamming": float(np.median(ham.min(1))),
                          "median_max_cosine": round(float(np.median(cos.max(1))), 4)}
        print(f"  最小汉明距离 {ham.min()}/99 (中位 {np.median(ham.min(1)):.0f}), "
              f"最大余弦 {cos.max():.3f} (中位 {np.median(cos.max(1)):.3f}), 触发 {flagged}")

    (OUT / "leakage.json").write_text(json.dumps(
        {"summary": summary, "findings": findings}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n阈值: 汉明 <= {HAMMING_FLAG}/99 或 余弦 >= {PIXEL_FLAG}")
    if not findings:
        print("结论: 未发现近似重复 —— 45 张测试图与 train/valid 无视觉重叠,交付数字不受泄漏影响")
    else:
        print(f"结论: {len(findings)} 项需人工确认:")
        for f in findings[:20]:
            print(f"  {f['test']} <-> {f['split']}/{f['phash_match']} "
                  f"(汉明 {f['hamming']}, 余弦 {f['cosine']}) via {f['by']}")
        print(f"  全部写入 {OUT/'leakage.json'}")


if __name__ == "__main__":
    main()
