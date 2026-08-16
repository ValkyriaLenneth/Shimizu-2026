#!/usr/bin/env python3
"""Build 5-fold splits over every labelled column-base image.

Every training-side conclusion this week died the same way: the effect was
smaller than the run-to-run spread. That spread is 0.06 to 0.10, and the cause is
not the training -- it is that each run is scored on 45 images holding 72 boxes,
of which 15 are grade C and 10 are grade D. A statistic computed on ten boxes
cannot resolve the differences being asked about, so five interventions were
measured, all looked promising once, and none survived replication.

The client has no further labelled data, but the project does hold more than the
evaluation uses: 179 training images with 248 boxes sit alongside the 45-image
test split and are never scored. Cross-validation spends them. Each of five folds
is scored by a model that did not train on it, and the fold results aggregate to
an evaluation over all 224 images and 320 boxes -- 4.4 times the boxes and, for
grade D, 39 instead of 10.

Two things this does not do, stated plainly so the outputs are not misread:

  It is not a delivery measurement. The frozen 45-image split is the delivery
  protocol and stays untouched by it; these folds mix that split into training,
  so nothing computed here may be quoted as a delivered number.

  It does not remove the epoch-selection contamination. Fine-tuning starts from
  a checkpoint that already saw the 179 training images, and those images appear
  in four of the five evaluation folds. That inflates every arm identically, so
  paired comparisons between arms remain valid while absolute values do not.

Folds are stratified on the rarest grade present in each image, so the 39 D boxes
spread across folds instead of concentrating in one.
"""
from __future__ import annotations
import json, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_bcd_20260725_test_as_valid")
DST = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_cv5")
K = 5
GRADES = "BCD"


def collect():
    items = []
    for split in ("train", "test"):
        idir, ldir = SRC / split / "images", SRC / split / "labels"
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab = ldir / f"{p.stem}.txt"
            lines = [l for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
            cls = {int(l.split()[0]) for l in lines}
            items.append({"img": p, "lab": lab, "classes": sorted(cls),
                          "n": len(lines), "src": split})
    return items


def main():
    items = collect()
    print(f"合并 {len(items)} 图 ({Counter(i['src'] for i in items)})")
    # Stratify on the rarest grade the image contains: D dominates the key when
    # present, then C, then B. Ten D-bearing images spread evenly matters far
    # more than balancing the plentiful B.
    def key(it):
        for c in (2, 1, 0):
            if c in it["classes"]:
                return c
        return -1
    buckets = defaultdict(list)
    for it in items:
        buckets[key(it)].append(it)
    folds = [[] for _ in range(K)]
    for k in sorted(buckets):
        for i, it in enumerate(sorted(buckets[k], key=lambda x: x["img"].name)):
            folds[i % K].append(it)

    if DST.exists():
        shutil.rmtree(DST)
    meta = []
    for f in range(K):
        val = folds[f]
        trn = [it for g in range(K) if g != f for it in folds[g]]
        for split, group in (("train", trn), ("valid", val), ("test", val)):
            (DST / f"fold{f}" / split / "images").mkdir(parents=True, exist_ok=True)
            (DST / f"fold{f}" / split / "labels").mkdir(parents=True, exist_ok=True)
            for it in group:
                dst_i = DST / f"fold{f}" / split / "images" / it["img"].name
                if not dst_i.exists():
                    dst_i.symlink_to(it["img"])
                dst_l = DST / f"fold{f}" / split / "labels" / f"{it['img'].stem}.txt"
                if not dst_l.exists():
                    if it["lab"].exists():
                        dst_l.symlink_to(it["lab"])
                    else:
                        dst_l.write_text("")
        cnt = Counter()
        for it in val:
            for l in ([x for x in it["lab"].read_text().splitlines() if x.strip()]
                      if it["lab"].exists() else []):
                cnt[int(l.split()[0])] += 1
        meta.append({"fold": f, "n_train": len(trn), "n_val": len(val),
                     **{f"val_{GRADES[c]}": cnt[c] for c in range(3)}})
        print(f"  fold{f}: train {len(trn)} / eval {len(val)} 图, "
              f"框 B{cnt[0]} C{cnt[1]} D{cnt[2]}")
    tot = {g: sum(m[f"val_{g}"] for m in meta) for g in GRADES}
    print(f"\n五折合计评测 {sum(m['n_val'] for m in meta)} 图 / "
          f"{sum(tot.values())} 框 (B{tot['B']} C{tot['C']} D{tot['D']})")
    print(f"对照: 冻结测试集 45 图 / 72 框 (B47 C15 D10)")
    (DST / "folds.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # The config the trainer reads points at a dataset directory; each fold gets
    # its own copy so the existing runner needs no new flags.
    cfg = SRC.parent.parent / "systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml"
    print(f"\n数据集写入 {DST}/fold0..{K-1}")


if __name__ == "__main__":
    main()
