#!/usr/bin/env python3
"""Oversample the rare grades, adding no images the corpus does not already hold.

The four-target constraint fails on grade D more than on anything else: D carries
10 boxes in the frozen test split against B's 47, and the constraint admits a
solution in only 46% of bootstrap resamples largely because of it. Training is
imbalanced the same way -- 165 B boxes, 67 C, 30 D in a fold -- so the model has
seen the grade that decides feasibility least often.

Both interventions tried before this one failed for a shared reason: each brought
the client's 29 sound photographs into training and shifted the training
distribution away from the evaluation distribution. Negatives cost 0.043 of
damaged-side precision, copy-paste cost 0.075 and made false alarms worse by 0.56
boxes per image. So this one introduces no new imagery at all. It repeats
existing training images that carry the rare grades, which changes how often the
model sees them without changing what it sees.

Repetition is by image rather than by box because the loader works in images: an
image is duplicated as many times as its rarest grade is under-represented, at
most three extra copies, so a D-bearing image that also carries B does not
multiply B out of proportion.

Evaluation folds are untouched and byte-identical to the control's, which is what
keeps the arms paired.
"""
from __future__ import annotations
import collections, json, shutil, sys
from pathlib import Path

CV = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_cv5")
BAL = Path("/workspace/Shimizu-2026/data/rfdetr_column_base_cv5_bal")
MAX_EXTRA = 3
GRADES = "BCD"


def main():
    if BAL.exists():
        shutil.rmtree(BAL)
    summary = []
    for f in range(5):
        src, dst = CV / f"fold{f}", BAL / f"fold{f}"
        for split in ("train", "valid", "test"):
            for kind in ("images", "labels"):
                (dst / split / kind).mkdir(parents=True, exist_ok=True)
                for p in (src / split / kind).iterdir():
                    t = dst / split / kind / p.name
                    if not t.exists():
                        t.symlink_to(p.resolve() if p.is_symlink() else p)
        # Image-level frequency of each grade in this fold's training split.
        holds = collections.defaultdict(set)
        for lab in (src / "train" / "labels").iterdir():
            cs = {int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()}
            for c in cs:
                holds[c].add(lab.stem)
        freq = {c: len(holds[c]) for c in range(3)}
        top = max(freq.values())
        # How many extra copies each grade would need to reach the commonest.
        need = {c: min(MAX_EXTRA, max(0, round(top / max(freq[c], 1)) - 1)) for c in range(3)}
        made = 0
        for lab in sorted((src / "train" / "labels").iterdir()):
            cs = {int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()}
            if not cs:
                continue
            # Driven by the rarest grade present, so a D image that also holds B
            # does not multiply B along with it.
            extra = max(need[c] for c in cs)
            if not extra:
                continue
            imgs = list((src / "train" / "images").glob(f"{lab.stem}.*"))
            if not imgs:
                continue
            img = imgs[0]
            for k in range(extra):
                ni = dst / "train" / "images" / f"rep{k}__{img.name}"
                nl = dst / "train" / "labels" / f"rep{k}__{lab.stem}.txt"
                if not ni.exists():
                    ni.symlink_to(img.resolve() if img.is_symlink() else img)
                if not nl.exists():
                    nl.symlink_to(lab.resolve() if lab.is_symlink() else lab)
                made += 1
        (dst / "data.yaml").write_text(
            f"path: {dst}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 3\n"
            "names:\n  0: 柱脚の損傷程度B\n  1: 柱脚の損傷程度C\n  2: 柱脚の損傷程度D\n")
        # Resulting box-level balance.
        box = collections.Counter()
        for lab in (dst / "train" / "labels").iterdir():
            for l in lab.read_text().splitlines():
                if l.strip():
                    box[int(l.split()[0])] += 1
        n = len(list((dst / "train" / "images").iterdir()))
        summary.append({"fold": f, "extra": made, "train_total": n,
                        **{f"box_{GRADES[c]}": box[c] for c in range(3)}})
        print(f"fold{f}: 图级频率 B{freq[0]} C{freq[1]} D{freq[2]} -> 复制系数 "
              f"B+{need[0]} C+{need[1]} D+{need[2]}  新增 {made} 条,训练集 {n}")
        print(f"        框数 B{box[0]} C{box[1]} D{box[2]}  "
              f"(原 B165 C67 D30 量级)")
    same = all({p.name for p in (BAL / f"fold{f}" / "test" / "images").iterdir()} ==
               {p.name for p in (CV / f"fold{f}" / "test" / "images").iterdir()} for f in range(5))
    print(f"\n评测集与对照组逐张一致: {same}")
    Path("/workspace/exp_cb/cv_bal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
