#!/usr/bin/env python3
"""Can a plain classifier tell a damaged element crop from a sound one?

This is a diagnostic, not a product. Everything measured on 2026-08-04 says the
detector learned to find the *element* rather than the damage: it fires on 0% of
the real training negatives (memorised) but on 95.7% of unseen sound images, and
train recall is 0.995 against test 0.590. Ten interventions at the detector level
failed.

Before proposing a two-stage design - detect the element, then classify the crop -
the premise has to be checked: are damaged and sound crops separable at all by a
frozen feature extractor plus a linear head? Classification needs far less data
than detection, so if the answer is yes the detector's failure is a framing
problem; if the answer is no, the labels or the crops themselves are the problem
and no amount of detector work will help.

Deliberately minimal: frozen DINOv2 features (the same backbone RF-DETR uses, so
this is not a capacity argument) plus logistic regression. Split is by SOURCE
IMAGE, never by crop, because several crops can come from one photograph and a
random crop split would leak the scene and inflate the score.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--crops-dir", default="outputs/gemini_synth/grade_references/crops")
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--audit-dir", default="outputs/rfdetr_new_classes/empty_label_audit")
    p.add_argument("--min-site-score", type=float, default=0.20)
    p.add_argument("--sound-per-image", type=int, default=4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def sound_crops(paired: Path, cat: str, audit_dir: Path, min_score: float,
                per_image: int, rng: random.Random) -> list[tuple[Image.Image, str]]:
    """Element patches cut from the zero-box photographs, via the baseline's own boxes."""
    path = audit_dir / f"{cat}_audit.json"
    sites = {}
    if path.exists():
        audit = json.loads(path.read_text(encoding="utf-8"))
        for rec in audit["records"]:
            d = [x for x in rec.get("detections", []) if x["score"] >= min_score]
            if d:
                sites[rec["stem"]] = d
    out = []
    for lab in sorted((paired / cat / "labels").glob("*.txt")):
        if lab.read_text().strip():
            continue
        det = sites.get(lab.stem, [])
        if not det:
            continue
        img_path = next((p for p in (paired / cat / "images").iterdir()
                         if p.stem == lab.stem and p.suffix.lower() in IMAGE_EXTS), None)
        if img_path is None:
            continue
        with Image.open(img_path) as h:
            im = h.convert("RGB")
        for d in det[:per_image]:
            x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im.width, x2), min(im.height, y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            out.append((im.crop((x1, y1, x2, y2)), lab.stem))
    return out


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dmg: list[tuple[Image.Image, str, str]] = []
    for g in ("B", "C", "D"):
        d = Path(args.crops_dir) / args.category / g
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            # crop filenames carry the source stem as a prefix
            src = p.stem.split("_")[0]
            with Image.open(p) as h:
                dmg.append((h.convert("RGB"), src, g))
    snd_raw = sound_crops(Path(args.paired_dir), args.category, Path(args.audit_dir),
                          args.min_site_score, args.sound_per_image, rng)
    snd = [(im, src, "none") for im, src in snd_raw]
    print(f"category={args.category}  damaged crops={len(dmg)}  sound crops={len(snd)}")
    if len(snd) < 20 or len(dmg) < 20:
        print("  not enough crops on one side to conclude anything")
        return 1

    # Split by SOURCE IMAGE so no photograph contributes to both sides of the split.
    srcs = sorted({s for _, s, _ in dmg} | {s for _, s, _ in snd})
    rng.shuffle(srcs)
    cut = int(len(srcs) * 0.75)
    train_src, test_src = set(srcs[:cut]), set(srcs[cut:])

    import torchvision.transforms as T
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    device = torch.device(args.device)
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    backbone.eval().to(device)

    def feats(items):
        X, y, keep = [], [], []
        with torch.no_grad():
            for i in range(0, len(items), 64):
                batch = items[i:i + 64]
                t = torch.stack([tf(im) for im, _, _ in batch]).to(device)
                f = backbone(t).cpu().numpy()
                X.append(f)
                y += [0 if lab == "none" else 1 for _, _, lab in batch]
                keep += [lab for _, _, lab in batch]
        return np.concatenate(X), np.array(y), keep

    all_items = dmg + snd
    tr = [it for it in all_items if it[1] in train_src]
    te = [it for it in all_items if it[1] in test_src]
    print(f"  split by source image: train {len(tr)} crops / test {len(te)} crops")
    Xtr, ytr, _ = feats(tr)
    Xte, yte, gte = feats(te)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum()); tn = int(((pred == 0) & (yte == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    auc = roc_auc_score(yte, prob) if len(set(yte)) > 1 else float("nan")
    print(f"\n  === 有损伤 vs 完好，二分类（冻结 DINOv2 + 逻辑回归）===")
    print(f"  test crops: {len(yte)}  (damaged {int(yte.sum())} / sound {int((yte==0).sum())})")
    print(f"  ROC-AUC   : {auc:.3f}")
    print(f"  precision : {prec:.3f}   recall: {rec:.3f}")
    print(f"  混淆矩阵  : TP={tp} FP={fp} FN={fn} TN={tn}")
    # recall at a precision floor, to compare against the detector's operating point
    order = np.argsort(-prob)
    best = None
    for k in range(1, len(order) + 1):
        sel = order[:k]
        p_ = float((yte[sel] == 1).mean())
        r_ = float((yte[sel] == 1).sum() / max(1, yte.sum()))
        if p_ >= 0.60:
            best = (r_, p_)
    if best:
        print(f"  P>=0.60 下最高 recall: {best[0]:.3f} (P={best[1]:.3f})")
    by_grade = Counter(g for g, yy, pp in zip(gte, yte, pred) if yy == 1 and pp == 1)
    tot_grade = Counter(g for g, yy in zip(gte, yte) if yy == 1)
    print("  按等级的召回: " + ", ".join(
        f"{g} {by_grade[g]}/{tot_grade[g]}={by_grade[g]/tot_grade[g]:.2f}"
        for g in ("B", "C", "D") if tot_grade[g]))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "category": args.category, "auc": auc, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
