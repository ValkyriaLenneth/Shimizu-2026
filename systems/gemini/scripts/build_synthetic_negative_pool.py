#!/usr/bin/env python3
"""Assemble the accepted S1 negatives into a training-ready pool.

Two jobs, and the second one is the reason this is a script rather than a copy
command.

**Selection.** Only images the calibrated gate accepted are carried through.
`--accept pass` is the strict pool; `--accept pass,review` includes the
borderline ones, which is worth having as a second arm because the review bucket
is dominated by the single-image edit question, whose false-positive rate on
genuine photographs is 8%.

**Split isolation.** A synthetic negative inherits everything from its source
photograph - same element, same scene, same camera position. If the source sits
in the frozen test split, putting its counterfactual into train leaks the test
scene into training as surely as copying the image itself would. Sources are
therefore intersected with the frozen train split, and anything derived from a
test image is dropped and counted. The frozen split also groups near-identical
views under `scene_group_id`, so a source is rejected when it is absent from the
train list, not merely when it appears in the test list.

Output is an ImageFolder-shaped pool plus a manifest carrying full provenance,
so a later training view can mix it at any ratio without re-deriving anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

CATEGORIES = ("brace", "column_base")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synth-dir", default="outputs/gemini_synth/s1_counterfactual_negatives")
    p.add_argument("--split-dir", default=".local_artifacts/handoff_20260726/split")
    p.add_argument("--out-dir", default="outputs/gemini_synth/s1_accepted")
    p.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    p.add_argument("--accept", default="pass",
                   help="comma separated verdicts to carry through, e.g. pass,review")
    p.add_argument("--require-split", action="store_true", default=True,
                   help="drop negatives whose source is not in the frozen train split")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_train_stems(split_dir: Path, category: str) -> set[str] | None:
    path = split_dir / f"{category}_split.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    splits = data.get("splits", {})
    train = splits.get("train", {})
    stems = set(train.keys()) if isinstance(train, dict) else set(train)
    stems |= set(data.get("train_negatives", []))
    return stems


def main() -> int:
    args = parse_args()
    accept = {v.strip() for v in args.accept.split(",") if v.strip()}
    split_dir = Path(args.split_dir)
    out_root = Path(args.out_dir)

    grand: dict[str, dict] = {}

    for category in args.categories:
        synth = Path(args.synth_dir) / category
        qc_path = synth / "qc_results.json"
        if not qc_path.exists():
            print(f"{category}: no qc_results.json, skipping")
            continue

        rows = json.loads(qc_path.read_text())
        gen: dict[str, dict] = {}
        gen_path = synth / "generation_results.jsonl"
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("ok"):
                    gen[d["stem"]] = d

        train_stems = load_train_stems(split_dir, category)
        if train_stems is None and args.require_split:
            print(f"{category}: no frozen split file - refusing to build without "
                  f"split isolation")
            continue

        out_cat = out_root / category
        img_dir, lab_dir = out_cat / "images", out_cat / "labels"
        if not args.dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)

        stats = Counter()
        entries = []
        for r in rows:
            stem, verdict = r["stem"], r.get("verdict")
            stats[f"verdict_{verdict}"] += 1
            if verdict not in accept:
                stats["dropped_verdict"] += 1
                continue
            if train_stems is not None and stem not in train_stems:
                stats["dropped_not_in_train_split"] += 1
                continue
            rec = gen.get(stem)
            if not rec:
                stats["dropped_no_record"] += 1
                continue
            src = Path(rec["output_image"])
            if not src.exists():
                stats["dropped_missing_file"] += 1
                continue

            if not args.dry_run:
                shutil.copy2(src, img_dir / src.name)
                (lab_dir / f"{src.stem}.txt").write_text("")
            stats["accepted"] += 1
            entries.append({
                "synthetic_stem": src.stem,
                "source_stem": stem,
                "verdict": verdict,
                "reasons": r.get("reasons", []),
                "damage_profile": rec.get("damage_profile"),
                "n_boxes_repaired": rec.get("n_boxes"),
                "judge": {k: (rec.get("judge") or {}).get(k) for k in ()},
                "residual": (r.get("judge") or {}).get("residual_at_repair_site"),
            })

        manifest = {
            "category": category,
            "accepted_verdicts": sorted(accept),
            "split_isolated": train_stems is not None,
            "counts": dict(stats),
            "entries": entries,
        }
        grand[category] = manifest
        if not args.dry_run:
            (out_cat / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2))

        print(f"\n=== {category} ===")
        print(f"  judged            {len(rows)}")
        for k in ("verdict_pass", "verdict_review", "verdict_reject"):
            if stats[k]:
                print(f"    {k:22s} {stats[k]}")
        print(f"  accepted          {stats['accepted']}")
        for k in ("dropped_verdict", "dropped_not_in_train_split",
                  "dropped_no_record", "dropped_missing_file"):
            if stats[k]:
                print(f"    {k:26s} {stats[k]}")
        prof = Counter(e["damage_profile"] for e in entries)
        if prof:
            print(f"  damage profile    {dict(prof)}")

    if not args.dry_run and grand:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "pool_summary.json").write_text(
            json.dumps({k: v["counts"] for k, v in grand.items()},
                       ensure_ascii=False, indent=2))
        print(f"\nwrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
