#!/usr/bin/env python3
"""Calibrate the S1 quality judge against real photographs.

The judge rejected every synthetic negative in the 2026-08-03 rounds, mostly for
`looks_edited` and `inauthentic`. Before tuning generation against those
verdicts, the instrument itself has to be measured: a judge that calls genuine
survey photographs edited is not measuring edit quality, and optimising against
it would chase noise.

Two controls, run through the exact QC prompt:

* **Specificity** - real empty-label images, unedited, passed as both IMAGE 1 and
  IMAGE 2. Correct answers are damage_visible=False, looks_edited=False,
  authentic=True. Anything else is the judge's false-alarm rate.
* **Sensitivity** - real annotated damaged images, likewise unedited. Correct
  answer is damage_visible=True. A miss here means the gate would wave residual
  damage through.

Both controls pass identical pixels for IMAGE 1 and IMAGE 2, so any
`looks_edited` or `only_change` finding is by construction a false positive.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path

from PIL import Image

import synth_common as sc
from qc_synthetic_negatives import JUDGE_PROMPT, JUDGE_SCHEMA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--out-dir", default="outputs/gemini_synth/qc_calibration")
    p.add_argument("--category", default="column_base", choices=["brace", "column_base"])
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--n-negatives", type=int, default=16)
    p.add_argument("--n-positives", type=int, default=16)
    p.add_argument("--send-size", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--seed", type=int, default=20260803)
    return p.parse_args()


def judge_one(path: Path, args, api_key: str) -> dict:
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
        # identical pixels on both sides: any "edit" finding is a false positive
        enc = sc.encode_image(img, max_side=args.send_size)
        parts = [enc, enc,
                 {"text": JUDGE_PROMPT.format(element_ja=sc.CATEGORY_JA[args.category])}]
        j, err = sc.generate_text(args.model, parts, api_key, timeout=args.timeout,
                                  max_retries=2, response_schema=JUDGE_SCHEMA)
        return {"file": path.name, "judge": j, "error": err}
    except Exception as exc:  # noqa: BLE001
        return {"file": path.name, "judge": None, "error": f"{type(exc).__name__}: {exc}"}


def summarise(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if r.get("judge")]
    n = len(ok)
    if not n:
        print(f"{label}: no successful judgements")
        return {}
    dmg = sum(1 for r in ok if r["judge"].get("damage_visible"))
    edited = sum(1 for r in ok if r["judge"].get("looks_edited"))
    inauth = sum(1 for r in ok if not r["judge"].get("authentic_survey_photo", True))
    notonly = sum(1 for r in ok if not r["judge"].get("only_change_is_damage_removal", True))
    print(f"\n--- {label} (n={n}, unedited real photographs) ---")
    print(f"  damage_visible                 {dmg:3d}  ({dmg/n:.0%})")
    print(f"  looks_edited        FALSE POS  {edited:3d}  ({edited/n:.0%})")
    print(f"  not authentic       FALSE POS  {inauth:3d}  ({inauth/n:.0%})")
    print(f"  collateral change   FALSE POS  {notonly:3d}  ({notonly/n:.0%})")
    return {"n": n, "damage_visible": dmg, "looks_edited": edited,
            "inauthentic": inauth, "collateral": notonly}


def main() -> int:
    args = parse_args()
    api_key = sc.require_api_key()
    rng = random.Random(args.seed)

    data = sc.load_category(Path(args.paired_dir), args.category)
    negs = [j["image"] for j in data["negatives"]]
    poss = [j["image"] for j in data["damaged"]]
    rng.shuffle(negs)
    rng.shuffle(poss)
    negs = negs[: args.n_negatives]
    poss = poss[: args.n_positives]

    print(f"calibrating {args.model} on {args.category}: "
          f"{len(negs)} real negatives, {len(poss)} real positives")

    out: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        neg_rows = list(ex.map(lambda p: judge_one(p, args, api_key), negs))
        pos_rows = list(ex.map(lambda p: judge_one(p, args, api_key), poss))

    out["negatives"] = summarise(neg_rows, "REAL NEGATIVES (should read clean)")
    out["positives"] = summarise(pos_rows, "REAL POSITIVES (should read damaged)")

    for label, rows in (("negatives", neg_rows), ("positives", pos_rows)):
        flagged = [r for r in rows if r.get("judge") and r["judge"].get("looks_edited")]
        if flagged:
            print(f"\n  {label}: examples judged edited although untouched")
            for r in flagged[:4]:
                print(f"    {r['file']}: {(r['judge'].get('edit_evidence') or '')[:150]}")

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{args.category}_calibration.json"
    path.write_text(json.dumps(
        {"summary": out, "negatives": neg_rows, "positives": pos_rows},
        ensure_ascii=False, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
