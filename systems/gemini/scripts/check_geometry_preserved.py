#!/usr/bin/env python3
"""Does the repair keep the member, or rebuild it?

The dominant failure mode of S1 on steel is not a bad repair, it is a convincing
substitution: `e-00085` had a severed brace replaced by a clean bolted
connection, `e-00034` grew a solid gusset plate at a bracing crossing, and
`f-00254` turned a corroded rod into a differently-shaped member. All three were
passed by the QC gate, because the gate asks whether the damage is gone and a
rebuilt member scores full marks on that question.

Such an image is worse than a failed repair. It enters training labelled "no
damage" while showing a component that does not exist anywhere in the real
corpus, so the model can learn the fabricated part as evidence of soundness.

Adding the question to the existing gate is not enough, and the reason is worth
recording: the gate crops to the repaired window at native resolution, because
the 2026-08-03 calibration found that hairline cracks vanish when the whole
frame is downscaled. But a member's outline is not visible inside that crop.
Asked the geometry question on the crop, the judge answered "preserved" for
`e-00085`; asked the same question on the whole frame, it answered

    structure_preserved: false
    structure_change_description: "Splice plate and bolts added."   (confidence 0.95)

So the two questions need different framings and cannot share one call. This
script runs the geometry question at whole-frame scale and writes a verdict per
image, to be intersected with the gate's own verdicts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from PIL import Image

import synth_common as sc

GEOMETRY_SCHEMA = {
    "type": "object",
    "properties": {
        "structure_preserved": {"type": "boolean"},
        "structure_change_description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["structure_preserved", "structure_change_description", "confidence"],
}

GEOMETRY_PROMPT = """IMAGE 1 is a Japanese building survey photograph of {element_ja} with structural
damage. IMAGE 2 is the same photograph after an automated repair that was supposed to
remove the damage and change nothing else.

Judge ONE thing: is IMAGE 2 the SAME PHYSICAL OBJECT as IMAGE 1, only in sound condition?

A valid repair changes surface condition only - a crack is filled, rust is cleaned off,
a spall is made good. The steel and concrete keep their shapes.

Answer structure_preserved = false if the member was rebuilt or substituted, for example:
  - a bolt, nut, gusset plate, splice plate, weld, bracket or stiffener appears that was
    not in IMAGE 1, or one present in IMAGE 1 disappears
  - the outline, width, cross-section or edge profile of a member changes
  - a severed or deformed member is shown as a new, differently-shaped component
  - a rod, pipe or bar becomes an object of another shape

Answer true only if every structural element keeps its original shape, position and
count, and the only difference is the condition of surfaces.

Be strict. A convincingly rebuilt member is a WORSE outcome than a visibly failed
repair, so when a member's geometry looks different, answer false even though the
result looks like a plausible sound structure.

structure_change_description: name the component added, removed or reshaped; empty
string if structure_preserved is true."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth-dir", default="outputs/gemini_synth/s1_counterfactual_negatives")
    p.add_argument("--paired-dir",
                   default=".local_artifacts/handoff_20260726/data/new_classes_paired_20260724")
    p.add_argument("--category", default="brace", choices=["brace", "column_base"])
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--send-size", type=int, default=1280,
                   help="whole frame, large enough to read a member outline")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def judge(stem: str, src: Path, syn: Path, args, key: str) -> dict:
    try:
        with Image.open(src) as a, Image.open(syn) as b:
            orig, edited = a.convert("RGB"), b.convert("RGB")
            parts = [
                sc.encode_image(orig, max_side=args.send_size),
                sc.encode_image(edited, max_side=args.send_size),
                {"text": GEOMETRY_PROMPT.format(element_ja=sc.CATEGORY_JA[args.category])},
            ]
    except Exception as exc:
        return {"stem": stem, "error": f"open_failed: {exc}"}
    out, err = sc.generate_text(args.model, parts, key, timeout=args.timeout,
                                response_schema=GEOMETRY_SCHEMA)
    if out is None:
        return {"stem": stem, "error": err or "judge_failed"}
    return {
        "stem": stem,
        "structure_preserved": bool(out.get("structure_preserved")),
        "change": out.get("structure_change_description", ""),
        "confidence": out.get("confidence"),
    }


def main() -> int:
    args = parse_args()
    key = sc.require_api_key()
    syn_dir = Path(args.synth_dir) / args.category / "images"
    src_dir = Path(args.paired_dir) / args.category / "images"
    src_by_stem = {p.stem: p for p in src_dir.iterdir()}

    jobs = []
    for p in sorted(syn_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = p.stem.replace("_s1neg", "")
        src = src_by_stem.get(stem)
        if src is not None:
            jobs.append((stem, src, p))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"geometry check  category={args.category}  images={len(jobs)}  model={args.model}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(judge, s, a, b, args, key): s for s, a, b in jobs}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if "error" in r:
                print(f"[{i}/{len(jobs)}] ERR  {r['stem']}: {r['error'][:60]}")
            else:
                mark = "keep" if r["structure_preserved"] else "REBUILT"
                print(f"[{i}/{len(jobs)}] {mark:<8}{r['stem']:<12}{r['change'][:60]}")

    results.sort(key=lambda r: r["stem"])
    kept = [r for r in results if r.get("structure_preserved")]
    rebuilt = [r for r in results if "error" not in r and not r["structure_preserved"]]
    print(f"\n  几何保持 (可用) : {len(kept):>3} / {len(results)}")
    print(f"  几何被重建 (剔除): {len(rebuilt):>3}")
    if rebuilt:
        from collections import Counter
        words = Counter()
        for r in rebuilt:
            for w in ("plate", "bolt", "weld", "bracket", "stiffener", "rod", "outline", "shape"):
                if w in r["change"].lower():
                    words[w] += 1
        print(f"  被改造的部件词频: {dict(words)}")

    out = Path(args.out_json) if args.out_json else \
        Path(args.synth_dir) / args.category / "geometry_check.json"
    out.write_text(json.dumps({
        "category": args.category, "model": args.model, "send_size": args.send_size,
        "counts": {"preserved": len(kept), "rebuilt": len(rebuilt)},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
