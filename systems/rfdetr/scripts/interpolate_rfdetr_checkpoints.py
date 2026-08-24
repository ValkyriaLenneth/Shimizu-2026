#!/usr/bin/env python3
"""Linearly interpolate compatible RF-DETR checkpoint model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="finetuned weight in [0, 1]")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0, 1]")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.finetuned, map_location="cpu", weights_only=False)
    state_key = "model" if "model" in base else "state_dict"
    if state_key not in tuned:
        raise KeyError(f"missing {state_key} in finetuned checkpoint")
    if base[state_key].keys() != tuned[state_key].keys():
        raise ValueError("checkpoint parameter keys differ")
    merged = {}
    for name, base_value in base[state_key].items():
        tuned_value = tuned[state_key][name]
        if torch.is_floating_point(base_value):
            merged[name] = base_value.mul(1 - args.alpha).add(tuned_value, alpha=args.alpha)
        else:
            merged[name] = base_value
    base[state_key] = merged
    base.setdefault("interpolation", {})
    base["interpolation"] = {
        "base": str(Path(args.base).resolve()),
        "finetuned": str(Path(args.finetuned).resolve()),
        "alpha": args.alpha,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    print(f"wrote {output} alpha={args.alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
