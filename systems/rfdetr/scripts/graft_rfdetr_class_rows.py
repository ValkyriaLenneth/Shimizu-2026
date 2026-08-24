#!/usr/bin/env python3
"""Graft selected final-classifier rows while keeping all shared RF-DETR weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FINAL_KEYS = ("class_embed.weight", "class_embed.bias")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--classes", default="3,4")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--include-encoder-heads", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0, 1]")
    classes = [int(value) for value in args.classes.split(",") if value.strip()]

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.finetuned, map_location="cpu", weights_only=False)
    state_key = "model" if "model" in base else "state_dict"
    base_state = base[state_key]
    tuned_state = tuned[state_key]
    keys = list(FINAL_KEYS)
    if args.include_encoder_heads:
        keys.extend(
            key for key in base_state
            if key.startswith("transformer.enc_out_class_embed.")
            and (key.endswith(".weight") or key.endswith(".bias"))
        )
    for key in keys:
        if key not in base_state or key not in tuned_state:
            raise KeyError(key)
        updated = base_state[key].clone()
        for class_id in classes:
            updated[class_id] = base_state[key][class_id].mul(1 - args.alpha).add(
                tuned_state[key][class_id], alpha=args.alpha
            )
        base_state[key] = updated
    base["class_row_graft"] = {
        "base": str(Path(args.base).resolve()),
        "finetuned": str(Path(args.finetuned).resolve()),
        "classes": classes,
        "alpha": args.alpha,
        "keys": keys,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    print(f"wrote {output} classes={classes} alpha={args.alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
