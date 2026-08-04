#!/usr/bin/env python3
"""Load an RF-DETR checkpoint at the resolution it was actually trained at.

``rfdetr.from_checkpoint`` restores the model class and ``num_classes`` but
**not** ``resolution`` - that field is absent from the 72 keys a checkpoint
stores under ``args``. A checkpoint trained at a non-default resolution is
therefore rebuilt at the class default (576 for RFDETRMedium) and then asked to
load weights whose positional-encoding tensor has a different length. Depending
on the loader that either raises or silently drops the tensor, and the second
case is worse: the model still produces detections, so an evaluation looks
successful while reporting numbers for a model whose position embeddings never
loaded.

The resolution is recoverable from the weights themselves. The backbone stores
``position_embeddings`` shaped ``(1, 1 + grid * grid, dim)``, where the leading
token is the CLS token and ``grid == resolution // patch_size``:

    576 -> (1, 1297, 384)   1296 = 36 x 36, 36 * 16 = 576
    896 -> (1, 3137, 384)   3136 = 56 x 56, 56 * 16 = 896

This module reads that tensor and passes the recovered resolution back into the
constructor, which keeps train-time and eval-time preprocessing on the same
path. Path mismatches of exactly this kind are what invalidated the 2026-07-26
tiled-inference experiment, so the resolution actually used is logged rather
than assumed.
"""

from __future__ import annotations

import math

import torch

PE_KEY_SUFFIX = "encoder.embeddings.position_embeddings"
DEFAULT_PATCH_SIZE = 16


def _state_dict(checkpoint: dict) -> dict:
    for key in ("model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def resolution_from_checkpoint(path: str, patch_size: int = DEFAULT_PATCH_SIZE) -> int | None:
    """Recover the training resolution from the positional-encoding tensor.

    Returns ``None`` when the tensor is absent or its length is not
    ``1 + grid**2``, so callers can fall back to the class default rather than
    guess a wrong resolution.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for name, tensor in _state_dict(checkpoint).items():
        if not name.endswith(PE_KEY_SUFFIX):
            continue
        if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
            continue
        tokens = int(tensor.shape[1]) - 1  # drop the CLS token
        grid = int(round(math.sqrt(tokens)))
        if grid * grid == tokens and grid > 0:
            return grid * patch_size
    return None


def from_checkpoint_matched(path: str, *, verbose: bool = True, **kwargs):
    """``rfdetr.from_checkpoint`` with the checkpoint's own resolution restored.

    Any explicit ``resolution`` in *kwargs* wins, so a caller can still override
    deliberately; otherwise the value recovered from the weights is used.
    """
    import rfdetr

    if "resolution" not in kwargs:
        recovered = resolution_from_checkpoint(path)
        if recovered is not None:
            kwargs["resolution"] = recovered
        elif verbose:
            print(
                f"  [resolution] could not recover from {path}; "
                f"falling back to the model-class default"
            )
    if verbose and "resolution" in kwargs:
        print(f"  [resolution] building model at {kwargs['resolution']} px (from checkpoint)")
    return rfdetr.from_checkpoint(path, **kwargs)
