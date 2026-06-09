"""In-memory region slicing and coordinate mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RegionView:
    image: np.ndarray
    xyxy: tuple[int, int, int, int]
    source_shape: tuple[int, int]


def padded_xyxy(
    xyxy: Sequence[float],
    image_shape: Sequence[int],
    padding_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """Expand xyxy bbox and clamp it to image bounds."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    px = bw * padding_ratio
    py = bh * padding_ratio
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(width, int(round(x2 + px))),
        min(height, int(round(y2 + py))),
    )


def make_region_view(
    image: np.ndarray,
    xyxy: Sequence[float],
    padding_ratio: float = 0.10,
    make_contiguous: bool = False,
) -> RegionView:
    """Return an in-memory slice for downstream inference without writing a crop file."""
    x1, y1, x2, y2 = padded_xyxy(xyxy, image.shape, padding_ratio)
    region = image[y1:y2, x1:x2]
    if make_contiguous:
        region = np.ascontiguousarray(region)
    return RegionView(image=region, xyxy=(x1, y1, x2, y2), source_shape=tuple(image.shape[:2]))


def map_region_xyxy_to_original(
    region_xyxy: Sequence[float],
    region: RegionView,
) -> tuple[float, float, float, float]:
    """Map a detection bbox from region coordinates back to original image coordinates."""
    ox, oy, _, _ = region.xyxy
    x1, y1, x2, y2 = [float(v) for v in region_xyxy]
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
