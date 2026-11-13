"""Region utilities: boxes <-> masks, overlap measures, and the
shape-matched translation control (same area and same perimeter as C)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

Box = Sequence[float]  # [x1, y1, x2, y2] in original-image pixel coordinates


def boxes_to_mask(boxes: Sequence[Box], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        if b is None or len(b) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        xa, xb = int(np.floor(max(0.0, x1))), int(np.ceil(min(float(w), x2)))
        ya, yb = int(np.floor(max(0.0, y1))), int(np.ceil(min(float(h), y2)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    return m


def mask_to_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def area(mask: np.ndarray) -> int:
    return int(mask.sum())


def overlap_frac(a: np.ndarray, b: np.ndarray) -> float:
    """|a ∩ b| / |a|  (0.0 if a is empty). GP = overlap_frac(C, M)."""
    na = a.sum()
    if na == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / na)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / u)


def translate_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    max_overlap: float = 0.1,
    max_tries: int = 200,
    avoid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return a copy of `mask` rigidly shifted to a random in-bounds location.

    The shifted region has exactly the same shape (area, perimeter, seam
    length) as the original, so reverting it produces the same amount of
    real/generated seam. Among tries, we accept the first shift whose overlap
    with the original is <= max_overlap; otherwise we return the
    least-overlapping shift found. Returns (shifted_mask, overlap_frac).

    avoid: region the shifted copy must stay out of (default: the mask
    itself). Controls for an audit should avoid BOTH the cited region and the
    true edit mask M: a control that lands on M removes real evidence and
    biases the necessity gap downwards.
    """
    ref = mask if avoid is None else avoid
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy(), 0.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    dy_lo, dy_hi = -int(y0), int(h - 1 - y1)
    dx_lo, dx_hi = -int(x0), int(w - 1 - x1)
    if dy_lo == dy_hi == 0 and dx_lo == dx_hi == 0:
        # Region spans the whole image: no shape-matched control exists.
        return mask.copy(), 1.0
    best, best_ov = None, 2.0
    for _ in range(max_tries):
        dy = int(rng.integers(dy_lo, dy_hi + 1))
        dx = int(rng.integers(dx_lo, dx_hi + 1))
        if dy == 0 and dx == 0:
            continue
        shifted = np.zeros_like(mask)
        shifted[ys + dy, xs + dx] = True
        ov = overlap_frac(shifted, ref)
        if ov < best_ov:
            best, best_ov = shifted, ov
        if ov <= max_overlap:
            break
    if best is None:  # extremely constrained geometry
        return mask.copy(), 1.0
    return best, float(best_ov)


def blob_in_region(region: np.ndarray, area: int, rng: np.random.Generator) -> np.ndarray | None:
    """A compact set of `area` pixels of `region`: the pixels of the region
    nearest to a random seed inside it. Returns None if the region is smaller
    than `area`. Used for the matched in-edit control: an equally large piece
    of the true edit that the detector did NOT cite."""
    ys, xs = np.nonzero(region)
    if area <= 0 or len(ys) < area:
        return None
    i = int(rng.integers(0, len(ys)))
    d = (ys - ys[i]) ** 2 + (xs - xs[i]) ** 2
    take = np.argsort(d, kind="stable")[:area]
    out = np.zeros_like(region, dtype=bool)
    out[ys[take], xs[take]] = True
    return out
