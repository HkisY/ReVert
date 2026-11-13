"""Forensic-neutral interventions.

    R(x, S) = S * x_r + (1 - S) * x      revert region S to real pixels
    K(x, S) = S * x   + (1 - S) * x_r    keep only region S of x

Every pixel these operators introduce comes from the real camera image x_r,
so they add no new generative trace. The only side effect is a seam at the
region border, which the shape-matched translation control cancels.
"""

from __future__ import annotations

import numpy as np


def _check(base: np.ndarray, real: np.ndarray, region: np.ndarray) -> None:
    if base.shape != real.shape:
        raise ValueError(f"base {base.shape} and real {real.shape} differ in shape")
    if region.shape != base.shape[:2]:
        raise ValueError(f"region {region.shape} does not match image {base.shape[:2]}")


def _alpha(region: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return region.astype(np.float32)[..., None]
    import cv2

    k = 2 * feather_px + 1
    a = cv2.GaussianBlur(region.astype(np.float32), (k, k), sigmaX=feather_px / 2.0)
    return np.clip(a, 0.0, 1.0)[..., None]


def revert(base: np.ndarray, real: np.ndarray, region: np.ndarray, feather_px: int = 0) -> np.ndarray:
    """R(base, region): replace `region` of `base` with real pixels."""
    _check(base, real, region)
    if feather_px <= 0:
        out = base.copy()
        out[region] = real[region]
        return out
    a = _alpha(region, feather_px)
    return np.rint(a * real + (1.0 - a) * base).astype(np.uint8)


def keep(base: np.ndarray, real: np.ndarray, region: np.ndarray, feather_px: int = 0) -> np.ndarray:
    """K(base, region): keep only `region` of `base`, everything else real."""
    _check(base, real, region)
    if feather_px <= 0:
        out = real.copy()
        out[region] = base[region]
        return out
    a = _alpha(region, feather_px)
    return np.rint(a * base + (1.0 - a) * real).astype(np.uint8)


def seam_alpha(region: np.ndarray, mode: str) -> np.ndarray:
    """Reversion weight per pixel for the seam experiment (E4). Every pixel of
    `region` is ALWAYS fully reverted (weight 1); the modes differ only outside it:
      hard        nothing else (the default operator: a sharp seam at the border)
      ramp:N      weight falls linearly from 1 at the border to 0 at N px outside
                  (soft seam)
      dilate:N    weight 1 up to N px outside, then 0 (sharp seam, region grown)
    ramp:2N and dilate:N revert about the same extra area, so their difference
    isolates the seam's sharpness from how many extra pixels were reverted."""
    region = region.astype(bool)
    if mode in ("hard", "", None):
        return region.astype(np.float32)
    kind, _, n = mode.partition(":")
    n = float(n)
    if n <= 0:
        return region.astype(np.float32)
    import cv2

    d = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)  # 0 inside the region
    if kind == "ramp":
        return np.clip(1.0 - d / n, 0.0, 1.0).astype(np.float32)
    if kind == "dilate":
        return (d <= n).astype(np.float32)
    raise ValueError(f"unknown seam mode {mode!r}")


def revert_seam(base: np.ndarray, real: np.ndarray, region: np.ndarray, mode: str) -> np.ndarray:
    _check(base, real, region)
    if mode in ("hard", "", None):
        return revert(base, real, region)
    a = seam_alpha(region, mode)[..., None]
    return np.rint(a * real + (1.0 - a) * base).astype(np.uint8)


def composite(raw: np.ndarray, real: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """x_f = M * x_raw + (1 - M) * x_r  (INP-X style exchange)."""
    return keep(raw, real, mask)


def mask_from_difference(
    real: np.ndarray,
    edited: np.ndarray,
    threshold: float = 12.0,
    dilate_px: int = 4,
    min_component: int = 64,
) -> np.ndarray:
    """Estimate the edit mask of a whole-image editor (Kontext, Qwen-Image-Edit,
    commercial APIs) or of an INP-X 'exchanged' image by thresholding the
    per-pixel difference to the real image, then cleaning it up."""
    import cv2

    if real.shape != edited.shape:
        raise ValueError("real and edited must have the same shape (align first)")
    diff = np.abs(real.astype(np.int16) - edited.astype(np.int16)).max(axis=2).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    m = (diff > threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_component]
    m = np.isin(lab, keep_ids).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m.astype(bool)
