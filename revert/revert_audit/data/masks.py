"""Edit masks for real photos that have no object annotations (your own
photos, high-resolution sets): free-form brush strokes, the standard mask
family of inpainting research (LaMa / DeepFill style)."""

from __future__ import annotations

import numpy as np


def freeform_mask(h: int, w: int, rng: np.random.Generator, min_frac: float = 0.03, max_frac: float = 0.25,
                  max_tries: int = 50) -> np.ndarray:
    """Random connected brush strokes covering min_frac..max_frac of the image."""
    import cv2

    best = None
    target = rng.uniform(min_frac, max_frac)
    for _ in range(max_tries):
        m = np.zeros((h, w), np.uint8)
        cx, cy = rng.integers(w // 6, 5 * w // 6), rng.integers(h // 6, 5 * h // 6)
        width = int(rng.integers(max(8, min(h, w) // 25), max(12, min(h, w) // 8)))
        for _ in range(int(rng.integers(3, 9))):
            ang = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(0.05, 0.25) * max(h, w)
            nx = int(np.clip(cx + length * np.cos(ang), 0, w - 1))
            ny = int(np.clip(cy + length * np.sin(ang), 0, h - 1))
            cv2.line(m, (int(cx), int(cy)), (nx, ny), 1, width)
            cv2.circle(m, (nx, ny), width // 2, 1, -1)
            cx, cy = nx, ny
            if m.mean() >= target:
                break
        frac = m.mean()
        if min_frac <= frac <= max_frac:
            return m.astype(bool)
        if best is None or abs(frac - target) < abs(best.mean() - target):
            best = m
    return best.astype(bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return mask
    k = 2 * px + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
