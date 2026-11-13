"""A tiny synthetic forensic world with KNOWN ground truth, used to check
that every metric can fail for the reason it tests (before trusting any
number from a real model).

    real  x_r  : smooth random texture
    raw   x_raw: x_r + weak column stripes over the WHOLE image  (mimics the
                 global VAE trace of diffusion inpainting, cf. INP-X)
                 + a checkerboard signature inside the edit mask M
    fake  x_f  : M * x_raw + (1 - M) * x_r   (only the local signature)

Mock detectors (all respond with the standard answer schema):
    local    decides from the local checkerboard and cites where it is
             -> faithful: high CNG, high GP, low PER, low DR
    cheater  decides from the global stripes but cites the local signature
             -> plausible but unfaithful: CNG ~ 0 on x_raw, DR ~ 1
    random   decides from the local signature but cites a random box
             -> CNG ~ 0, low GP
    fixed    decides from the local signature but always cites the centre
             -> high PER when the centre is reverted (phantom evidence)
"""

from __future__ import annotations

import os

import numpy as np

from .detectors.base import Detector
from .interventions import composite
from .io import save_mask, save_rgb, write_jsonl

PATCH = 16
CB_AMP = 14.0  # local checkerboard amplitude
ST_AMP = 6.0  # global stripe amplitude


def _smooth_texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    import cv2

    base = rng.normal(0, 1, (h // 8, w // 8, 3)).astype(np.float32)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (40 + 170 * img).astype(np.float32)


def _checker(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy + xx) % 2) * 2 - 1).astype(np.float32)


def _stripes(h: int, w: int) -> np.ndarray:
    xx = np.mgrid[0:h, 0:w][1]
    return ((xx % 2) * 2 - 1).astype(np.float32)


def make_sample(rng: np.random.Generator, size: int = 128):
    h = w = size
    real = _smooth_texture(rng, h, w)
    mh, mw = int(rng.integers(24, 41)), int(rng.integers(24, 41))
    y0, x0 = int(rng.integers(0, h - mh)), int(rng.integers(0, w - mw))
    mask = np.zeros((h, w), bool)
    mask[y0 : y0 + mh, x0 : x0 + mw] = True
    raw = real + ST_AMP * _stripes(h, w)[..., None]
    raw = raw + CB_AMP * (_checker(h, w) * mask)[..., None]
    real_u8 = np.clip(np.rint(real), 0, 255).astype(np.uint8)
    raw_u8 = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
    return real_u8, raw_u8, mask


def build_synthetic_dataset(out_dir: str, n: int = 60, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        real, raw, mask = make_sample(rng)
        fake = composite(raw, real, mask)
        sid = f"syn{i:04d}"
        p = lambda k: os.path.join(out_dir, "images", f"{sid}_{k}.png")
        save_rgb(real, p("real"))
        save_rgb(raw, p("raw"))
        save_rgb(fake, p("fake"))
        save_mask(mask, os.path.join(out_dir, "images", f"{sid}_mask.png"))
        rows.append(
            {
                "id": sid,
                "real": p("real"),
                "raw": p("raw"),
                "fake": p("fake"),
                "mask": os.path.join(out_dir, "images", f"{sid}_mask.png"),
                "editor": "synthetic",
                "edit_type": "inpaint",
                "source": "synthetic",
            }
        )
    path = os.path.join(out_dir, "samples.jsonl")
    write_jsonl(rows, path)
    return path


# --------------------------------------------------------------- detectors
def _highpass(img: np.ndarray) -> np.ndarray:
    import cv2

    g = img.astype(np.float32).mean(axis=2)
    return g - cv2.blur(g, (3, 3))


def _patch_energy(img: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Mean |high-pass * pattern| response per PATCH x PATCH cell."""
    import cv2

    r = cv2.blur(_highpass(img) * pattern, (PATCH // 2, PATCH // 2))
    r = np.abs(r)
    h, w = r.shape
    gh, gw = h // PATCH, w // PATCH
    return r[: gh * PATCH, : gw * PATCH].reshape(gh, PATCH, gw, PATCH).mean(axis=(1, 3))


def _sig(x: float, thr: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(x - thr) / (thr / 4.0))))


class _MockBase(Detector):
    # Thresholds calibrated on the generator constants above (measured cell
    # energies: real ~0.04, x_f ~0.3-0.8 from checkerboard leakage, x_raw ~8
    # for the stripe response).
    cb_thr = 0.2 * CB_AMP * 0.5
    st_thr = ST_AMP / 3.0

    def _local(self, im: np.ndarray):
        e = _patch_energy(im, _checker(*im.shape[:2]))
        hot = np.argwhere(e > self.cb_thr)
        boxes = [[float(c * PATCH), float(r * PATCH), float((c + 1) * PATCH), float((r + 1) * PATCH)] for r, c in hot]
        return float(e.max()), boxes

    def _global(self, im: np.ndarray) -> float:
        return float(_patch_energy(im, _stripes(*im.shape[:2])).mean())

    @staticmethod
    def _ans(p: float, boxes: list) -> dict:
        v = "fake" if p >= 0.5 else "real"
        return {
            "verdict": v,
            "p_fake": p,
            "boxes": boxes if v == "fake" else [],
            "mask_path": None,
            "artifacts": ["checkerboard texture"] * (len(boxes) if v == "fake" else 0),
            "raw_text": "",
            "parse_mode": "native",
        }


class LocalDetector(_MockBase):
    name = "mock-local"

    def predict(self, images, explain=True):
        out = []
        for im in images:
            e, boxes = self._local(im)
            out.append(self._ans(_sig(e, self.cb_thr), boxes))
        return out


class CheaterDetector(_MockBase):
    name = "mock-cheater"

    def predict(self, images, explain=True):
        out = []
        for im in images:
            _, boxes = self._local(im)  # plausible-looking citation
            out.append(self._ans(_sig(self._global(im), self.st_thr), boxes))
        return out


class RandomCiterDetector(_MockBase):
    name = "mock-random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, images, explain=True):
        out = []
        for im in images:
            e, boxes = self._local(im)
            h, w = im.shape[:2]
            rb = []
            for _ in boxes:
                x, y = int(self.rng.integers(0, w - PATCH)), int(self.rng.integers(0, h - PATCH))
                rb.append([float(x), float(y), float(x + PATCH), float(y + PATCH)])
            out.append(self._ans(_sig(e, self.cb_thr), rb))
        return out


class FixedCiterDetector(_MockBase):
    name = "mock-fixed"

    def predict(self, images, explain=True):
        out = []
        for im in images:
            e, _ = self._local(im)
            h, w = im.shape[:2]
            box = [[w / 2 - 16.0, h / 2 - 16.0, w / 2 + 16.0, h / 2 + 16.0]]
            out.append(self._ans(_sig(e, self.cb_thr), box))
        return out


MOCKS = {
    "mock-local": LocalDetector,
    "mock-cheater": CheaterDetector,
    "mock-random": RandomCiterDetector,
    "mock-fixed": FixedCiterDetector,
}


def build_synthetic_pairs(out_dir: str, n: int = 40, seed: int = 0, weak: float = 0.15) -> str:
    """Two-object synthetic samples for the E4b test: object 1 carries the full
    local signature, object 2 a weak one (amplitude `weak` x), so the faithful
    mock cites object 1 and decides on it."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        h = w = 128
        real = _smooth_texture(rng, h, w)
        m1 = np.zeros((h, w), bool)
        m2 = np.zeros((h, w), bool)
        m1[16:48, 16:48] = True
        m2[80:112, 80:112] = True
        if rng.random() < 0.5:
            m1, m2 = m2, m1
        raw = real + CB_AMP * (_checker(h, w) * m1)[..., None] + weak * CB_AMP * (_checker(h, w) * m2)[..., None]
        real_u8 = np.clip(np.rint(real), 0, 255).astype(np.uint8)
        raw_u8 = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
        fake = composite(raw_u8, real_u8, m1 | m2)
        sid = f"pair{i:04d}"
        p = lambda k: os.path.join(out_dir, "images", f"{sid}_{k}.png")
        save_rgb(real_u8, p("real"))
        save_rgb(fake, p("fake"))
        save_mask(m1 | m2, p("mask"))
        save_mask(m1, p("mask1"))
        save_mask(m2, p("mask2"))
        rows.append({"id": sid, "real": p("real"), "fake": p("fake"), "mask": p("mask"), "editor": "synthetic",
                     "edit_type": "pair", "pair": True, "source": "synthetic",
                     "parts": [{"mask": p("mask1"), "obj": "strong"}, {"mask": p("mask2"), "obj": "weak"}]})
    path = os.path.join(out_dir, "samples.jsonl")
    write_jsonl(rows, path)
    return path
