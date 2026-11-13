"""File I/O helpers. All images are stored as lossless PNG on purpose:
JPEG re-encoding would add compression artifacts to every intervened image
and confound the audit."""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    """Load an image as HxWx3 uint8."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_rgb(arr: np.ndarray, path: str) -> None:
    if not path.lower().endswith(".png"):
        raise ValueError(f"Intervened images must be PNG (lossless), got {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), mode="RGB").save(path)


def load_mask(path: str, threshold: int = 127) -> np.ndarray:
    """Load a binary mask (white = edited) as HxW bool."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > threshold


def save_mask(mask: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(row: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
