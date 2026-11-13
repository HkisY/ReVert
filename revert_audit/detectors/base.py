"""Detector interface.

A detector maps a batch of RGB images to a list of answers:
    {
      "verdict":    "fake" | "real" | None,
      "p_fake":     float in [0, 1] or None (None for API / hard-label models),
      "boxes":      [[x1, y1, x2, y2], ...] in ORIGINAL image pixel coordinates,
      "mask_path":  optional path to a predicted forgery mask (SIDA, FakeShield),
      "artifacts":  [str, ...] textual claims,
      "raw_text":   full model output,
      "parse_mode": "json" | "regex" | "failed" | "native",
    }

SIDA runs in its own env through revert_audit.detectors.sida (a worker process).
Other detectors that live in their own repos/environments (FakeShield,
Veritas, FakeVLM, AIGI-Holmes) are NOT wrapped here: run their official
inference on the query images and convert their outputs to this schema with
`revert_audit.adapters` (see README). The audit only needs the answers file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, images: list[np.ndarray], explain: bool = True) -> list[dict]:
        """explain=False may skip text generation and return only p_fake
        (used for control interventions, whose explanations are not needed)."""
        ...


def answer_score(ans: dict) -> float:
    """Scalar 'fake' score used by all metrics: p_fake if available, else the
    hard verdict (1.0 fake / 0.0 real / 0.5 unparsable)."""
    p = ans.get("p_fake")
    if p is not None:
        return float(p)
    v = ans.get("verdict")
    if v == "fake":
        return 1.0
    if v == "real":
        return 0.0
    return 0.5


def answer_verdict(ans: dict) -> str | None:
    """Effective decision. When a probability is available we threshold it,
    so that the verdict and the score used for CNG/CSG are the same decision."""
    if ans.get("verdict_native") and ans.get("verdict") in ("fake", "real"):
        return ans["verdict"]  # multi-class detectors (SIDA): their own argmax decision
    p = ans.get("p_fake")
    if p is not None:
        return "fake" if float(p) >= 0.5 else "real"
    return ans.get("verdict")
