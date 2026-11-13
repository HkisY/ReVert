"""Detector registry. Heavy dependencies (torch/transformers) are imported
lazily so that the audit logic and tests run on a CPU-only machine."""

from __future__ import annotations

from .base import Detector, answer_score, answer_verdict


def build_detector(name: str, **kw) -> Detector:
    name = name.lower()
    if name.startswith("mock-"):
        from ..synthetic import MOCKS

        return MOCKS[name]()
    if name in ("qwen", "qwen-vl", "qwen25vl", "qwen3vl"):
        from .qwen_vl import QwenVLDetector

        return QwenVLDetector(**kw)
    if name in ("sida", "sida-7b"):
        from .sida import SidaDetector

        return SidaDetector(**kw)
    if name in ("sofake", "so-fake-r1", "sofake-r1"):
        from .sofake import SoFakeDetector

        return SoFakeDetector(**kw)
    if name in ("fakeshield", "fakeshield-v1-22b"):
        from .fakeshield import FakeShieldDetector

        return FakeShieldDetector(**kw)
    if name == "clip-probe":
        from .clip_probe import ClipProbeDetector

        return ClipProbeDetector(**kw)
    raise ValueError(
        f"Unknown detector '{name}'. Built-in: qwen-vl, sida, clip-probe, mock-*. "
        "For SIDA / FakeShield / Veritas / FakeVLM / AIGI-Holmes run their official "
        "inference on the query images and convert with `revert_audit.adapters`."
    )


__all__ = ["Detector", "answer_score", "answer_verdict", "build_detector"]
