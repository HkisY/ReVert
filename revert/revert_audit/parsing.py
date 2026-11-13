"""Parse free-form detector outputs into {verdict, boxes, artifacts}.

Detectors are asked for JSON, but MLLMs often wrap it in prose or code
fences, or break it. We try strict JSON first, then fall back to regexes.
Every parse records how it succeeded (`parse_mode`) so that parse failures
can be reported in the paper instead of silently dropped."""

from __future__ import annotations

import json
import re
from typing import Any

FAKE_WORDS = ("fake", "ai-generated", "ai generated", "generated", "edited", "manipulated", "tampered", "synthetic", "forged")
REAL_WORDS = ("real", "authentic", "genuine", "pristine", "not manipulated", "not edited")

_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def normalize_verdict(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    # check negated "real" phrases before generic fake words
    for w in ("not real", "not authentic", "not genuine"):
        if w in s:
            return "fake"
    for w in ("not manipulated", "not edited", "not fake", "not ai"):
        if w in s:
            return "real"
    if any(s.startswith(w) or s == w for w in FAKE_WORDS):
        return "fake"
    if any(s.startswith(w) or s == w for w in REAL_WORDS):
        return "real"
    if any(w in s for w in FAKE_WORDS):
        return "fake"
    if any(w in s for w in REAL_WORDS):
        return "real"
    return None


def _first_json_object(text: str) -> dict | None:
    """Return the first balanced {...} block that parses as JSON."""
    text = text.replace("```json", "```")
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _boxes_from_evidence(ev: Any) -> tuple[list[list[float]], list[str]]:
    boxes, arts = [], []
    if not isinstance(ev, list):
        return boxes, arts
    for item in ev:
        if isinstance(item, dict):
            b = item.get("region") or item.get("bbox") or item.get("box") or item.get("bbox_2d")
            a = item.get("artifact") or item.get("description") or item.get("label") or ""
        elif isinstance(item, (list, tuple)):
            b, a = item, ""
        else:
            continue
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                boxes.append([float(x) for x in b])
                arts.append(str(a))
            except (TypeError, ValueError):
                continue
        elif isinstance(a, str) and a:
            arts.append(a)  # textual claim without a region (global claim)
    return boxes, arts


def parse_response(text: str) -> dict:
    """Return {verdict, boxes, artifacts, explanation, parse_mode}.

    `boxes` are in whatever coordinate frame the model used; the detector
    wrapper is responsible for mapping them to original-image pixels.
    """
    obj = _first_json_object(text or "")
    if obj is not None:
        verdict = normalize_verdict(obj.get("verdict") or obj.get("label") or obj.get("answer"))
        boxes, arts = _boxes_from_evidence(obj.get("evidence") or obj.get("regions") or [])
        return {
            "verdict": verdict,
            "boxes": boxes,
            "artifacts": arts,
            "explanation": str(obj.get("explanation", "")),
            "parse_mode": "json",
        }
    # Fallback: regex
    t = (text or "").strip()
    m = re.search(r"verdict\W+(\w[\w\- ]*)", t, flags=re.I)
    verdict = normalize_verdict(m.group(1)) if m else normalize_verdict(t[:80])
    boxes = [[float(g) for g in mm.groups()] for mm in _BOX_RE.finditer(t)]
    return {
        "verdict": verdict,
        "boxes": boxes,
        "artifacts": [],
        "explanation": t,
        "parse_mode": "regex" if (verdict or boxes) else "failed",
    }
