"""Faithfulness metrics, all defined directly on pixels and detector scores.

Per-sample record (built by analyze.py) for one (sample, base) pair:
    s0        score on the base image (x_f or x_raw), p(fake) or hard 0/1
    v0        verdict on base ('fake'/'real'/None)
    s_revC    score on R(base, C)
    s_revCt   list of scores on R(base, C~_k)  (shape-matched controls)
    s_keepC   score on K(base, C)
    s_keepCt  list of scores on K(base, C~_k)
    gp        |C ∩ M| / |C|
    per       1 if the re-explanation of R(base, C) still cites C, else 0
              (None if the detector gives no regions)
    c_area    |C| / (H*W)

Definitions (see the design doc):
    CNG = E[ s(x) - s(R(x,C)) ] - E[ s(x) - s(R(x,C~)) ] = E[ mean_k s(R(x,C~_k)) - s(R(x,C)) ]
    CSG = E[ s(K(x,C)) - mean_k s(K(x,C~_k)) ]
    GP  = E[ |C ∩ M| / |C| ]
    PER = E[ per ]
    DR  = P[ model says 'real' on x_f | model says 'fake' on x_raw and GP_raw >= 0.5 ]
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def _nanmean(xs: Sequence[float]) -> float:
    a = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(a.mean()) if a.size else float("nan")


def necessity_terms(recs: list[dict]) -> list[float]:
    return [float(np.mean(r["s_revCt"]) - r["s_revC"]) for r in recs if r.get("s_revCt")]


def sufficiency_terms(recs: list[dict]) -> list[float]:
    return [float(r["s_keepC"] - np.mean(r["s_keepCt"])) for r in recs if r.get("s_keepCt")]


def matched_terms(recs: list[dict]) -> list[float]:
    """Per record: mean score after reverting an equally large UNCITED piece of the
    edit minus the score after reverting the cited region (x_f: only its forged part changes)."""
    return [float(np.mean(r["s_revCin"]) - r["s_revC"]) for r in recs if r.get("s_revCin")]


def mng(recs: list[dict]) -> float:
    return _nanmean(matched_terms(recs))


def cng(recs: list[dict]) -> float:
    return _nanmean(necessity_terms(recs))


def csg(recs: list[dict]) -> float:
    return _nanmean(sufficiency_terms(recs))


def gp(recs: list[dict]) -> float:
    return _nanmean([r["gp"] for r in recs])


def per(recs: list[dict]) -> float:
    return _nanmean([r["per"] for r in recs if r.get("per") is not None])


def decoupling_rate(pairs: list[dict], gp_min: float = 0.5) -> tuple[float, int]:
    """pairs: per-sample dicts with keys v_raw, gp_raw, v_fake.
    Returns (DR, n_eligible)."""
    elig = [p for p in pairs if p.get("v_raw") == "fake" and (p.get("gp_raw") or 0.0) >= gp_min and p.get("v_fake") is not None]
    if not elig:
        return float("nan"), 0
    return float(np.mean([p["v_fake"] == "real" for p in elig])), len(elig)


def auc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties count 1/2)."""
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def bootstrap_ci(
    items: list,
    stat: Callable[[list], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI by resampling items (samples) with replacement."""
    point = stat(items)
    if len(items) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(items)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat([items[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
