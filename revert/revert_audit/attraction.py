"""E4-0 seam attraction (no new model queries): after an UNCITED piece B of the
edit is reverted (the matched in-edit control, revCin), B holds real pixels and
is bounded by a fresh real/forged seam. Does the detector's new citation turn
to B?

For every revCin answer (model citations, x_f) the reverted blob B is rebuilt
exactly as stage 2 drew it (same seed), and compared with null blobs N of the
same size (or smaller, down to a quarter, when less is left) drawn from the
still-forged, uncited, unreverted part of the edit (M minus C minus B) of the
SAME image:

    cover(X) = |C' & X| / |X|        C' = region cited after reverting B
    A        = cover(B) - mean cover(N)     (per answer, averaged per record)

A > 0: the freshly reverted, seam-bounded spot attracts more citation than an
equally large piece of remaining forgery (seam attraction). A < 0: citations
follow the remaining forged evidence, not the seam. Also reported: the cover of
an equally large blob outside M (real pixels never touched: the floor).

    python -m revert_audit.attraction --run runs\\pilot_sida --tag sida7b --samples data\\pilot\\samples_reviewed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .io import load_mask, load_rgb, read_jsonl
from .pipeline import _seed, answer_region, answer_verdict, qid
from .regions import blob_in_region

N_NULL = 5


def _boot(vals, n=1000, seed=0):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _cover(c2: np.ndarray, x: np.ndarray) -> float:
    return float((c2 & x).sum() / max(1, x.sum()))


def attraction(run: str, tag: str, samples_path: str, n_boot: int = 1000, check_images: bool = True,
               cond: str | None = None) -> dict:
    """cond: a seam condition of the E4a run (e.g. 'ramp24', 'dilate12'); default the pilot (hard seams)."""
    from .interventions import seam_alpha

    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    s2 = os.path.join(run, f"stage2_{tag}_model" + (f"_{cond}" if cond else ""))
    meta = [m for m in read_jsonl(os.path.join(s2, "meta.jsonl")) if m.get("base") == "fake" and m.get("cin_feasible")]
    ans = {a["query_id"]: a for a in read_jsonl(os.path.join(s2, "answers.jsonl"))}
    queries = {q["query_id"]: q for q in read_jsonl(os.path.join(s2, "queries.jsonl"))}
    recs, mismatch, checked = [], 0, 0
    for m in meta:
        sid = m["sample_id"]
        s = samples.get(sid)
        if s is None:
            continue
        M = load_mask(s["mask"])
        C = load_mask(m["cmask"])
        avail = M & ~C
        a_cf = int((C & M).sum())
        rng_in = np.random.default_rng(_seed(sid, "fake", "cin"))
        n = len(m["ctrl_overlap"])
        blobs = [blob_in_region(avail, a_cf, rng_in) for _ in range(n)]
        rng_null = np.random.default_rng(_seed(sid, "fake", "cin_null"))
        outside = ~M & ~C
        per_k, per_k_out, cited_after = [], [], []
        for k, B in enumerate(blobs):
            a = ans.get(qid(sid, 2, "fake", "revCin", k))
            if B is None or a is None:
                continue
            if check_images and checked < 40:  # the rebuilt blob must be the one that was reverted
                q = queries.get(qid(sid, 2, "fake", "revCin", k))
                base = os.path.join(run, "stage1", "images", f"{sid}__fake.png")
                if q and os.path.exists(q["path"]) and os.path.exists(base):
                    diff = (load_rgb(q["path"]) != load_rgb(base)).any(-1)
                    checked += 1
                    touched = seam_alpha(B, m.get("seam") or "hard") > 0
                    if (diff & ~touched).any():
                        mismatch += 1
            c2 = answer_region(a, *M.shape) if answer_verdict(a) == "fake" else np.zeros_like(M)
            cited_after.append(float(c2.any()))
            rest = avail & ~B  # still forged, uncited, not reverted
            a_null = min(a_cf, int(rest.sum()))
            if a_null < max(1, a_cf // 4):  # not enough remaining forgery for a comparable null
                continue
            nulls = [blob_in_region(rest, a_null, rng_null) for _ in range(N_NULL)]
            nulls = [x for x in nulls if x is not None]
            if not nulls:
                continue
            per_k.append(_cover(c2, B) - float(np.mean([_cover(c2, x) for x in nulls])))
            o = blob_in_region(outside, a_cf, rng_null)
            per_k_out.append((_cover(c2, B), float(np.mean([_cover(c2, x) for x in nulls])),
                              _cover(c2, o) if o is not None else float("nan")))
        if per_k:
            covs = np.array(per_k_out, float)
            recs.append({"sample_id": sid, "editor": s.get("editor"), "gp": m.get("gp"), "A": float(np.mean(per_k)),
                         "cover_B": float(np.nanmean(covs[:, 0])), "cover_null_in_M": float(np.nanmean(covs[:, 1])),
                         "cover_outside_M": float(np.nanmean(covs[:, 2])), "still_cites": float(np.mean(cited_after))})
    out = {"run": run, "tag": tag, "n": len(recs), "image_check": {"checked": checked, "mismatch": mismatch},
           "A": _boot([r["A"] for r in recs], n_boot),
           "cover_B": _boot([r["cover_B"] for r in recs], n_boot),
           "cover_null_in_M": _boot([r["cover_null_in_M"] for r in recs], n_boot),
           "cover_outside_M": _boot([r["cover_outside_M"] for r in recs], n_boot),
           "still_cites": _boot([r["still_cites"] for r in recs], n_boot),
           "B_minus_floor": _boot([r["cover_B"] - r["cover_outside_M"] for r in recs
                                   if r["cover_outside_M"] == r["cover_outside_M"]], n_boot),
           "by_editor": {}}
    for ed in sorted({r["editor"] for r in recs if r.get("editor")}):
        rr = [r for r in recs if r.get("editor") == ed]
        out["by_editor"][ed] = {"n": len(rr), "A": _boot([r["A"] for r in rr], n_boot),
                                "cover_B": _boot([r["cover_B"] for r in rr], n_boot),
                                "cover_null_in_M": _boot([r["cover_null_in_M"] for r in rr], n_boot)}
    lo, hi = out["A"][1], out["A"][2]
    out["verdict"] = ("seam attraction" if lo > 0 else "no seam attraction (citations follow the remaining forgery)"
                      if hi < 0 else "inconclusive")
    out["cond"] = cond or "hard"
    with open(os.path.join(run, f"attraction_{tag}" + (f"_{cond}" if cond else "") + ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def _fmt(t):
    p, lo, hi = t
    return "n/a" if p != p else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def print_report(out):
    print(f"\n=== E4-0 seam attraction: {out['tag']} [{out.get('cond', 'hard')}] (x_f, n={out['n']} records with matched in-edit controls) ===")
    ic = out["image_check"]
    print(f"rebuilt blobs checked against the stage-2 images: {ic['checked']}, mismatches {ic['mismatch']} (must be 0)")
    print(f"answers that still cite something after an uncited piece B is reverted: {_fmt(out['still_cites'])}")
    print(f"share of B cited afterwards:                          {_fmt(out['cover_B'])}")
    print(f"share of an equally large still-forged uncited piece: {_fmt(out['cover_null_in_M'])}")
    print(f"share of an equally large real piece outside M:       {_fmt(out['cover_outside_M'])}")
    print(f"A = B minus still-forged piece (paired):             {_fmt(out['A'])}  -> {out['verdict']}")
    print(f"B minus untouched real piece outside M (paired):     {_fmt(out['B_minus_floor'])}")
    for ed, e in out["by_editor"].items():
        print(f"   {ed:12s} n={e['n']:3d}  A {_fmt(e['A'])}  cover_B {_fmt(e['cover_B'])}  cover_null {_fmt(e['cover_null_in_M'])}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cond", default=None, help="seam condition of the E4a run, e.g. ramp24 or dilate12")
    a = ap.parse_args(argv)
    out = attraction(a.run, a.tag, a.samples, cond=a.cond)
    print_report(out)
    print(f"\n[attraction] saved under {a.run}")


if __name__ == "__main__":
    main()
