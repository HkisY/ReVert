"""E4b: seam-free test on two-object forgeries (whole-component reversion).

Each two-object sample (data.build --pairs) has two separately edited objects
M1, M2 (>= 24 px apart). The detector's citation C on x_f decides which object
it points at: Mc = the object holding >= cite_min of C, Mo = the other one.
Eligible: >= cite_min of C on one object and < other_max on the other (records
citing both objects are counted and reported, not analysed).
Two interventions, each reverting ONE WHOLE object, so neither leaves a new
real/forged boundary nor a forged rim (the result is exactly a one-object
forgery of the other object):

    Rc = R(x_f, Mc)     the cited object restored
    Ro = R(x_f, Mo)     the uncited object restored

Per eligible record
    cPER    after Rc: still 'fake' and C' covers >= half of C&Mc   (seam-free phantom evidence:
            it still points at the part of the object it cited, which is now entirely real)
    cSTAY   after Ro: still 'fake' and C' covers >= half of C&Mc   (its comparison)
    switch  after Rc: still 'fake' and >= half of C' on Mo     (follows the remaining evidence)
    cMNG    s(Ro) - s(Rc)                                    (cited object matters more than the uncited one)

    python -m revert_audit.cli component --samples data\\pairs\\samples.jsonl --run runs\\e4b_sida --tag sida7b \\
        --ref-run runs\\pilot_sida <detector args>
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import pipeline
from .interventions import revert
from .io import load_mask, load_rgb, read_jsonl, save_rgb, write_jsonl
from .pipeline import _explained, answer_region, answer_score, answer_verdict, qid

KINDS = ("comp1", "comp2")


def _boot(vals, n=1000, seed=0):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pair_samples(samples_path: str, out_path: str, limit: int | None = None) -> str:
    rows = [s for s in read_jsonl(samples_path) if s.get("pair") and s.get("parts")]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{samples_path}: no two-object samples (build them with: python -m revert_audit.data.build --pairs ...)")
    write_jsonl(rows, out_path)
    return out_path


COV_CITED, COV_OTHER = 0.3, 0.1     # PRIMARY eligibility rule (coverage rule)
SHARE_CITED, SHARE_OTHER = 0.5, 0.25  # original rule, kept as the secondary analysis


def eligibility(C, parts):
    """Which object the citation points at, under both rules.
    coverage rule (primary): >= 30% of one object is cited and < 10% of the other;
    share rule (secondary): >= 50% of the citation lies on one object and < 25% on the other."""
    share = [float((C & m).sum() / C.sum()) for m in parts]
    cov = [float((C & m).sum() / max(1, m.sum())) for m in parts]
    cc, cs = int(np.argmax(cov)), int(np.argmax(share))
    return {"share": share, "cov": cov,
            "cov_ok": bool(cov[cc] >= COV_CITED and cov[1 - cc] < COV_OTHER), "cov_cited": cc,
            "share_ok": bool(share[cs] >= SHARE_CITED and share[1 - cs] < SHARE_OTHER), "share_cited": cs,
            "cites_both": bool(min(cov) >= COV_CITED), "cites_neither": bool(max(cov) < COV_OTHER)}


def build_queries(samples_path, s1_dir, s1_answers, out_dir) -> str:
    from .io import save_mask

    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    a1 = {a["query_id"]: a for a in read_jsonl(s1_answers)}
    qrows, meta = [], []
    for sid, s in samples.items():
        row = {"sample_id": sid, "skipped": None}
        a = a1.get(qid(sid, 1, "fake"))
        if a is None or answer_verdict(a) != "fake":
            row["skipped"] = "verdict_not_fake" if a else "no_stage1_answer"
            meta.append(row)
            continue
        parts = [load_mask(p["mask"]) for p in s["parts"]]
        C = answer_region(a, *parts[0].shape)
        if not C.any():
            row["skipped"] = "no_region"
            meta.append(row)
            continue
        row.update(eligibility(C, parts))
        if not (row["cov_ok"] or row["share_ok"]):
            row["skipped"] = ("cites_both_objects" if row["cites_both"] else
                              "cites_neither_object" if row["cites_neither"] else "citation_not_on_one_object")
            meta.append(row)
            continue
        real = load_rgb(s["real"])
        base = load_rgb(os.path.join(s1_dir, "images", f"{sid}__fake.png"))
        cpath = os.path.join(out_dir, "regions", f"{sid}__C.png")
        save_mask(C, cpath)
        row["cmask"] = cpath
        for k, m in enumerate(parts, 1):  # restore object k entirely
            p = os.path.join(out_dir, "images", f"{sid}__comp{k}.png")
            if not os.path.exists(p):
                save_rgb(revert(base, real, m), p)
            qrows.append({"query_id": qid(sid, 2, "fake", f"comp{k}", 0), "sample_id": sid, "stage": 2, "base": "fake",
                          "kind": f"comp{k}", "k": 0, "path": p})
        meta.append(row)
    write_jsonl(meta, os.path.join(out_dir, "meta.jsonl"))
    out = os.path.join(out_dir, "queries.jsonl")
    write_jsonl(qrows, out)
    return out


def _summary(recs, n_boot):
    return {"n": len(recs),
            "cPER": _boot([r["cPER"] for r in recs], n_boot),
            "cSTAY": _boot([r["cSTAY"] for r in recs], n_boot),
            "STAY_minus_cPER": _boot([r["cSTAY"] - r["cPER"] for r in recs
                                      if r["cSTAY"] is not None and r["cPER"] is not None], n_boot),
            "switch": _boot([r["switch"] for r in recs], n_boot),
            "cMNG": _boot([r["s_o"] - r["s_c"] for r in recs], n_boot),
            "nec_cited": _boot([r["s0"] - r["s_c"] for r in recs], n_boot),
            "nec_uncited": _boot([r["s0"] - r["s_o"] for r in recs], n_boot),
            "fake_after_cited_reverted": _boot([r["fake_after_c"] for r in recs], n_boot)}


def _verdicts(res, ref):
    lo = res["cPER"][1]
    res["verdict_phantom"] = ("n/a (no reference oracle PER)" if not ref else
                              "seam-free phantom evidence" if lo > ref[2] else "not shown")
    m_lo, m_hi = res["cMNG"][1], res["cMNG"][2]
    res["verdict_cMNG"] = ("cited object drives the decision" if m_lo > 0.05 else
                           "cited object no more decisive than the uncited one" if m_hi < 0.05 else "inconclusive")
    return res


def analyze(samples_path, s1_answers, out_dir, n_boot=1000, ref_run=None, tag=None) -> dict:
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    a1 = {a["query_id"]: a for a in read_jsonl(s1_answers)}
    a2 = {a["query_id"]: a for a in read_jsonl(os.path.join(out_dir, "answers.jsonl"))}
    meta = read_jsonl(os.path.join(out_dir, "meta.jsonl"))
    recs = {"coverage": [], "share": []}
    for m in meta:
        if m.get("skipped"):
            continue
        sid = m["sample_id"]
        a0 = a1.get(qid(sid, 1, "fake"))
        ans = [a2.get(qid(sid, 2, "fake", f"comp{k}", 0)) for k in (1, 2)]
        if a0 is None or any(x is None for x in ans):
            continue
        C = load_mask(m["cmask"])
        parts = [load_mask(p["mask"]) for p in samples[sid]["parts"]]
        h, w = C.shape
        for rule, ok, c in (("coverage", m["cov_ok"], m["cov_cited"]), ("share", m["share_ok"], m["share_cited"])):
            if not ok:
                continue
            o = 1 - c
            rc, ro = ans[c], ans[o]
            target = C & parts[c]  # the part of the cited object that the detector pointed at

            def still_cites(x):
                if not _explained(x):
                    return None
                c2 = answer_region(x, h, w)
                return float(answer_verdict(x) == "fake" and (c2 & target).sum() >= 0.5 * target.sum())

            sw = None
            if _explained(rc):
                c2 = answer_region(rc, h, w)
                sw = float(answer_verdict(rc) == "fake" and c2.any() and (c2 & parts[o]).sum() / c2.sum() >= 0.5)
            recs[rule].append({"sample_id": sid, "editor": samples[sid].get("editor"), "rule": rule,
                               "cov": m["cov"], "share": m["share"],
                               "area_ratio": float(parts[c].sum() / max(1, parts[o].sum())),
                               "s0": answer_score(a0), "s_c": answer_score(rc), "s_o": answer_score(ro),
                               "cPER": still_cites(rc), "cSTAY": still_cites(ro), "switch": sw,
                               "fake_after_c": float(answer_verdict(rc) == "fake"),
                               "fake_after_o": float(answer_verdict(ro) == "fake")})
    write_jsonl(recs["coverage"] + recs["share"], os.path.join(out_dir, "records.jsonl"))
    counts = {}
    for m in meta:
        k = m.get("skipped") or ("eligible_coverage" if m.get("cov_ok") else "eligible_share_only")
        counts[k] = counts.get(k, 0) + 1
    ref = None
    if ref_run and tag:
        mp = os.path.join(ref_run, "results", f"{tag}_oracle", "metrics.json")
        if os.path.exists(mp):
            ref = json.load(open(mp, encoding="utf-8")).get("by_base", {}).get("fake", {}).get("PER")
    res = _verdicts(_summary(recs["coverage"], n_boot), ref)
    res.update({"counts": counts, "oracle_PER_ref": ref, "rule": "coverage (primary)",
                "share_rule": _verdicts(_summary(recs["share"], n_boot), ref), "by_editor": {}})
    for ed in sorted({r["editor"] for r in recs["coverage"] if r.get("editor")}):
        rr = [r for r in recs["coverage"] if r.get("editor") == ed]
        res["by_editor"][ed] = {"n": len(rr), "cPER": _boot([r["cPER"] for r in rr], n_boot),
                                "STAY_minus_cPER": _boot([r["cSTAY"] - r["cPER"] for r in rr if r["cSTAY"] is not None
                                                          and r["cPER"] is not None], n_boot),
                                "cMNG": _boot([r["s_o"] - r["s_c"] for r in rr], n_boot)}
    with open(os.path.join(out_dir, "component.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def run_component(samples, run, detector, tag, n_boot=1000, batch_size=4, limit=None, ref_run=None, log_every=50) -> dict:
    os.makedirs(run, exist_ok=True)
    sp = pair_samples(samples, os.path.join(run, "pair_samples.jsonl"), limit)
    s1 = os.path.join(run, "stage1")
    q1 = pipeline.stage1(sp, s1, bases=("fake", "real"))
    a1 = os.path.join(s1, f"answers_{tag}.jsonl")
    pipeline.run_detector(q1, detector, a1, batch_size, log_every)
    s1sum = pipeline.stage1_summary(sp, a1)
    out = os.path.join(run, f"component_{tag}")
    q2 = build_queries(sp, s1, a1, out)
    pipeline.run_detector(q2, detector, os.path.join(out, "answers.jsonl"), batch_size, log_every, explain_kinds=KINDS)
    res = analyze(sp, a1, out, n_boot, ref_run, tag)
    res["stage1"] = {k: s1sum.get(k) for k in ("n_fake", "acc_fake", "n_real", "acc_real", "auc_fake_vs_real")}
    with open(os.path.join(out, "component.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def _fmt(t):
    p, lo, hi = t
    return "n/a" if p != p else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def print_report(res, tag=""):
    print(f"\n=== E4b seam-free component test {tag} (two-object x_f) ===")
    if res.get("stage1"):
        print(f"stage 1: {res['stage1']}")
    print(f"records: {res['counts']}")
    for name, r in (("PRIMARY coverage rule (>=30% of one object cited, <10% of the other)", res),
                    ("secondary share rule (>=50% of the citation on one object, <25% on the other)", res["share_rule"])):
        print(f"--- {name}: n={r['n']}")
        print(f"  cPER  (cited object restored, still cites it)      {_fmt(r['cPER'])}")
        print(f"  cSTAY (uncited object restored, still cites it)    {_fmt(r['cSTAY'])}")
        print(f"  paired cSTAY - cPER                                {_fmt(r['STAY_minus_cPER'])}")
        print(f"  switch (cited restored -> now cites the other)     {_fmt(r['switch'])}")
        print(f"  still 'fake' after the cited object is restored    {_fmt(r['fake_after_cited_reverted'])}")
        print(f"  necessity: cited {_fmt(r['nec_cited'])}  uncited {_fmt(r['nec_uncited'])}")
        print(f"  cMNG = s(uncited restored) - s(cited restored)    {_fmt(r['cMNG'])}  -> {r['verdict_cMNG']}")
        ref = res.get("oracle_PER_ref")
        print(f"  vs single-object oracle PER {_fmt(ref) if ref else 'n/a'}  -> {r['verdict_phantom']}")
    for ed, e in res.get("by_editor", {}).items():
        print(f"   {ed:12s} n={e['n']:3d}  cPER {_fmt(e['cPER'])}  cSTAY-cPER {_fmt(e['STAY_minus_cPER'])}  cMNG {_fmt(e['cMNG'])}")


def eligibility_summary(samples_path, s1_answers, thresholds=((0.5, 0.25), (0.4, 0.2))) -> dict:
    """Where do the detector's citations fall on two-object x_f (no new queries)?
    share_k = |C & M_k| / |C| (how much of the citation is on object k) and
    cov_k = |C & M_k| / |M_k| (how much of object k is cited). Counts how many
    flagged records an alternative eligibility rule would keep: by share
    (cited >= a, other < b) and by coverage (cited object >= 0.3 covered, other < 0.1)."""
    samples = {s["id"]: s for s in read_jsonl(samples_path) if s.get("pair")}
    a1 = {a["query_id"]: a for a in read_jsonl(s1_answers)}
    rows = []
    for sid, s in samples.items():
        a = a1.get(qid(sid, 1, "fake"))
        if a is None or answer_verdict(a) != "fake":
            continue
        parts = [load_mask(p["mask"]) for p in s["parts"]]
        C = answer_region(a, *parts[0].shape)
        if not C.any():
            rows.append({"empty": True})
            continue
        share = [float((C & m).sum() / C.sum()) for m in parts]
        cov = [float((C & m).sum() / max(1, m.sum())) for m in parts]
        rows.append({"empty": False, "share": share, "cov": cov, "on_edit": sum(share),
                     "c_frac": float(C.mean())})
    ok = [r for r in rows if not r["empty"]]
    out = {"flagged": len(rows), "empty_citation": len(rows) - len(ok)}
    if ok:
        out["median_share_on_edit"] = float(np.median([r["on_edit"] for r in ok]))
        out["median_citation_area"] = float(np.median([r["c_frac"] for r in ok]))
        out["share_rules"] = {f"cited>={a},other<{b}": sum(max(r["share"]) >= a and min(r["share"]) < b for r in ok)
                              for a, b in thresholds}
        out["coverage_rule cited>=0.3,other<0.1"] = sum(max(r["cov"]) >= 0.3 and min(r["cov"]) < 0.1 for r in ok)
        out["coverage_rule cited>=0.5,other<0.1"] = sum(max(r["cov"]) >= 0.5 and min(r["cov"]) < 0.1 for r in ok)
        out["cites_neither (both cov < 0.1)"] = sum(max(r["cov"]) < 0.1 for r in ok)
        out["cites_both (both cov >= 0.3)"] = sum(min(r["cov"]) >= 0.3 for r in ok)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="E4b eligibility summary from stage-1 answers (no model calls)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = eligibility_summary(os.path.join(a.run, "pair_samples.jsonl"),
                              os.path.join(a.run, "stage1", f"answers_{a.tag}.jsonl"))
    print(json.dumps(res, indent=2, ensure_ascii=False))
