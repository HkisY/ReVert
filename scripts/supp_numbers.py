"""Numbers requested for the final paper (code-release brief, section 10). Reads existing outputs only:
no GPU, no model, no image is needed, and nothing in runs/ is changed.

    python scripts/supp_numbers.py          # prints everything and writes supp_numbers.json

  1  intervals re-estimated with a bootstrap clustered on source photographs (old interval next to it)
  2  matched baseline for component reversion: the original x_r judged fake and still pointed at like cPER
  3  phantom-evidence cases: how often the original photograph x_r is judged real
  4  exact eligibility rules of component reversion (coverage rule, share rule)
  5  qualifying counts of the Fig. F1 rows and the extra conditions of the selection script
  6  false-positive rate on the original photographs (single-object and two-object sets)
  7  cPER split by how well the citation is localized on x_f (GP >= 0.5 vs < 0.5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from revert_audit.attraction import N_NULL, _cover  # noqa: E402
from revert_audit.component import COV_CITED, COV_OTHER, SHARE_CITED, SHARE_OTHER  # noqa: E402
from revert_audit.diagnose import load_explainer  # noqa: E402
from revert_audit.hypotheses import cluster_boot  # noqa: E402
from revert_audit.io import load_mask, read_jsonl  # noqa: E402
from revert_audit.pipeline import _explained, _seed, answer_region, answer_verdict, dr_pairs, qid  # noqa: E402
from revert_audit.regions import blob_in_region, overlap_frac  # noqa: E402

N_BOOT, SEED = 1000, 0
OUT = {}


def old_boot(vals, n=N_BOOT, seed=SEED):
    """The record-level bootstrap used so far (diagnose/component/attraction _boot)."""
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def mean_of(key):
    def f(rs):
        v = [r[key] for r in rs if r.get(key) is not None and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")
    return f


def fmt(t):
    return "  n/a" if t is None or t[0] != t[0] else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def row(label, recs, key, store):
    """recs carry 'photo' and the value under key; prints old (record) vs new (photo-clustered) interval."""
    vals = [r for r in recs if r.get(key) is not None and r[key] == r[key]]
    old = old_boot([r[key] for r in vals])
    new = cluster_boot(vals, mean_of(key), N_BOOT, SEED)
    n_ph = len({r["photo"] for r in vals})
    print(f"   {label:<34} n={len(vals):4d} photos={n_ph:4d}   record: {fmt(old)}   photo-clustered: {fmt(new)}")
    store[label] = {"n": len(vals), "photos": n_ph, "record_bootstrap": old, "photo_clustered": new}


# ------------------------------------------------------------------ 1a single-object set
def ba_records(run, tag, samples):
    """Per-record boundary attraction, rebuilt exactly as revert_audit.attraction does (same seeds)."""
    s2 = os.path.join(run, f"stage2_{tag}_model")
    meta = [m for m in read_jsonl(os.path.join(s2, "meta.jsonl")) if m.get("base") == "fake" and m.get("cin_feasible")]
    ans = {a["query_id"]: a for a in read_jsonl(os.path.join(s2, "answers.jsonl"))}
    recs = []
    for m in meta:
        sid = m["sample_id"]
        s = samples.get(sid)
        if s is None:
            continue
        M, C = load_mask(s["mask"]), load_mask(m["cmask"])
        avail = M & ~C
        a_cf = int((C & M).sum())
        rng_in = np.random.default_rng(_seed(sid, "fake", "cin"))
        blobs = [blob_in_region(avail, a_cf, rng_in) for _ in range(len(m["ctrl_overlap"]))]
        rng_null = np.random.default_rng(_seed(sid, "fake", "cin_null"))
        outside = ~M & ~C
        per_k, covs = [], []
        for k, B in enumerate(blobs):
            a = ans.get(qid(sid, 2, "fake", "revCin", k))
            if B is None or a is None:
                continue
            c2 = answer_region(a, *M.shape) if answer_verdict(a) == "fake" else np.zeros_like(M)
            rest = avail & ~B
            a_null = min(a_cf, int(rest.sum()))
            if a_null < max(1, a_cf // 4):
                continue
            nulls = [x for x in (blob_in_region(rest, a_null, rng_null) for _ in range(N_NULL)) if x is not None]
            if not nulls:
                continue
            per_k.append(_cover(c2, B) - float(np.mean([_cover(c2, x) for x in nulls])))
            o = blob_in_region(outside, a_cf, rng_null)
            covs.append((_cover(c2, B), float(np.mean([_cover(c2, x) for x in nulls])),
                         _cover(c2, o) if o is not None else float("nan")))
        if per_k:
            c = np.array(covs, float)
            recs.append({"sample_id": sid, "BA": float(np.mean(per_k)), "cover_B": float(np.nanmean(c[:, 0])),
                         "cover_N": float(np.nanmean(c[:, 1])), "cover_out": float(np.nanmean(c[:, 2]))})
    return recs


def single_set(name, run, tag, samples_path):
    print(f"\n== 1  {name}: single-object set ({run})")
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    photo = lambda sid: samples.get(sid, {}).get("coco_image_id") or sid  # noqa: E731
    R = {ex: [r for r in load_explainer(run, tag, ex, samples) if r["base"] == "fake"] for ex in ("model", "oracle", "random")}
    for rs in R.values():
        for r in rs:
            r["photo"] = photo(r["sample_id"])
    st = OUT.setdefault(name, {}).setdefault("single_object", {})
    for ex, lab in (("model", "CNG: cited region"), ("oracle", "CNG: true edit"), ("random", "CNG: random region")):
        row(lab, R[ex], "nec", st)
    m = R["model"]
    for r in m:
        r["raises"] = float(r["s_revC"] > r["s0"] + 0.05)
        r["mng"] = float(np.mean(r["s_revCin"]) - r["s_revC"]) if r.get("s_revCin") else None
        r["strict_per"] = r["per"] if (r.get("cov") is not None and r["cov"] >= 0.9) else None
        r["stay_minus_per"] = (r["stay_cin"] - r["per"]) if (r["gp"] >= 0.5 and r.get("per") is not None
                                                             and r.get("stay_cin") is not None) else None
    row("revert raises the score", m, "raises", st)
    row("PER", m, "per", st)
    row("PER strict (coverage >= 0.9)", m, "strict_per", st)
    row("PER_M (true-edit reference)", R["oracle"], "per", st)
    row("MNG", m, "mng", st)
    row("STAY - PER (GP >= 0.5)", m, "stay_minus_per", st)
    a1 = os.path.join(run, "stage1", f"answers_{tag}.jsonl")
    dr = [{"photo": photo(p["sample_id"]), "dr": float(p["v_fake"] == "real")}
          for p in dr_pairs(samples_path, a1) if p["v_raw"] == "fake" and p["gp_raw"] >= 0.5 and p["v_fake"] is not None]
    row("DR", dr, "dr", st)
    ba = ba_records(run, tag, samples)
    for r in ba:
        r["photo"] = photo(r["sample_id"])
    for k, lab in (("BA", "BA"), ("cover_B", "BA: restored region B cited"), ("cover_N", "BA: equal forged region cited"),
                   ("cover_out", "BA: untouched authentic region cited")):
        row(lab, ba, k, st)
    # per inpainting model: does any photo appear twice within one model? (if not, clustering changes nothing)
    dup = {}
    for ed in sorted({samples[r["sample_id"]].get("editor") for r in m}):
        c = Counter(r["photo"] for r in m if samples[r["sample_id"]].get("editor") == ed)
        dup[ed] = max(c.values()) if c else 0
    verdict = ("clustering leaves these intervals unchanged" if dup and max(dup.values()) <= 1
               else "RECOMPUTE per model" if dup else "no records")
    print(f"   per-model tables (Fig. 3, Tab. E2): max records per photo within one model = {dup} -> {verdict}")
    st["max_records_per_photo_within_model"] = dup


# ------------------------------------------------------------------ component reversion
def component_set(name, run, tag, single_run):
    print(f"\n== {name}: two-object set, component reversion ({run})")
    comp = os.path.join(run, f"component_{tag}")
    samples = {s["id"]: s for s in read_jsonl(os.path.join(run, "pair_samples.jsonl"))}
    photo = lambda sid: samples.get(sid, {}).get("coco_image_id") or sid  # noqa: E731
    recs = read_jsonl(os.path.join(comp, "records.jsonl"))
    meta = {m["sample_id"]: m for m in read_jsonl(os.path.join(comp, "meta.jsonl"))}
    a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
    st = OUT.setdefault(name, {})
    for rule in ("coverage", "share"):
        rr = [dict(r, photo=photo(r["sample_id"])) for r in recs if r["rule"] == rule]
        for r in rr:
            r["stay_minus"] = (r["cSTAY"] - r["cPER"]) if (r.get("cSTAY") is not None and r.get("cPER") is not None) else None
            r["cMNG"] = r["s_o"] - r["s_c"]
            r["nec_c"], r["nec_o"] = r["s0"] - r["s_c"], r["s0"] - r["s_o"]
        print(f"   -- {rule} rule")
        s = st.setdefault(f"component_{rule}", {})
        for k, lab in (("cPER", "cPER"), ("cSTAY", "cSTAY"), ("stay_minus", "cSTAY - cPER"), ("switch", "switch"),
                       ("cMNG", "cMNG"), ("nec_c", "score drop, cited object"), ("nec_o", "score drop, uncited object"),
                       ("fake_after_c", "still fake after cited restored")):
            row(lab, rr, k, s)
        if rule != "coverage":
            continue
        # 2 matched baseline on x_r, 3 phantom cases, 7 cPER by localization
        base, phantom_real, by_gp = [], [], []
        for r in rr:
            sid, m = r["sample_id"], meta[r["sample_id"]]
            parts = [load_mask(p["mask"]) for p in samples[sid]["parts"]]
            C = load_mask(m["cmask"])
            c = m["cov_cited"]
            target = C & parts[c]
            ar = a1.get(qid(sid, 1, "real"))
            if ar is not None:
                v = None
                if _explained(ar):
                    c2 = answer_region(ar, *C.shape)
                    v = float(answer_verdict(ar) == "fake" and (c2 & target).sum() >= 0.5 * target.sum())
                base.append({"photo": r["photo"], "x_r_still_cited": v, "x_r_fake": float(answer_verdict(ar) == "fake")})
                if r.get("cPER") == 1.0:
                    phantom_real.append({"photo": r["photo"], "x_r_real": float(answer_verdict(ar) == "real")})
            U = parts[0] | parts[1]
            by_gp.append({"photo": r["photo"], "gp": overlap_frac(C, U), "iou_c": float((C & parts[c]).sum() / max(1, (C | parts[c]).sum())),
                          "cPER": r.get("cPER")})
        print("   -- 2  matched baseline: both objects restored = the original x_r (same rule as cPER)")
        row("x_r judged fake", base, "x_r_fake", s)
        row("x_r fake and still pointing at C&Mc", base, "x_r_still_cited", s)
        print("   -- 3  phantom-evidence cases (cPER = 1)")
        row("original x_r judged real", phantom_real, "x_r_real", s)
        print("   -- 7  cPER by localization of the citation on x_f")
        hi = [r for r in by_gp if r["gp"] >= 0.5]
        lo = [r for r in by_gp if r["gp"] < 0.5]
        row("cPER, GP >= 0.5", hi, "cPER", s)
        row("cPER, GP < 0.5", lo, "cPER", s)
        row("cPER, IoU(C, Mc) >= 0.5", [r for r in by_gp if r["iou_c"] >= 0.5], "cPER", s)
        row("cPER, IoU(C, Mc) < 0.5", [r for r in by_gp if r["iou_c"] < 0.5], "cPER", s)
    # 6 false positives on the originals of the two-object set
    real = [a for q, a in a1.items() if q.split("|")[2:3] == ["real"]]
    fp = float(np.mean([answer_verdict(a) == "fake" for a in real])) if real else float("nan")
    print(f"   -- 6  originals of the two-object set judged fake: {fp:.3f} (n = {len(real)})")
    st["false_positive_two_object"] = {"rate": fp, "n": len(real)}
    return samples, recs, meta


def false_positive_single(name, run, tag):
    a1 = read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))
    real = [a for a in a1 if a["query_id"].split("|")[2:3] == ["real"]]
    fp = float(np.mean([answer_verdict(a) == "fake" for a in real])) if real else float("nan")
    print(f"\n== 6  {name}: originals of the single-object set judged fake: {fp:.3f} (n = {len(real)})")
    OUT.setdefault(name, {})["false_positive_single"] = {"rate": fp, "n": len(real)}


# ------------------------------------------------------------------ 5 Fig. F1 selection
def fig_f1_counts(pairs_sida, pairs_fs, min_area=0.03):
    print("\n== 5  Fig. F1 rows: component records vs the selection script's rules")
    from make_fig4 import TEASER_SID, comp_case, follows_ok  # noqa: E402
    from make_teaser import Run  # noqa: E402

    res = {}
    Rs = Run(pairs_sida, "sida7b")
    steps = Counter()
    for sid, m in Rs.meta.items():
        k = comp_case(Rs, sid, "SIDA")
        if k is None:
            continue
        st = k["st"]
        conds = [("eligible (coverage rule)", True),
                 ("cPER = 1 (phantom evidence)", k["sida0"]["v"] == "fake" and k["sida1"]["v"] == "fake" and st["cper"]),
                 ("before: >= 50% of A covered (rule: 30%)", st["cov0_a"] >= 0.5),
                 ("before: >= 50% of the mask on A", st["share0_a"] >= 0.5),
                 ("after: >= 50% of A covered, < 10% of B", st["cov1_a"] >= 0.5 and st["cov1_b"] < 0.1),
                 ("after: >= 50% of the mask on A, < 50% on B", st["share1_a"] >= 0.5 and st["share1_b"] < 0.5),
                 (f"both objects >= {min_area:.0%} of the image", st["area_a"] >= min_area and st["area_b"] >= min_area),
                 ("not the Fig. 1 sample", sid != TEASER_SID)]
        for lab, ok in conds:
            if not ok:
                break
            steps[lab] += 1
    print("   (c) SIDA phantom evidence, records left after each condition:")
    for lab, _ in conds:
        print(f"      {steps[lab]:4d}  {lab}")
    res["c_sida_phantom"] = dict(steps)
    Rf = Run(pairs_fs, "fakeshield")
    n_switch = n_follow = 0
    for sid, m in Rf.meta.items():
        k = comp_case(Rf, sid, "FakeShield")
        if k is None:
            continue
        x1, B = k["x1"], k["B"]
        sw = x1["v"] == "fake" and x1["mask"].any() and (x1["mask"] & B).sum() / x1["mask"].sum() >= 0.5
        n_switch += bool(sw)
        n_follow += bool(follows_ok(k))
    print(f"   (b) FakeShield: switch (>= 50% of the new mask on B) {n_switch}; selection rule (>= 30% of B covered, "
          f"< 10% of A, before >= 30% of A and < 10% of B, both objects >= 2% of the image) {n_follow}")
    res["b_fakeshield_follows"] = {"switch": n_switch, "selection_rule": n_follow}
    OUT["fig_F1"] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="data/full_single/samples.jsonl")
    ap.add_argument("--sida", default="runs/full_sida")
    ap.add_argument("--fs", default="runs/full_fakeshield")
    ap.add_argument("--pairs-sida", default="runs/e4b_full_sida")
    ap.add_argument("--pairs-fs", default="runs/e4b_full_fakeshield")
    ap.add_argument("--out", default="supp_numbers.json")
    a = ap.parse_args()
    print(f"bootstrap: {N_BOOT} resamples, seed {SEED}; 'photo-clustered' resamples COCO photographs with all their forgeries")
    for name, run, tag, prun in (("SIDA", a.sida, "sida7b", a.pairs_sida), ("FakeShield", a.fs, "fakeshield", a.pairs_fs)):
        for fn, args in ((single_set, (name, run, tag, a.samples)), (component_set, (name, prun, tag, run)),
                         (false_positive_single, (name, run, tag))):
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001
                print(f"   !! {fn.__name__} for {name} failed: {e.__class__.__name__}: {e}")
    print("\n== 4  eligibility rules of component reversion (revert_audit/component.py)")
    rules = (f"coverage rule (primary): at least {COV_CITED:.0%} of one object's pixels are covered by the mask C and less "
             f"than {COV_OTHER:.0%} of the other object's; share rule (secondary): at least {SHARE_CITED:.0%} of C lies on one "
             f"object and less than {SHARE_OTHER:.0%} on the other. cPER: after the cited object M_c is restored the verdict "
             f"is still fake and the new mask covers at least half of C & M_c; switch: still fake and at least half of the "
             f"new mask lies on the other object.")
    print("   " + rules)
    OUT["rules"] = rules
    try:
        fig_f1_counts(a.pairs_sida, a.pairs_fs)
    except Exception as e:  # noqa: BLE001
        print(f"   !! Fig. F1 counts failed: {e.__class__.__name__}: {e}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n[supp_numbers] wrote {a.out}")


if __name__ == "__main__":
    main()
