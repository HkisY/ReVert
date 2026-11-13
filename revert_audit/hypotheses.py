"""Hypotheses H-COV and H-OOD of the full experiment (decision rule: 95% cluster-bootstrap interval).

    H-COV  PER(coverage of M < 0.25) - PER(coverage >= 0.75) > 0
    H-OOD  in-family editor (sdxl) vs out-of-family editors (flux-fill, kandinsky) pooled:
           dPER = PER(out) - PER(in) > 0  and  dGP = GP(out) - GP(in) < 0

Records = model citations on x_f (stage 2, explainer 'model', base 'fake'). CIs are
cluster bootstraps over COCO photos (one photo can appear under several editors).
Each contrast: 'supported' if the 95% CI excludes 0 in the predicted direction,
'contradicted' if it excludes 0 in the other direction, else 'inconclusive';
'not testable' if a compared group has fewer than MIN_N records.

    python -m revert_audit.hypotheses --run runs\\full_sida --tag sida7b --samples data\\full_single\\samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .diagnose import load_explainer
from .io import read_jsonl
from .pipeline import answer_verdict, qid

MIN_N = 30
IN_FAMILY = ("sdxl",)
OUT_FAMILY = ("flux-fill", "kandinsky", "qwen-image", "qwen-edit-inpaint")
COV_LO, COV_HI = 0.25, 0.75


def family(editor: str | None) -> str | None:
    e = (editor or "").lower()
    if any(e == f or e.startswith(f + "-") for f in IN_FAMILY):
        return "in"
    if any(e == f or e.startswith(f + "-") for f in OUT_FAMILY):
        return "out"
    return None


def _mean(rs, key):
    v = [r[key] for r in rs if r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def _count(rs, key):
    return sum(r.get(key) is not None for r in rs)


def cluster_boot(recs: list[dict], stat, n_boot: int = 2000, seed: int = 0):
    """stat(records) -> float; resample photos (clusters) with replacement."""
    groups = defaultdict(list)
    for r in recs:
        groups[r["photo"]].append(r)
    keys = list(groups)
    est = stat(recs)
    if not keys or est != est:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat([r for i in pick for r in groups[keys[i]]])
        if v == v:
            bs.append(v)
    if not bs:
        return [est, float("nan"), float("nan")]
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def decide(ci, sign: int, ns) -> str:
    """sign +1: predicted > 0; -1: predicted < 0."""
    if min(ns) < MIN_N:
        return "not testable"
    _, lo, hi = ci
    if lo != lo or hi != hi:
        return "not testable"
    if (sign > 0 and lo > 0) or (sign < 0 and hi < 0):
        return "supported"
    if (sign > 0 and hi < 0) or (sign < 0 and lo > 0):
        return "contradicted"
    return "inconclusive"


def _diff(key, sel_a, sel_b):
    def f(rs):
        a, b = [r for r in rs if sel_a(r)], [r for r in rs if sel_b(r)]
        return _mean(a, key) - _mean(b, key)
    return f


def hypotheses(run: str, tag: str, samples_path: str, n_boot: int = 2000) -> dict:
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    recs = [r for r in load_explainer(run, tag, "model", samples) if r["base"] == "fake"]
    for r in recs:
        s = samples.get(r["sample_id"], {})
        r["photo"] = s.get("coco_image_id") or r["sample_id"]
        r["editor"] = r.get("editor") or s.get("editor")
        r["fam"] = family(r["editor"])
    out = {"run": run, "tag": tag, "n_records": len(recs), "min_n": MIN_N}

    lo = lambda r: r["cov"] is not None and r["cov"] < COV_LO  # noqa: E731
    hi = lambda r: r["cov"] is not None and r["cov"] >= COV_HI  # noqa: E731
    ci = cluster_boot(recs, _diff("per", lo, hi), n_boot)
    ns = [_count([r for r in recs if lo(r)], "per"), _count([r for r in recs if hi(r)], "per")]
    out["H_COV"] = {"dPER_low_minus_high": ci, "n_low": ns[0], "n_high": ns[1],
                    "PER_low": _mean([r for r in recs if lo(r)], "per"), "PER_high": _mean([r for r in recs if hi(r)], "per"),
                    "verdict": decide(ci, +1, ns)}

    is_out = lambda r: r["fam"] == "out"  # noqa: E731
    is_in = lambda r: r["fam"] == "in"  # noqa: E731
    fam_recs = [r for r in recs if r["fam"]]
    ci_per = cluster_boot(fam_recs, _diff("per", is_out, is_in), n_boot)
    ci_gp = cluster_boot(fam_recs, _diff("gp", is_out, is_in), n_boot, seed=1)
    n_per = [_count([r for r in fam_recs if is_out(r)], "per"), _count([r for r in fam_recs if is_in(r)], "per")]
    n_gp = [_count([r for r in fam_recs if is_out(r)], "gp"), _count([r for r in fam_recs if is_in(r)], "gp")]
    v_per, v_gp = decide(ci_per, +1, n_per), decide(ci_gp, -1, n_gp)
    if v_per == v_gp == "supported":
        v = "supported"
    elif "contradicted" in (v_per, v_gp):
        v = "contradicted"
    elif "not testable" in (v_per, v_gp):
        v = "not testable"
    else:
        v = "inconclusive"
    out["H_OOD"] = {"dPER_out_minus_in": ci_per, "dGP_out_minus_in": ci_gp, "n_per_out_in": n_per, "n_gp_out_in": n_gp,
                    "verdict_PER": v_per, "verdict_GP": v_gp, "verdict": v}

    # secondary: per editor (incl. stage-1 flag rate: PER and GP are conditional on flagging), within coverage bins
    a1p = os.path.join(run, "stage1", f"answers_{tag}.jsonl")
    a1 = {a["query_id"]: a for a in read_jsonl(a1p)} if os.path.exists(a1p) else {}
    per_ed = {}
    for ed in sorted({s.get("editor") for s in samples.values() if s.get("editor")}):
        ids = [sid for sid, s in samples.items() if s.get("editor") == ed]
        vs = [answer_verdict(a1[qid(sid, 1, "fake")]) for sid in ids if qid(sid, 1, "fake") in a1]
        er = [r for r in recs if r["editor"] == ed]
        per_ed[ed] = {"family": family(ed), "n_samples": len(ids), "flag_rate": float(np.mean([v == "fake" for v in vs])) if vs else None,
                      "n_records": len(er), "PER": cluster_boot(er, lambda rs: _mean(rs, "per"), n_boot),
                      "GP": cluster_boot(er, lambda rs: _mean(rs, "gp"), n_boot)}
    out["by_editor"] = per_ed
    out["H_OOD_by_coverage"] = {}
    for name, sel in (("cov<0.25", lo), ("0.25<=cov<0.75", lambda r: r["cov"] is not None and COV_LO <= r["cov"] < COV_HI),
                      ("cov>=0.75", hi)):
        rs = [r for r in fam_recs if sel(r)]
        out["H_OOD_by_coverage"][name] = {"n_out": sum(map(is_out, rs)), "n_in": sum(map(is_in, rs)),
                                          "dPER": cluster_boot(rs, _diff("per", is_out, is_in), n_boot)}
    with open(os.path.join(run, f"hypotheses_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def _fmt(t):
    p, lo, hi = t
    return "n/a" if p != p else f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def print_report(o: dict) -> None:
    print(f"\n=== confirmatory hypotheses: {o['tag']} ({o['n_records']} model records on x_f) ===")
    h = o["H_COV"]
    print(f"H-COV  PER(cov<{COV_LO}) {h['PER_low']:.3f} (n={h['n_low']})  -  PER(cov>={COV_HI}) {h['PER_high']:.3f} "
          f"(n={h['n_high']})  = {_fmt(h['dPER_low_minus_high'])}  -> {h['verdict']}")
    h = o["H_OOD"]
    print(f"H-OOD  dPER out-in {_fmt(h['dPER_out_minus_in'])} (n out/in {h['n_per_out_in']}) -> {h['verdict_PER']}")
    print(f"       dGP  out-in {_fmt(h['dGP_out_minus_in'])} (n out/in {h['n_gp_out_in']}) -> {h['verdict_GP']}")
    print(f"       H-OOD overall -> {h['verdict']}")
    print("   secondary, per editor:")
    for ed, e in o["by_editor"].items():
        fr = "n/a" if e["flag_rate"] is None else f"{e['flag_rate']:.2f}"
        print(f"   {ed:16s} [{e['family'] or '-'}] samples {e['n_samples']:4d} flagged {fr}  records {e['n_records']:4d}  "
              f"PER {_fmt(e['PER'])}  GP {_fmt(e['GP'])}")
    for name, d in o["H_OOD_by_coverage"].items():
        print(f"   {name:15s} n out/in {d['n_out']}/{d['n_in']}  dPER {_fmt(d['dPER'])}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    o = hypotheses(a.run, a.tag, a.samples, a.n_boot)
    print_report(o)
    print(f"\n[hypotheses] saved {os.path.join(a.run, f'hypotheses_{a.tag}.json')}")


if __name__ == "__main__":
    main()
