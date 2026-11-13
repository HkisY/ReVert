"""Diagnostics for a finished pilot: are the numbers driven by the detector,
or by side effects of the intervention?

    python -m revert_audit.diagnose --run runs\\pilot_sida --tag sida7b --samples data\\pilot\\samples_reviewed.jsonl

Checks (each printed, and saved to <run>/diagnose_<tag>.json):
  1. control placement  shape-matched controls are meant to touch neither the
                        cited region nor the true edit M. With large regions
                        that is not always possible: how often do they land
                        on M, and do such controls change the score?
  2. clean-control CNG  CNG and R recomputed only on records whose controls
                        all stay off M (overlap <= 0.1).
  3. seam signature     reverting a region that cuts THROUGH the edit leaves a
                        generated/real boundary, itself a splice-like trace. If
                        the detector reacts to it, reverting raises the fake
                        score; how often, by how much of M the region covers.
  4. strict PER         phantom evidence only where reverting C removed >=90%
                        of the edit, so no generated pixels remain to cite.
  5. matched control    MNG and the PAIRED gap STAY_cin - PER on the same
                        records (GP >= 0.5): does removing the cited evidence
                        make the model stop citing it more often than removing
                        an equally large uncited piece of the edit?
  --skip N              analyse only samples after the first N of the samples
                        file (e.g. --skip 150: the part not seen in the 150 pilot).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .io import load_mask, read_jsonl

CLEAN = 0.1


def _boot(vals, fn=np.mean, n=1000, seed=0):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    bs = [fn(a[rng.integers(0, len(a), len(a))]) for _ in range(n)]
    return [float(fn(a)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _fmt(t):
    p, lo, hi = t
    return "n/a" if p != p else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def load_explainer(run: str, tag: str, ex: str, samples: dict) -> list[dict]:
    meta_p = os.path.join(run, f"stage2_{tag}_{ex}", "meta.jsonl")
    rec_p = os.path.join(run, "results", f"{tag}_{ex}", "records.jsonl")
    if not (os.path.exists(meta_p) and os.path.exists(rec_p)):
        return []
    meta = {(m["sample_id"], m["base"]): m for m in read_jsonl(meta_p) if not m.get("skipped")}
    out = []
    for r in read_jsonl(rec_p):
        m = meta.get((r["sample_id"], r["base"]))
        if m is None:
            continue
        r = dict(r)
        r["ctrl_overlap_M"] = m.get("ctrl_overlap_M")
        cov = None
        s = samples.get(r["sample_id"])
        if s and m.get("cmask") and os.path.exists(m["cmask"]):
            M, C = load_mask(s["mask"]), load_mask(m["cmask"])
            if M.shape == C.shape and M.sum():
                cov = float((M & C).sum() / M.sum())
        r["cov"] = cov
        r["nec"] = float(np.mean(r["s_revCt"]) - r["s_revC"])
        r["ctrl_drop"] = float(r["s0"] - np.mean(r["s_revCt"]))
        out.append(r)
    return out


def _out_name(tag, skip):
    return f"diagnose_{tag}.json" if not skip else f"diagnose_{tag}_skip{skip}.json"


def _clean(r):
    ov = r.get("ctrl_overlap_M")
    return ov is not None and len(ov) > 0 and max(ov) <= CLEAN


def diagnose(run: str, tag: str, samples_path: str, skip: int = 0) -> dict:
    order = read_jsonl(samples_path)
    samples = {s["id"]: s for s in order}
    keep = {s["id"] for s in order[skip:]}
    recs = {ex: [r for r in load_explainer(run, tag, ex, samples) if r["sample_id"] in keep]
            for ex in ("model", "oracle", "random")}
    out = {"run": run, "tag": tag, "skip": skip, "clean_threshold": CLEAN, "by_base": {}}
    for base in ("fake", "raw"):
        R = {ex: [r for r in rs if r["base"] == base] for ex, rs in recs.items()}
        if not R["model"]:
            continue
        d = {}
        # 1. control placement
        for ex, rs in R.items():
            with_ov = [r for r in rs if r.get("ctrl_overlap_M")]
            clean = [r for r in rs if _clean(r)]
            dirty = [r for r in with_ov if not _clean(r)]
            d[f"controls_{ex}"] = {
                "n": len(rs),
                "mean_ctrl_overlap_M": float(np.mean([np.mean(r["ctrl_overlap_M"]) for r in with_ov])) if with_ov else None,
                "frac_clean": len(clean) / len(with_ov) if with_ov else None,
                "ctrl_score_drop_clean": _boot([r["ctrl_drop"] for r in clean]),
                "ctrl_score_drop_dirty": _boot([r["ctrl_drop"] for r in dirty]),
            }
        # 2. CNG and R on clean controls only
        cng = {ex: _boot([r["nec"] for r in rs if _clean(r)]) for ex, rs in R.items()}
        d["CNG_clean"] = cng
        o, rnd, mdl = cng["oracle"][0], cng["random"][0], cng["model"][0]
        d["R_clean"] = float((mdl - rnd) / (o - rnd)) if all(x == x for x in (o, rnd, mdl)) and o - rnd > 1e-9 else None
        d["CNG_all"] = {ex: _boot([r["nec"] for r in rs]) for ex, rs in R.items()}
        # 3. seam signature (model citations)
        m = R["model"]
        rise = [float(r["s_revC"] > r["s0"] + 0.05) for r in m]
        d["revert_raises_score"] = _boot(rise)
        bins = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.01)]
        d["by_coverage"] = []
        for lo, hi in bins:
            rb = [r for r in m if r["cov"] is not None and lo <= r["cov"] < hi]
            d["by_coverage"].append({
                "coverage_of_M": f"[{lo:.2f},{min(hi, 1):.2f}]", "n": len(rb),
                "raises_score": _boot([float(r["s_revC"] > r["s0"] + 0.05) for r in rb]),
                "necessity_term": _boot([r["nec"] for r in rb]),
                "PER": _boot([r["per"] for r in rb if r.get("per") is not None]),
                "GP": _boot([r["gp"] for r in rb]),
            })
        # 4. strict PER
        strict = [r for r in m if r["cov"] is not None and r["cov"] >= 0.9 and r.get("per") is not None]
        d["PER_all"] = _boot([r["per"] for r in m if r.get("per") is not None])
        d["PER_strict_cov90"] = {"n": len(strict), "PER": _boot([r["per"] for r in strict])}
        d["PER_oracle"] = _boot([r["per"] for r in R["oracle"] if r.get("per") is not None])
        # 5. matched in-edit control, recomputed from records (respects --skip)
        mg = [r for r in m if r.get("s_revCin")]
        g50 = [r for r in m if r["gp"] >= 0.5]
        pair = [r for r in g50 if r.get("per") is not None and r.get("stay_cin") is not None]
        d["MNG"] = {
            "n": len(mg),
            "MNG": _boot([float(np.mean(r["s_revCin"]) - r["s_revC"]) for r in mg]),
            "PER_gp50": _boot([r["per"] for r in g50 if r.get("per") is not None]),
            "STAY_cin_gp50": _boot([r["stay_cin"] for r in g50 if r.get("stay_cin") is not None]),
            "n_pair": len(pair),
            "STAY_minus_PER": _boot([r["stay_cin"] - r["per"] for r in pair]),
        }
        # 6. by editor (is the finding specific to one inpainting model?)
        d["by_editor"] = {}
        for ed in sorted({r.get("editor") for r in m if r.get("editor")}):
            E = {ex: [r for r in rs if r.get("editor") == ed] for ex, rs in R.items()}
            c = {ex: _boot([r["nec"] for r in rs if _clean(r)]) for ex, rs in E.items()}
            o_, r_, m_ = c["oracle"][0], c["random"][0], c["model"][0]
            rc = float((m_ - r_) / (o_ - r_)) if all(x == x for x in (o_, r_, m_)) and o_ - r_ > 1e-9 else None
            em = E["model"]
            d["by_editor"][ed] = {
                "n": len(em), "CNG_clean": c, "R_clean": rc,
                "PER": _boot([r["per"] for r in em if r.get("per") is not None]),
                "GP": _boot([r["gp"] for r in em]),
                "MNG": _boot([float(np.mean(r["s_revCin"]) - r["s_revC"]) for r in em if r.get("s_revCin")]),
            }
        out["by_base"][base] = d
    with open(os.path.join(run, _out_name(tag, skip)), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def print_report(out: dict) -> None:
    for base, d in out["by_base"].items():
        mg = d.get("MNG")
        print(f"\n=== base: {base} ===")
        print("1. controls (mean overlap with M | share of records whose 3 controls all stay off M | "
              "score drop caused by controls: clean / touching M)")
        for ex in ("model", "oracle", "random"):
            c = d.get(f"controls_{ex}", {})
            if not c.get("n"):
                continue
            mo = c["mean_ctrl_overlap_M"]
            fc = c["frac_clean"]
            print(f"   {ex:6s} n={c['n']:3d}  overlap_M={'n/a' if mo is None else f'{mo:.3f}'}  "
                  f"clean={'n/a' if fc is None else f'{fc:.0%}'}  drop clean {_fmt(c['ctrl_score_drop_clean'])}  "
                  f"drop touching-M {_fmt(c['ctrl_score_drop_dirty'])}")
        print("2. CNG on clean-control records only:")
        for ex in ("model", "oracle", "random"):
            print(f"   {ex:6s} {_fmt(d['CNG_clean'][ex])}   (all records: {_fmt(d['CNG_all'][ex])})")
        rc = d["R_clean"]
        print(f"   R_clean = {'n/a' if rc is None else f'{rc:.0%}'}")
        print(f"3. reverting the cited region RAISES the fake score (>+0.05): {_fmt(d['revert_raises_score'])}")
        print("   by share of the true edit M covered by the cited region:")
        for b in d["by_coverage"]:
            print(f"   cov {b['coverage_of_M']}  n={b['n']:3d}  raises {_fmt(b['raises_score'])}  "
                  f"necessity {_fmt(b['necessity_term'])}  PER {_fmt(b['PER'])}  GP {_fmt(b['GP'])}")
        s = d["PER_strict_cov90"]
        print(f"4. PER all {_fmt(d['PER_all'])} | strict (cited region covers >=90% of M, n={s['n']}) {_fmt(s['PER'])} "
              f"| oracle (image fully restored) {_fmt(d['PER_oracle'])}")
        if mg and mg.get("n"):
            print(f"5. matched in-edit control (n={mg['n']}): MNG {_fmt(mg['MNG'])}  "
                  "(> 0: the cited forged pixels matter more than an equally large uncited piece of the edit)")
            print(f"   citation mostly on the edit (GP>=0.5): keeps citing the same place after ITS evidence is reverted "
                  f"{_fmt(mg['PER_gp50'])} vs after an UNCITED piece is reverted {_fmt(mg['STAY_cin_gp50'])}")
            if mg.get("n_pair"):
                print(f"   paired on the same records (n={mg['n_pair']}): STAY_cin - PER {_fmt(mg['STAY_minus_PER'])}  "
                      "(> 0: removing its own evidence makes it stop citing that place more often)")
        else:
            print("5. matched in-edit control: not run yet (rerun the pilot with v0.4.0 to add these queries)")
        if d.get("by_editor"):
            print("6. by editor (clean-control CNG; R_clean; PER; GP; MNG):")
            for ed, e in d["by_editor"].items():
                rc = e["R_clean"]
                print(f"   {ed:12s} n={e['n']:3d}  model {_fmt(e['CNG_clean']['model'])}  oracle {_fmt(e['CNG_clean']['oracle'])}  "
                      f"random {_fmt(e['CNG_clean']['random'])}  R_clean={'n/a' if rc is None else f'{rc:.0%}'}")
                print(f"   {'':12s}        PER {_fmt(e['PER'])}  GP {_fmt(e['GP'])}  MNG {_fmt(e['MNG'])}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--skip", type=int, default=0, help="ignore the first N samples of the samples file")
    a = ap.parse_args(argv)
    out = diagnose(a.run, a.tag, a.samples, a.skip)
    if a.skip:
        print(f"[diagnose] samples after the first {a.skip} only")
    print_report(out)
    print(f"\n[diagnose] saved {os.path.join(a.run, _out_name(a.tag, a.skip))}")


if __name__ == "__main__":
    main()
