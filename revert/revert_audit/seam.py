"""E4 seam experiment: does the sharp real/forged seam left at the border of a
partially reverted region drive PER (phantom evidence) and the 'reverting raises
the fake score' cases?

For the model's own citations on x_f, the stage-2 queries are rebuilt with a
different border treatment (interventions.seam_alpha) and re-run, reusing the
stage-1 answers:
    hard       the pilot (sharp seam)                     -> results/<tag>_model
    ramp:N     soft seam over N px outside the region     -> results/<tag>_model_ramp<N>
    dilate:N/2 sharp seam, region grown by N/2 px          -> results/<tag>_model_dilate<N/2>
ramp:N and dilate:N/2 revert about the same extra (outside-C) area, so
ramp vs dilate isolates seam sharpness; hard vs dilate shows the effect of the
extra reverted area.

    python -m revert_audit.cli seam --samples ... --run runs\\pilot_sida --tag sida7b --ramps 8,24 <detector args>
    python -m revert_audit.seam --run runs\\pilot_sida --tag sida7b        (comparison only)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import pipeline
from .io import read_jsonl


def conditions(ramps) -> list[tuple[str, str]]:
    out = []
    for n in ramps:
        n = int(n)
        out.append((f"ramp{n}", f"ramp:{n}"))
        out.append((f"dilate{n // 2}", f"dilate:{n // 2}"))
    return out


def run_seam(samples, run, detector, tag, ramps=(8, 24), n_controls=3, batch_size=4, n_boot=1000, limit=None,
             log_every=50):
    s1 = os.path.join(run, "stage1")
    a1 = os.path.join(s1, f"answers_{tag}.jsonl")
    if not os.path.exists(a1):
        raise FileNotFoundError(f"{a1}: run the pilot first (the seam experiment reuses its stage-1 answers)")
    for name, mode in conditions(ramps):
        print(f"\n===== seam condition {name} ({mode}) =====", flush=True)
        s2 = os.path.join(run, f"stage2_{tag}_model_{name}")
        q2 = pipeline.stage2(samples, s1, a1, s2, explainer="model", n_controls=n_controls, bases=("fake",),
                             limit=limit, seam=mode)
        a2 = os.path.join(s2, "answers.jsonl")
        pipeline.run_detector(q2, detector, a2, batch_size, log_every)
        pipeline.analyze(samples, a1, os.path.join(s2, "meta.jsonl"), a2,
                         os.path.join(run, "results", f"{tag}_model_{name}"), n_boot)
    return compare(run, tag, [n for n, _ in conditions(ramps)], n_boot)


def _boot(vals, n=1000, seed=0):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _recs(run, tag, name):
    p = os.path.join(run, "results", f"{tag}_model" + ("" if name == "hard" else f"_{name}"), "records.jsonl")
    return {(r["sample_id"], r["base"]): r for r in read_jsonl(p) if r["base"] == "fake"} if os.path.exists(p) else {}


def compare(run, tag, names=None, n_boot=1000) -> dict:
    if names is None:
        names = sorted({d.split(f"{tag}_model_")[1] for d in os.listdir(os.path.join(run, "results"))
                        if d.startswith(f"{tag}_model_")})
    hard = _recs(run, tag, "hard")
    out = {"run": run, "tag": tag, "conditions": {}}
    for name in ["hard"] + list(names):
        R = hard if name == "hard" else _recs(run, tag, name)
        if not R:
            continue
        keys = sorted(set(R) & set(hard))
        rs = [R[k] for k in keys]
        nec = [float(np.mean(r["s_revCt"]) - r["s_revC"]) for r in rs]
        c = {
            "n": len(rs),
            "CNG": _boot(nec, n_boot),
            "PER": _boot([r["per"] for r in rs if r.get("per") is not None], n_boot),
            "PER_gp50": _boot([r["per"] for r in rs if r.get("per") is not None and r["gp"] >= 0.5], n_boot),
            "raises": _boot([float(r["s_revC"] > r["s0"] + 0.05) for r in rs], n_boot),
            "MNG": _boot([float(np.mean(r["s_revCin"]) - r["s_revC"]) for r in rs if r.get("s_revCin")], n_boot),
        }
        if name != "hard":  # paired on the same records
            pr = [(R[k]["per"], hard[k]["per"]) for k in keys if R[k].get("per") is not None and hard[k].get("per") is not None]
            c["dPER_vs_hard"] = _boot([a - b for a, b in pr], n_boot)
            c["dRaises_vs_hard"] = _boot([float(R[k]["s_revC"] > R[k]["s0"] + 0.05) - float(hard[k]["s_revC"] > hard[k]["s0"] + 0.05)
                                          for k in keys], n_boot)
        out["conditions"][name] = c
    # primary E4 contrast: soft vs sharp seam at about the same extra reverted area
    out["ramp_vs_dilate"] = {}
    for name in names:
        if not name.startswith("ramp"):
            continue
        n = int(name[4:])
        A, B = _recs(run, tag, name), _recs(run, tag, f"dilate{n // 2}")
        keys = sorted(set(A) & set(B))
        pr = [(A[k]["per"], B[k]["per"]) for k in keys if A[k].get("per") is not None and B[k].get("per") is not None]
        if pr:
            out["ramp_vs_dilate"][name] = {
                "n": len(pr), "dPER": _boot([a - b for a, b in pr], n_boot),
                "dRaises": _boot([float(A[k]["s_revC"] > A[k]["s0"] + 0.05) - float(B[k]["s_revC"] > B[k]["s0"] + 0.05)
                                  for k in keys], n_boot)}
    with open(os.path.join(run, f"seam_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def _fmt(t):
    p, lo, hi = t
    return "n/a" if p != p else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def print_report(out):
    print(f"\n=== seam experiment: {out['tag']} (x_f, model citations) ===")
    for name, c in out["conditions"].items():
        print(f"{name:10s} n={c['n']:3d}  PER {_fmt(c['PER'])}  PER_gp50 {_fmt(c['PER_gp50'])}  raises {_fmt(c['raises'])}")
        print(f"{'':10s}        CNG {_fmt(c['CNG'])}  MNG {_fmt(c['MNG'])}")
        if "dPER_vs_hard" in c:
            print(f"{'':10s}        paired vs hard: dPER {_fmt(c['dPER_vs_hard'])}  dRaises {_fmt(c['dRaises_vs_hard'])}")
    for name, c in out.get("ramp_vs_dilate", {}).items():
        print(f"PRIMARY {name} vs dilate{int(name[4:]) // 2} (soft vs sharp seam, same extra area), paired n={c['n']}: "
              f"dPER {_fmt(c['dPER'])}  dRaises {_fmt(c['dRaises'])}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    out = compare(a.run, a.tag)
    print_report(out)
    print(f"\n[seam] saved {os.path.join(a.run, f'seam_{a.tag}.json')}")


if __name__ == "__main__":
    main()
