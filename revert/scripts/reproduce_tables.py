"""Recompute the paper's numbers from the released model outputs. No GPU, no model weights, no images.

    python scripts/reproduce_tables.py --data data            # released data folder (manifest/ and outputs/)
    python scripts/reproduce_tables.py --data data --keep work # also keep the expanded working copy

How it works: the released outputs (gzip JSONL, masks as COCO RLE) are expanded into the folder layout the
experiment code wrote (masks as PNG), and the unchanged analysis code of revert_audit is run on it:
pipeline.analyze, diagnose, hypotheses, attraction, component.analyze and the per-model table of Fig. 3.
At the end every number is compared with the value printed in the paper (tolerance: the last printed digit).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# paper value, decimals printed in the paper
EXPECTED = {
    "SIDA": {"flag_rate": (0.39, 2), "AUC": (0.713, 3), "auditable": (462, 0), "CNG_M": (0.576, 3),
             "CNG_rand": (-0.010, 3), "CNG_model": (0.144, 3), "R": (0.26, 2), "R_clean": (0.25, 2),
             "PER": (0.558, 3), "PER_M": (0.112, 3), "PER_strict": (0.109, 3), "n_strict": (46, 0),
             "MNG": (0.050, 3), "STAY_minus_PER": (-0.081, 3), "n_STAY": (45, 0), "DR": (0.187, 3),
             "BA": (0.236, 3), "n_BA": (221, 0), "cPER": (0.270, 3), "switch": (0.006, 3), "cMNG": (0.177, 3),
             "n_component": (159, 0), "H_COV_dPER": (0.572, 3), "H_OOD_dPER": (0.120, 3), "H_OOD_dGP": (-0.049, 3)},
    "FakeShield": {"flag_rate": (0.153, 3), "AUC": (0.741, 3), "auditable": (184, 0), "CNG_M": (0.467, 3),
                   "CNG_rand": (-0.014, 3), "CNG_model": (0.307, 3), "R": (0.67, 2), "R_clean": (0.68, 2),
                   "PER": (0.261, 3), "PER_M": (0.000, 3), "PER_strict": (0.040, 3), "n_strict": (25, 0),
                   "MNG": (0.123, 3), "STAY_minus_PER": (0.368, 3), "n_STAY": (39, 0), "DR": (0.127, 3),
                   "BA": (0.117, 3), "n_BA": (45, 0), "cPER": (0.056, 3), "switch": (0.183, 3),
                   "cMNG": (0.253, 3), "n_component": (126, 0), "H_COV_dPER": (0.497, 3),
                   "H_OOD_dPER": (0.102, 3), "H_OOD_dGP": (-0.013, 3)},
}
RUNS = {"SIDA": ("single_sida", "component_sida", "sida7b"),
        "FakeShield": ("single_fakeshield", "component_fakeshield", "fakeshield")}


# ------------------------------------------------------------------ expand the released outputs
def rle_to_png(rle, mask_dir):
    from PIL import Image
    from pycocotools import mask as mu

    key = hashlib.sha1((str(rle["size"]) + rle["counts"]).encode()).hexdigest()[:20]
    p = os.path.join(mask_dir, key[:2], key + ".png")
    if not os.path.exists(p):
        m = mu.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.fromarray(m.astype(np.uint8) * 255, mode="L").save(p)
    return p


def unrle(obj, mask_dir):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.endswith("_rle") and isinstance(v, dict) and "counts" in v:
                out[k[:-4]] = rle_to_png(v, mask_dir)
            else:
                out[k] = unrle(v, mask_dir)
        return out
    if isinstance(obj, list):
        return [unrle(v, mask_dir) for v in obj]
    return obj


def read_gz_parts(path_first):
    """All parts of one released file, in order."""
    base = path_first.replace(".part01.jsonl.gz", ".jsonl.gz")
    parts = sorted(glob.glob(base.replace(".jsonl.gz", ".part*.jsonl.gz"))) or [base]
    for p in parts:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def expand(data, work):
    mask_dir = os.path.join(work, "masks")
    seen = set()
    for src in sorted(glob.glob(os.path.join(data, "**", "*.jsonl.gz"), recursive=True)):
        if ".part" in os.path.basename(src) and ".part01." not in os.path.basename(src):
            continue
        rel = os.path.relpath(src, data).replace(".part01.jsonl.gz", ".jsonl.gz")[: -len(".gz")]
        if rel in seen:
            continue
        seen.add(rel)
        dst = os.path.join(work, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for rec in read_gz_parts(src):
                f.write(json.dumps(unrle(rec, mask_dir), ensure_ascii=False) + "\n")
    for src in glob.glob(os.path.join(data, "outputs", "**", "*.json"), recursive=True):
        dst = os.path.join(work, "reference", os.path.relpath(src, data))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"[reproduce] expanded {len(seen)} files into {work}", flush=True)


# ------------------------------------------------------------------ recompute
def recompute(work, n_boot=1000):
    from revert_audit import component, diagnose, hypotheses, pipeline
    from revert_audit.attraction import attraction

    samples = os.path.join(work, "manifest", "single_final.jsonl")
    got, extra = {}, {}
    for det, (single, comp, tag) in RUNS.items():
        run = os.path.join(work, "outputs", single)
        a1 = os.path.join(run, "stage1", f"answers_{tag}.jsonl")
        if not os.path.exists(a1):
            print(f"[reproduce] {det}: no released outputs, skipped")
            continue
        g = got[det] = {}
        res = {}
        for ex in ("model", "oracle", "random"):
            s2 = os.path.join(run, f"stage2_{tag}_{ex}")
            if os.path.exists(os.path.join(s2, "meta.jsonl")):
                res[ex] = pipeline.analyze(samples, a1, os.path.join(s2, "meta.jsonl"), os.path.join(s2, "answers.jsonl"),
                                           os.path.join(run, "results", f"{tag}_{ex}"), n_boot)
        s1 = pipeline.stage1_summary(samples, a1)
        g["flag_rate"], g["AUC"] = s1["acc_fake"], s1["auc_fake_vs_real"]   # AUC of x_f against the originals
        print(f"[reproduce] {det}: AUC x_f vs real {s1['auc_fake_vs_real']:.3f}, x_raw vs real {s1.get('auc_raw_vs_real', float('nan')):.3f}")
        m = res["model"]["by_base"]["fake"]
        g["auditable"] = m["n"]
        g["DR"] = res["model"]["DR"][0]
        d = diagnose.diagnose(run, tag, samples)["by_base"]["fake"]
        cng = {ex: d["CNG_all"][ex][0] for ex in ("model", "oracle", "random")}
        g["CNG_model"], g["CNG_M"], g["CNG_rand"] = cng["model"], cng["oracle"], cng["random"]
        g["R"] = (cng["model"] - cng["random"]) / (cng["oracle"] - cng["random"])
        g["R_clean"] = d["R_clean"]
        g["PER"], g["PER_M"] = d["PER_all"][0], d["PER_oracle"][0]
        g["PER_strict"], g["n_strict"] = d["PER_strict_cov90"]["PER"][0], d["PER_strict_cov90"]["n"]
        g["MNG"] = d["MNG"]["MNG"][0]
        g["STAY_minus_PER"], g["n_STAY"] = d["MNG"]["STAY_minus_PER"][0], d["MNG"]["n_pair"]
        h = hypotheses.hypotheses(run, tag, samples)
        g["H_COV_dPER"] = h["H_COV"]["dPER_low_minus_high"][0]
        g["H_OOD_dPER"], g["H_OOD_dGP"] = h["H_OOD"]["dPER_out_minus_in"][0], h["H_OOD"]["dGP_out_minus_in"][0]
        ba = attraction(run, tag, samples, check_images=False)
        g["BA"], g["n_BA"] = ba["A"][0], ba["n"]
        extra[det] = {"by_coverage": d["by_coverage"], "by_editor": d["by_editor"],
                      "BA_components": {k: ba[k][0] for k in ("cover_B", "cover_null_in_M", "cover_outside_M")},
                      "H_COV": h["H_COV"], "H_OOD": h["H_OOD"], "stage1": s1}
        crun = os.path.join(work, "outputs", comp)
        out_dir = os.path.join(crun, f"component_{tag}")
        if os.path.exists(os.path.join(out_dir, "meta.jsonl")):
            c = component.analyze(os.path.join(crun, "pair_samples.jsonl"), os.path.join(crun, "stage1", f"answers_{tag}.jsonl"),
                                  out_dir, n_boot, ref_run=run, tag=tag)
            g["cPER"], g["switch"], g["cMNG"] = c["cPER"][0], c["switch"][0], c["cMNG"][0]
            g["n_component"] = c["counts"].get("eligible_coverage", 0)
            extra[det]["component"] = c
    return got, extra


def compare(got):
    print("\n" + "=" * 96)
    print(f"{'detector':<11} {'quantity':<16} {'recomputed':>12} {'paper':>10}  result")
    print("=" * 96)
    bad = 0
    for det, exp in EXPECTED.items():
        for k, (v, dec) in exp.items():
            x = got.get(det, {}).get(k)
            if x is None or x != x:
                print(f"{det:<11} {k:<16} {'-':>12} {v:>10}  MISSING")
                bad += 1
                continue
            ok = round(float(x), dec) == round(float(v), dec) or abs(float(x) - float(v)) <= 0.5 * 10 ** (-dec) + 1e-9
            bad += not ok
            print(f"{det:<11} {k:<16} {x:>12.4f} {v:>10}  {'ok' if ok else 'DIFFERS'}")
    print("=" * 96)
    print(f"{'all values match the paper' if bad == 0 else f'{bad} value(s) missing or different'}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--keep", default=None, help="keep the expanded working copy in this folder")
    ap.add_argument("--json", default="reproduced_numbers.json")
    a = ap.parse_args()
    work = a.keep or tempfile.mkdtemp(prefix="revert_repro_")
    try:
        expand(a.data, work)
        got, extra = recompute(work)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"main": got, "details": extra}, f, indent=1, ensure_ascii=False, default=float)
        print(f"[reproduce] all recomputed values: {a.json}")
        sys.exit(1 if compare(got) else 0)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
