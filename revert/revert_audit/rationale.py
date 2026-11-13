"""Rationale persistence (EXPLORATORY; CPU only, no new queries).

Forensic MLLMs that write a rationale (FakeShield's DTE-FDM text) can be asked the
text-level version of the mask question: after the cited pixels are restored, does
the rationale still attribute the manipulation to the restored content?

Target terms of an edited object = its COCO category (+ a few synonyms, plurals) and,
for 'replace' edits, the inserted thing parsed from the editor prompt.

Single-object set (runs/full_<det>):
    named0   rationale on x_f names the target                       (grounding rate, among flagged)
    rPER     after R(x_f, C): still 'fake' and rationale names target (among named0)
    rSTAY    same after the matched in-edit controls R(x_f, B_k)      (mean over k; its comparison)
    oracle   rPER with the true mask restored (stage2 oracle run)     (reference)
    joint    rPER cross-tabulated with mask PER (C' IoU >= 0.5)
Two-object set (runs/e4b_full_<det>, coverage rule; pairs whose two objects share a term are excluded):
    rcPER    after Rc (cited object fully restored): still 'fake' and rationale names the cited object
    rcSTAY   after Ro (uncited object restored): same                 (comparison)
    rswitch  after Rc: still 'fake' and rationale names only the remaining forged object

SIDA under its paper protocol answers with a fixed template ('[CLS] The image is tampered [SEG]'):
there is no rationale to audit and the script says so.

    python -m revert_audit.rationale --det fakeshield --coco-ann $COCO_ROOT/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from .io import load_mask, read_jsonl, write_jsonl
from .pipeline import answer_region, answer_verdict, qid
from .regions import iou

SYN = {
    "person": ["person", "man", "woman", "people", "boy", "girl", "child", "kid", "lady", "guy", "player", "human",
               "figure", "men", "women", "children", "skier", "surfer", "rider"],
    "dining table": ["table"], "cell phone": ["phone", "cellphone", "smartphone"], "potted plant": ["plant"],
    "sports ball": ["ball"], "motorcycle": ["motorcycle", "motorbike"], "airplane": ["airplane", "plane", "aircraft"],
    "couch": ["couch", "sofa"], "tv": ["tv", "television", "screen", "monitor"], "bicycle": ["bicycle", "bike"],
    "car": ["car", "vehicle"], "laptop": ["laptop", "computer"], "teddy bear": ["teddy", "bear", "toy"],
    "hair drier": ["hair drier", "hair dryer", "dryer"], "wine glass": ["glass"], "baseball bat": ["bat"],
    "baseball glove": ["glove"], "tennis racket": ["racket", "racquet"], "fire hydrant": ["hydrant"],
    "traffic light": ["traffic light"], "stop sign": ["stop sign", "sign"], "parking meter": ["meter"],
    "hot dog": ["hot dog", "hotdog"], "donut": ["donut", "doughnut"], "skis": ["ski", "skis"],
    "vase of flowers": ["vase", "flowers"], "traffic cone": ["cone"], "lamp": ["lamp", "light"],
}
_PROMPT = re.compile(r"a photo of (?:an? )?(.+?), realistic")


def _plural(w: str) -> list[str]:
    if " " in w or w.endswith("s"):
        return [w]
    return [w, w + ("es" if w.endswith(("sh", "ch", "x")) else "s")]


def terms(name: str | None) -> set[str]:
    if not name:
        return set()
    name = name.lower().strip()
    name = re.sub(r"^(an?|the) ", "", name)
    base = SYN.get(name) or [name] + ([name.split()[-1]] if " " in name else [])
    return {t for w in base for t in _plural(w)}


def thing_of(prompt: str | None) -> str | None:
    m = _PROMPT.search(prompt or "")
    return m.group(1) if m else None


def names(text: str | None, ts: set[str]) -> bool:
    if not text or not ts:
        return False
    t = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in ts)


def text_of(a: dict | None) -> str:
    return (a or {}).get("dte_text") or (a or {}).get("raw_text") or ""


def is_template(answers: list[dict]) -> bool:
    texts = {text_of(a).strip() for a in answers if text_of(a).strip()}
    return bool(texts) and all(t.startswith("[CLS]") for t in texts)


def _cluster_boot(vals, clusters, n=1000, seed=0):
    pts = [(v, c) for v, c in zip(vals, clusters) if v is not None and v == v]
    if not pts:
        return {"n": 0, "mean": None, "ci": [None, None]}
    by = {}
    for v, c in pts:
        by.setdefault(c, []).append(v)
    keys = list(by)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        xs = [x for i in pick for x in by[keys[i]]]
        bs.append(np.mean(xs))
    return {"n": len(pts), "mean": float(np.mean([v for v, _ in pts])),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def coco_categories(ann_path: str) -> dict[int, str]:
    with open(ann_path, encoding="utf-8") as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    return {a["id"]: cats[a["category_id"]] for a in d["annotations"]}


def _persist(ans, ts) -> float:
    """still 'fake' and the rationale names the target (a 'real' answer has no rationale: 0)."""
    return float(ans is not None and answer_verdict(ans) == "fake" and names(text_of(ans), ts))


# ------------------------------------------------------------- single-object set
def single(run, tag, samples_path, ann2cat, n_boot=1000):
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
    flagged = [a for a in a1.values() if a["query_id"].split("|")[2] == "fake" and answer_verdict(a) == "fake"]
    if is_template(flagged):
        return {"skipped": "fixed-template answers (no free-form rationale)"}
    out = {}
    rows = []
    for ex in ("model", "oracle"):
        s2 = os.path.join(run, f"stage2_{tag}_{ex}")
        if not os.path.exists(os.path.join(s2, "meta.jsonl")):
            continue
        a2 = {a["query_id"]: a for a in read_jsonl(os.path.join(s2, "answers.jsonl"))}
        for m in read_jsonl(os.path.join(s2, "meta.jsonl")):
            if m.get("skipped") or m["base"] != "fake":
                continue
            sid = m["sample_id"]
            s = samples.get(sid, {})
            a0, rc = a1.get(qid(sid, 1, "fake")), a2.get(qid(sid, 2, "fake", "revC", 0))
            if a0 is None or rc is None:
                continue
            obj = s.get("obj") or ann2cat.get(s.get("coco_ann_id"))
            thing = thing_of(s.get("prompt")) if s.get("edit_type") == "replace" else None
            t_obj, t_thing = terms(obj), terms(thing) - terms(obj)
            t_all = t_obj | t_thing
            r = {"explainer": ex, "sample_id": sid, "photo": s.get("coco_image_id", sid), "editor": s.get("editor"),
                 "edit_type": s.get("edit_type"), "obj": obj, "thing": thing,
                 "named0": float(names(text_of(a0), t_all)), "named0_obj": float(names(text_of(a0), t_obj)),
                 "named0_thing": float(names(text_of(a0), t_thing)),
                 "fake_c": float(answer_verdict(rc) == "fake"), "rPER": _persist(rc, t_all),
                 "rPER_obj": _persist(rc, t_obj), "rPER_thing": _persist(rc, t_thing),
                 "text0": text_of(a0)[:400], "text_c": text_of(rc)[:400]}
            if ex == "model":
                C = load_mask(m["cmask"])
                if answer_verdict(rc) == "fake":
                    r["mask_per"] = float(iou(answer_region(rc, *C.shape), C) >= 0.5)
                else:
                    r["mask_per"] = 0.0
                if m.get("cin_feasible"):
                    cin = [a2.get(qid(sid, 2, "fake", "revCin", k)) for k in range(len(m["ctrl_overlap"]))]
                    if all(x is not None for x in cin):
                        r["rSTAY"] = float(np.mean([_persist(x, t_all) for x in cin]))
            rows.append(r)
        rr = [r for r in rows if r["explainer"] == ex]
        g = [r for r in rr if r["named0"]]
        res = {"n_records": len(rr), "n_with_category": sum(bool(r["obj"]) for r in rr),
               "named0": _cluster_boot([r["named0"] for r in rr], [r["photo"] for r in rr], n_boot),
               "named0_obj": _cluster_boot([r["named0_obj"] for r in rr], [r["photo"] for r in rr], n_boot),
               "named0_thing": _cluster_boot([r["named0_thing"] for r in rr if r["thing"]],
                                             [r["photo"] for r in rr if r["thing"]], n_boot),
               "rPER": _cluster_boot([r["rPER"] for r in g], [r["photo"] for r in g], n_boot),
               "rPER_obj": _cluster_boot([r["rPER_obj"] for r in g], [r["photo"] for r in g], n_boot),
               "rPER_thing": _cluster_boot([r["rPER_thing"] for r in g if r["thing"]],
                                           [r["photo"] for r in g if r["thing"]], n_boot)}
        if ex == "model":
            gs = [r for r in g if r.get("rSTAY") is not None]
            res["rSTAY"] = _cluster_boot([r["rSTAY"] for r in gs], [r["photo"] for r in gs], n_boot)
            res["rSTAY_minus_rPER"] = _cluster_boot([r["rSTAY"] - r["rPER"] for r in gs], [r["photo"] for r in gs], n_boot)
            ct = {"mask&text": 0, "mask_only": 0, "text_only": 0, "neither": 0}
            for r in g:
                k = ("mask&text" if r["mask_per"] and r["rPER"] else "mask_only" if r["mask_per"] else
                     "text_only" if r["rPER"] else "neither")
                ct[k] += 1
            res["joint_with_mask_PER"] = ct
            res["by_editor"] = {e: _cluster_boot([r["rPER"] for r in g if r["editor"] == e],
                                                 [r["photo"] for r in g if r["editor"] == e], n_boot)
                                for e in sorted({r["editor"] for r in g if r["editor"]})}
        out[ex] = res
    return out, rows


# ---------------------------------------------------------------- two-object set
def pairs(run, tag, n_boot=1000):
    comp = os.path.join(run, f"component_{tag}")
    sp = os.path.join(run, "pair_samples.jsonl")
    if not os.path.exists(os.path.join(comp, "meta.jsonl")):
        return {"skipped": f"{comp}/meta.jsonl missing"}, []
    samples = {s["id"]: s for s in read_jsonl(sp)}
    a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
    a2 = {a["query_id"]: a for a in read_jsonl(os.path.join(comp, "answers.jsonl"))}
    flagged = [a for a in a1.values() if a["query_id"].split("|")[2] == "fake" and answer_verdict(a) == "fake"]
    if is_template(flagged):
        return {"skipped": "fixed-template answers (no free-form rationale)"}, []
    rows, n_shared = [], 0
    for m in read_jsonl(os.path.join(comp, "meta.jsonl")):
        if m.get("skipped") or not m.get("cov_ok"):
            continue
        sid, c = m["sample_id"], m["cov_cited"]
        o = 1 - c
        s = samples[sid]
        a0 = a1.get(qid(sid, 1, "fake"))
        ans = [a2.get(qid(sid, 2, "fake", f"comp{k}", 0)) for k in (1, 2)]
        if a0 is None or any(x is None for x in ans):
            continue
        T = []
        for p in s["parts"]:
            th = thing_of(p.get("prompt")) if p.get("edit_type") == "replace" else None
            T.append(terms(p.get("obj")) | terms(th))
        if T[0] & T[1]:
            n_shared += 1
            continue
        rc, ro = ans[c], ans[o]
        rows.append({"sample_id": sid, "photo": s.get("coco_image_id", sid), "editor": s.get("editor"),
                     "obj_c": s["parts"][c].get("obj"), "obj_o": s["parts"][o].get("obj"),
                     "named0_c": float(names(text_of(a0), T[c])), "named0_o": float(names(text_of(a0), T[o])),
                     "rcPER": _persist(rc, T[c]), "rcSTAY": _persist(ro, T[c]),
                     "rswitch": float(answer_verdict(rc) == "fake" and names(text_of(rc), T[o])
                                      and not names(text_of(rc), T[c])),
                     "text0": text_of(a0)[:400], "text_c": text_of(rc)[:400], "text_o": text_of(ro)[:400]})
    g = [r for r in rows if r["named0_c"]]
    P = lambda rs: [r["photo"] for r in rs]
    res = {"n_eligible_distinct_terms": len(rows), "n_excluded_shared_terms": n_shared,
           "named0_c": _cluster_boot([r["named0_c"] for r in rows], P(rows), n_boot),
           "rcPER": _cluster_boot([r["rcPER"] for r in g], P(g), n_boot),
           "rcSTAY": _cluster_boot([r["rcSTAY"] for r in g], P(g), n_boot),
           "rcSTAY_minus_rcPER": _cluster_boot([r["rcSTAY"] - r["rcPER"] for r in g], P(g), n_boot),
           "rswitch": _cluster_boot([r["rswitch"] for r in g], P(g), n_boot)}
    return res, rows


def _fmt(d):
    if not isinstance(d, dict) or "mean" not in d:
        return str(d)
    if d["mean"] is None:
        return "n=0"
    return f"{d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] (n={d['n']})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, choices=["fakeshield", "sida"])
    ap.add_argument("--single-run")
    ap.add_argument("--pairs-run")
    ap.add_argument("--samples", default="data/full_single/samples.jsonl")
    ap.add_argument("--coco-ann", help="instances_val2017.json (single-object samples store only the COCO ann id)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    tag = "sida7b" if a.det == "sida" else "fakeshield"
    sr = a.single_run or f"runs/full_{a.det}"
    pr = a.pairs_run or f"runs/e4b_full_{a.det}"
    ann2cat = coco_categories(a.coco_ann) if a.coco_ann else {}
    res = {"detector": a.det, "status": "exploratory"}
    s = single(sr, tag, a.samples, ann2cat, a.n_boot)
    res["single"], srows = (s if isinstance(s, tuple) else (s, []))
    res["pairs"], prows = pairs(pr, tag, a.n_boot)
    od = os.path.join(sr, "results")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"rationale_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    write_jsonl(srows + [{"set": "pairs", **r} for r in prows], os.path.join(od, f"rationale_{tag}_records.jsonl"))
    print(f"[rationale] {a.det}  (exploratory)")
    for part in ("single", "pairs"):
        sec = res[part]
        if "skipped" in sec:
            print(f"  {part}: skipped - {sec['skipped']}")
            continue
        for ex, block in (sec.items() if part == "single" else [("e4b", sec)]):
            print(f"  {part}/{ex}:")
            for k, v in block.items():
                if k == "by_editor":
                    for e, d in v.items():
                        print(f"      rPER[{e}] {_fmt(d)}")
                else:
                    print(f"    {k:<22} {_fmt(v)}")
    if res["single"] and "model" in res["single"] and not res["single"]["model"]["n_with_category"]:
        print("  NOTE: no object categories resolved; pass --coco-ann $COCO_ROOT/annotations/instances_val2017.json")
    print(f"[rationale] wrote {od}/rationale_{tag}.json and _records.jsonl")


if __name__ == "__main__":
    main()
