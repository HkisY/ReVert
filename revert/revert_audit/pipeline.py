"""The audit pipeline: stage1 -> run -> stage2 -> run -> analyze.

Query ids are "<sample_id>|<stage>|<base>|<kind>|<k>" and are the join key
between query files and answer files, so any detector (built-in or external)
only has to echo the query_id back.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict

import numpy as np

from .detectors.base import answer_score, answer_verdict
from .interventions import composite, keep, revert, revert_seam
from .io import append_jsonl, load_mask, load_rgb, read_jsonl, save_mask, save_rgb, write_jsonl
from .metrics import mng, auc, bootstrap_ci, cng, csg, decoupling_rate, gp, necessity_terms, per, sufficiency_terms
from .regions import area, blob_in_region, boxes_to_mask, iou, overlap_frac, translate_mask


def qid(sample_id: str, stage: int, base: str, kind: str = "orig", k: int = 0) -> str:
    return f"{sample_id}|s{stage}|{base}|{kind}|{k}"


def _seed(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ stage 1
def stage1(samples_path: str, out_dir: str, bases=("fake", "raw", "real"), limit: int | None = None) -> str:
    """Write the original-image queries. Every query image is re-saved as PNG
    so that all queries share one container format. `limit` keeps only the
    first N samples (the builder writes samples in random order), which is
    how the progressive pilot grows 50 -> 150 -> 300 without redoing work."""
    samples = read_jsonl(samples_path)
    if limit:
        samples = samples[:limit]
    qrows = []
    for s in samples:
        real = load_rgb(s["real"])
        mask = load_mask(s["mask"])
        imgs = {"real": real}
        if s.get("raw"):
            imgs["raw"] = load_rgb(s["raw"])
        if s.get("fake"):
            imgs["fake"] = load_rgb(s["fake"])
        elif "raw" in imgs:
            imgs["fake"] = composite(imgs["raw"], real, mask)
        for b in bases:
            if b not in imgs:
                continue
            if imgs[b].shape != real.shape:
                raise ValueError(f"{s['id']}: {b} shape {imgs[b].shape} != real {real.shape}")
            p = os.path.join(out_dir, "images", f"{s['id']}__{b}.png")
            if not os.path.exists(p):
                save_rgb(imgs[b], p)
            qrows.append({"query_id": qid(s["id"], 1, b), "sample_id": s["id"], "stage": 1, "base": b, "kind": "orig", "path": p})
    out = os.path.join(out_dir, "queries.jsonl")
    write_jsonl(qrows, out)
    return out


# ---------------------------------------------------------------------- run
def _is_oom(e: BaseException) -> bool:
    return e.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _holders_note() -> str:
    try:
        from .detectors.sida import describe_gpu_holders

        return f"; GPU held by: {describe_gpu_holders()}"
    except Exception:  # noqa: BLE001
        return ""


def _free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


OOM_RETRIES = 3
OOM_BACKOFF_S = (30, 60, 120)  # waits before the retries: another job may hold memory for minutes


def run_detector(queries_path: str, detector, out_path: str, batch_size: int = 4, log_every: int = 50,
                 explain_kinds=("orig", "revC", "revCin")) -> str:
    """Run a built-in detector over a query file.

    - Resumable: query ids that already have an answer in out_path are skipped.
    - Only queries whose kind is in `explain_kinds` get a generated
      explanation (stage-1 originals and R(x, C), which PER needs); control
      queries are score-only, several times cheaper for MLLMs.
    - Trades time for memory on small GPUs: on CUDA out-of-memory the batch
      is halved (and stays smaller for the rest of the run). A single image
      that still does not fit after OOM_RETRIES retries is logged to
      <out>.failed.jsonl and skipped, so it never enters the metrics with a
      made-up score; a later run of the same command retries it (only
      answered query ids are skipped).
    """
    done = {a["query_id"] for a in read_jsonl(out_path)}
    todo = [q for q in read_jsonl(queries_path) if q["query_id"] not in done]
    groups = [(True, [q for q in todo if q.get("kind", "orig") in explain_kinds]),
              (False, [q for q in todo if q.get("kind", "orig") not in explain_kinds])]
    n_done, bs, n_fail = 0, max(1, batch_size), 0
    retries = 0  # OOMs of the same single query at batch size 1
    for explain, qs in groups:
        i = 0
        while i < len(qs):
            batch = qs[i : i + bs]
            t0 = time.time()
            try:
                answers = detector.predict([load_rgb(q["path"]) for q in batch], explain=explain)
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e):
                    raise
                _free_cuda()
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"[run] CUDA OOM -> batch size {bs}", flush=True)
                    continue
                if retries < OOM_RETRIES:  # a transient peak (fragmentation): the worker has freed its cache
                    retries += 1
                    wait_s = OOM_BACKOFF_S[min(retries, len(OOM_BACKOFF_S)) - 1]
                    print(f"[run] CUDA OOM at batch size 1 -> retry {retries}/{OOM_RETRIES} in {wait_s} s"
                          f"{_holders_note()}", flush=True)
                    time.sleep(wait_s)
                    continue
                append_jsonl({"query_id": batch[0]["query_id"], "error": "oom"}, out_path + ".failed.jsonl")
                n_fail += 1
                retries = 0
                i += 1
                continue
            retries = 0
            sec = (time.time() - t0) / len(batch)
            for q, a in zip(batch, answers):
                a = dict(a)
                a["query_id"] = q["query_id"]
                a["detector"] = getattr(detector, "name", "unknown")
                a["sec"] = round(sec, 3)  # GPU time per query, for the budget guard
                append_jsonl(a, out_path)
            i += len(batch)
            n_done += len(batch)
            if log_every and n_done % log_every < len(batch):
                print(f"[run] {n_done}/{len(todo)} (explain={explain}, bs={bs})", flush=True)
    if n_fail:
        print(f"[run] {n_fail} queries failed with OOM even at batch size 1; see {out_path}.failed.jsonl")
    finalize = getattr(detector, "finalize", None)
    if finalize is not None:  # two-stage detectors (FakeShield: masks from a second model)
        finalize(queries_path, out_path)
    return out_path


def _explained(ans: dict) -> bool:
    """Does this answer carry the detector's citation (possibly empty)? Score-only
    answers do not. An explained answer that says 'real', or 'fake' without a
    region, cites nothing: it is NOT still citing the old place (PER = 0), it must
    not drop out of PER. (Detectors without a boxes list, e.g. FakeShield whose
    mask exists only for 'fake', were otherwise counted only when still 'fake'.)"""
    if ans.get("boxes") is not None or ans.get("mask_path"):
        return True
    return ans.get("parse_mode") not in (None, "score_only", "failed") and answer_verdict(ans) is not None


def answer_region(ans: dict, h: int, w: int) -> np.ndarray:
    """Union of the cited boxes and the predicted mask (if any)."""
    m = boxes_to_mask(ans.get("boxes") or [], h, w)
    mp = ans.get("mask_path")
    if mp and os.path.exists(mp):
        pm = load_mask(mp)
        if pm.shape != (h, w):
            import cv2

            pm = cv2.resize(pm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        m |= pm
    return m


# ------------------------------------------------------------------ stage 2
def stage2(
    samples_path: str,
    s1_dir: str,
    s1_answers_path: str,
    out_dir: str,
    explainer: str = "model",
    n_controls: int = 5,
    bases=("fake", "raw"),
    feather_px: int = 0,
    require_fake: bool = True,
    max_ctrl_overlap: float = 0.1,
    limit: int | None = None,
    seam: str = "hard",
) -> str:
    """Build the intervention queries from each stage-1 answer.

    seam: how revert-type queries (revC, revCt, revCin) treat the region border;
    'hard' (default) or 'ramp:N' / 'dilate:N' for the seam experiment (E4).

    explainer:
      model   C = region the detector cited                (the audit)
      oracle  C = ground-truth edit mask M                  (sanity: upper reference)
      random  C = M rigidly shifted to a random location    (sanity: should give CNG ~ 0)
    """
    rows = read_jsonl(samples_path)
    if limit:
        rows = rows[:limit]
    samples = {s["id"]: s for s in rows}
    ans1 = {a["query_id"]: a for a in read_jsonl(s1_answers_path)}
    qrows, meta = [], []
    for sid, s in samples.items():
        real = load_rgb(s["real"])
        M = load_mask(s["mask"])
        h, w = M.shape
        for b in bases:
            a = ans1.get(qid(sid, 1, b))
            row = {"sample_id": sid, "base": b, "explainer": explainer, "skipped": None}
            if a is None:
                row["skipped"] = "no_stage1_answer"
                meta.append(row)
                continue
            base_path = os.path.join(s1_dir, "images", f"{sid}__{b}.png")
            if not os.path.exists(base_path):
                base_img = load_rgb(s[b]) if s.get(b) else composite(load_rgb(s["raw"]), real, M)
            else:
                base_img = load_rgb(base_path)
            if require_fake and answer_verdict(a) != "fake":
                row["skipped"] = "verdict_not_fake"
                meta.append(row)
                continue
            if explainer == "model":
                C = answer_region(a, h, w)
            elif explainer == "oracle":
                C = M.copy()
            elif explainer == "random":
                C, _ = translate_mask(M, np.random.default_rng(_seed(sid, b, "rand")), max_overlap=max_ctrl_overlap)
            else:
                raise ValueError(explainer)
            if area(C) == 0:
                row["skipped"] = "no_region"
                meta.append(row)
                continue
            cpath = os.path.join(out_dir, "regions", f"{sid}__{b}__C.png")
            save_mask(C, cpath)
            row.update({"cmask": cpath, "c_area": area(C) / float(h * w), "gp": overlap_frac(C, M), "ctrl_overlap": []})

            def add(kind, k, img):
                p = os.path.join(out_dir, "images", f"{sid}__{b}__{kind}{k}.png")
                if not os.path.exists(p):
                    save_rgb(img, p)
                qrows.append({"query_id": qid(sid, 2, b, kind, k), "sample_id": sid, "stage": 2, "base": b, "kind": kind, "k": k, "path": p})

            def rev(region):
                if seam not in ("hard", "", None):
                    return revert_seam(base_img, real, region, seam)
                return revert(base_img, real, region, feather_px)

            row["seam"] = seam
            add("revC", 0, rev(C))
            add("keepC", 0, keep(base_img, real, C, feather_px))
            rng = np.random.default_rng(_seed(sid, b, "ctrl"))
            for k in range(n_controls):
                # controls stay out of both the cited region and the true edit:
                # landing on M would remove real evidence and bias CNG down
                Ct, ov = translate_mask(C, rng, max_overlap=max_ctrl_overlap, avoid=C | M)
                row["ctrl_overlap"].append(ov)
                row.setdefault("ctrl_overlap_M", []).append(overlap_frac(Ct, M))
                add("revCt", k, rev(Ct))
                add("keepCt", k, keep(base_img, real, Ct, feather_px))
            # matched in-edit control (model citations): revert an equally large piece of the
            # TRUE edit that the detector did not cite. Both interventions remove forged pixels
            # and cut through M (seams), so the difference isolates WHERE the citation points.
            if explainer == "model":
                cited_forged = C & M
                a_cf = int(cited_forged.sum())
                avail = M & ~C
                row["cin_area"] = a_cf
                row["cin_feasible"] = bool(a_cf > 0 and avail.sum() >= a_cf)
                if row["cin_feasible"]:
                    rng_in = np.random.default_rng(_seed(sid, b, "cin"))
                    for k in range(n_controls):
                        blob = blob_in_region(avail, a_cf, rng_in)
                        add("revCin", k, rev(blob))
            meta.append(row)
    write_jsonl(meta, os.path.join(out_dir, "meta.jsonl"))
    out = os.path.join(out_dir, "queries.jsonl")
    write_jsonl(qrows, out)
    return out


# ------------------------------------------------------------------ analyze
def build_records(samples_path, s1_answers_path, s2_meta_path, s2_answers_path, per_iou: float = 0.5):
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    a1 = {a["query_id"]: a for a in read_jsonl(s1_answers_path)}
    a2 = {a["query_id"]: a for a in read_jsonl(s2_answers_path)}
    recs = []
    for m in read_jsonl(s2_meta_path):
        if m.get("skipped"):
            continue
        sid, b = m["sample_id"], m["base"]
        a0 = a1.get(qid(sid, 1, b))
        rc = a2.get(qid(sid, 2, b, "revC", 0))
        kc = a2.get(qid(sid, 2, b, "keepC", 0))
        n = len(m["ctrl_overlap"])
        rct = [a2.get(qid(sid, 2, b, "revCt", k)) for k in range(n)]
        kct = [a2.get(qid(sid, 2, b, "keepCt", k)) for k in range(n)]
        if a0 is None or rc is None or kc is None or any(x is None for x in rct + kct):
            continue  # stage-2 answers not complete for this sample yet
        C = load_mask(m["cmask"])
        h, w = C.shape
        per_v = None
        if _explained(rc):
            C2 = answer_region(rc, h, w)
            per_v = float(answer_verdict(rc) == "fake" and iou(C2, C) >= per_iou)
        rcin = [a2.get(qid(sid, 2, b, "revCin", k)) for k in range(n)] if m.get("cin_feasible") else []
        s_revCin, stay_cin = None, None
        if rcin and all(x is not None for x in rcin):
            s_revCin = [answer_score(x) for x in rcin]
            stays = []
            for x in rcin:
                if _explained(x):
                    stays.append(float(answer_verdict(x) == "fake" and iou(answer_region(x, h, w), C) >= per_iou))
            stay_cin = float(np.mean(stays)) if stays else None
        recs.append(
            {
                "sample_id": sid,
                "s_revCin": s_revCin,
                "stay_cin": stay_cin,
                "cin_area": m.get("cin_area"),
                "base": b,
                "editor": samples.get(sid, {}).get("editor"),
                "edit_type": samples.get(sid, {}).get("edit_type"),
                "s0": answer_score(a0),
                "s_revC": answer_score(rc),
                "s_revCt": [answer_score(x) for x in rct],
                "s_keepC": answer_score(kc),
                "s_keepCt": [answer_score(x) for x in kct],
                "gp": m["gp"],
                "per": per_v,
                "c_area": m["c_area"],
                "ctrl_overlap_mean": float(np.mean(m["ctrl_overlap"])) if n else None,
            }
        )
    return recs


def stage1_summary(samples_path, s1_answers_path):
    samples = read_jsonl(samples_path)
    a1 = {a["query_id"]: a for a in read_jsonl(s1_answers_path)}
    out = {}
    scores = defaultdict(list)
    verdicts = defaultdict(list)
    parse_fail = []
    for s in samples:
        for b in ("fake", "raw", "real"):
            a = a1.get(qid(s["id"], 1, b))
            if a is None:
                continue
            scores[b].append(answer_score(a))
            verdicts[b].append(answer_verdict(a))
            parse_fail.append(a.get("parse_mode") == "failed")
    for b in ("fake", "raw", "real"):
        if verdicts[b]:
            target = "real" if b == "real" else "fake"
            out[f"acc_{b}"] = float(np.mean([v == target for v in verdicts[b]]))
            out[f"n_{b}"] = len(verdicts[b])
    out["auc_fake_vs_real"] = auc(scores["fake"], scores["real"])
    out["auc_raw_vs_real"] = auc(scores["raw"], scores["real"])
    out["parse_fail_rate"] = float(np.mean(parse_fail)) if parse_fail else float("nan")
    return out


def dr_pairs(samples_path, s1_answers_path, oracle_citation: bool = False):
    """Per-sample (v_raw, gp_raw, v_fake). Note K(x_raw, M) == x_f exactly,
    so the decoupling test needs no extra query: it compares the stage-1
    answers on x_raw and on x_f."""
    samples = read_jsonl(samples_path)
    a1 = {a["query_id"]: a for a in read_jsonl(s1_answers_path)}
    pairs = []
    for s in samples:
        ar, af = a1.get(qid(s["id"], 1, "raw")), a1.get(qid(s["id"], 1, "fake"))
        if ar is None or af is None:
            continue
        if oracle_citation:  # for detectors without regions (e.g. the CLIP probe)
            g = 1.0
        else:
            M = load_mask(s["mask"])
            C = answer_region(ar, *M.shape)
            g = overlap_frac(C, M) if C.any() else 0.0
        pairs.append({"sample_id": s["id"], "v_raw": answer_verdict(ar), "gp_raw": g, "v_fake": answer_verdict(af)})
    return pairs


def analyze(samples_path, s1_answers_path, s2_meta_path, s2_answers_path, out_dir, n_boot: int = 1000, gp_min: float = 0.5,
            oracle_citation: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    recs = build_records(samples_path, s1_answers_path, s2_meta_path, s2_answers_path)
    write_jsonl(recs, os.path.join(out_dir, "records.jsonl"))
    meta = read_jsonl(s2_meta_path)
    result = {"stage1": stage1_summary(samples_path, s1_answers_path), "by_base": {}}
    skipped = defaultdict(int)
    for m in meta:
        skipped[f"{m['base']}:{m.get('skipped') or 'kept'}"] += 1
    result["stage2_counts"] = dict(skipped)
    for b in sorted({r["base"] for r in recs}):
        rb = [r for r in recs if r["base"] == b]
        result["by_base"][b] = {
            "n": len(rb),
            "CNG": bootstrap_ci(rb, cng, n_boot),
            "CSG": bootstrap_ci(rb, csg, n_boot),
            "GP": bootstrap_ci(rb, gp, n_boot),
            "PER": bootstrap_ci([r for r in rb if r["per"] is not None], per, n_boot),
            "mean_necessity_term": float(np.mean(necessity_terms(rb))) if rb else float("nan"),
            "mean_sufficiency_term": float(np.mean(sufficiency_terms(rb))) if rb else float("nan"),
            # matched in-edit control: does the CITED forged evidence matter more than an equally
            # large UNCITED piece of the same edit? (> 0: the citation points at what drives the decision)
            "MNG": bootstrap_ci([r for r in rb if r.get("s_revCin")], mng, n_boot),
            "n_MNG": len([r for r in rb if r.get("s_revCin")]),
            # phantom evidence where the citation was mostly on forged pixels, next to how often
            # it keeps citing the same place when an UNCITED piece of the edit is reverted instead
            "PER_gp50": bootstrap_ci([r for r in rb if r["per"] is not None and r["gp"] >= 0.5], per, n_boot),
            "STAY_cin_gp50": bootstrap_ci([r for r in rb if r.get("stay_cin") is not None and r["gp"] >= 0.5],
                                          lambda xs: float(np.mean([x["stay_cin"] for x in xs])) if xs else float("nan"), n_boot),
            "c_area_mean": float(np.mean([r["c_area"] for r in rb])) if rb else float("nan"),
        }
    pairs = dr_pairs(samples_path, s1_answers_path, oracle_citation)
    dr_point, n_el = decoupling_rate(pairs, gp_min)
    elig = [p for p in pairs if p["v_raw"] == "fake" and p["gp_raw"] >= gp_min and p["v_fake"] is not None]
    result["DR"] = bootstrap_ci(elig, lambda xs: float(np.mean([p["v_fake"] == "real" for p in xs])) if xs else float("nan"), n_boot)
    result["DR_n_eligible"] = n_el
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.md"), "w", encoding="utf-8") as f:
        f.write(format_table(result))
    return result


def _fmt(t):
    if isinstance(t, (list, tuple)) and len(t) == 3:
        p, lo, hi = t
        if p != p:  # NaN
            return "n/a"
        if lo != lo:
            return f"{p:.3f}"
        return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
    return str(t)


def format_table(res: dict) -> str:
    s1 = res["stage1"]
    lines = ["## Stage 1 (detection)", ""]
    lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in s1.items()]
    lines += ["", "## Faithfulness (95% bootstrap CI)", "", "| base | n | CNG | CSG | GP | PER |", "| --- | --- | --- | --- | --- | --- |"]
    for b, d in res["by_base"].items():
        lines.append(f"| {b} | {d['n']} | {_fmt(d['CNG'])} | {_fmt(d['CSG'])} | {_fmt(d['GP'])} | {_fmt(d['PER'])} |")
    lines += ["", f"DR (decoupling rate): {_fmt(res['DR'])}  (eligible n = {res['DR_n_eligible']})", "", f"Stage-2 counts: {res['stage2_counts']}", ""]
    return "\n".join(lines)
