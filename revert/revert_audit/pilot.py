"""Progressive run with fixed stop rules.

Runs the audit on the first 50 samples, evaluates the gates, and only if the
answer is still "need more data" grows to 150, then 300. Work already done is
reused (answers are cached per query id), so growing costs only the new
samples. After every step it writes:
    <run>/gates_step<N>.json        gate statuses + decision
    <run>/report_<tag>.html         readable report with examples (open it)
and prints the decision in one line. The loop stops as soon as the decision
is not MORE_DATA.
"""

from __future__ import annotations

import json
import os

from . import pipeline
from .gates import ACTIONS, config_hash, evaluate, load_config
from .io import read_jsonl
from .report import build_report


def gpu_hours(paths) -> float:
    tot = 0.0
    for p in paths:
        for a in read_jsonl(p):
            tot += float(a.get("sec") or 0.0)
    return tot / 3600


def run_pilot(samples: str, run: str, detector, tag: str, cfg_path: str, steps=None, explainers=("model", "oracle", "random"),
              n_controls: int = 3, batch_size: int = 4, n_boot: int = 1000, log_every: int = 50,
              s2_bases=("fake", "raw")) -> dict:
    cfg = load_config(cfg_path)
    h = config_hash(cfg_path)
    steps = list(steps or cfg["steps"])
    n_avail = len(read_jsonl(samples))
    steps = [min(s, n_avail) for s in steps]
    steps = sorted(set(steps))
    s1 = os.path.join(run, "stage1")
    a1 = os.path.join(s1, f"answers_{tag}.jsonl")
    result = None
    for i, n in enumerate(steps):
        print(f"\n===== step {i + 1}/{len(steps)}: first {n} samples =====", flush=True)
        q1 = pipeline.stage1(samples, s1, limit=n)
        pipeline.run_detector(q1, detector, a1, batch_size, log_every)
        answer_files = [a1]
        for ex in explainers:
            s2 = os.path.join(run, f"stage2_{tag}_{ex}")
            q2 = pipeline.stage2(samples, s1, a1, s2, explainer=ex, n_controls=n_controls, limit=n,
                                 bases=tuple(s2_bases))
            a2 = os.path.join(s2, "answers.jsonl")
            pipeline.run_detector(q2, detector, a2, batch_size, log_every)
            answer_files.append(a2)
            pipeline.analyze(samples, a1, os.path.join(s2, "meta.jsonl"), a2, os.path.join(run, "results", f"{tag}_{ex}"), n_boot)
        res = {ex: json.load(open(os.path.join(run, "results", f"{tag}_{ex}", "metrics.json"), encoding="utf-8"))
               for ex in explainers}
        result = evaluate(res, cfg)
        used = gpu_hours(answer_files)
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        last = nxt is None
        if result["decision"] == "MORE_DATA" and last:
            result["decision"] = "UNDECIDED"
            result["action"] = ACTIONS["UNDECIDED"]
        budget = {"used_h": used, "next_n": nxt, "next_h": used * (nxt / n - 1) if nxt else 0.0}
        result.update({"step_n": n, "config_hash": h, "budget": budget})
        with open(os.path.join(run, f"gates_step{n}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        rp = build_report(run, samples, tag, result, h, budget)
        print(f"[pilot] n={n}  decision = {result['decision']} ({result['action'][0]})  GPU h so far {used:.2f}")
        for g, v in result["gates"].items():
            print(f"        {g} {v['status']:<12} {v['evidence']}")
        print(f"[pilot] report: {rp}")
        if result["decision"] != "MORE_DATA":
            break
    return result
