"""Decision gates: turn metrics into "go / stop / switch / pivot".

The point is to decide in advance, with numbers, what result means what, so
that a dead end is recognised after 50-150 samples instead of after months.

    G0  data & parsing   is the pipeline producing usable answers?
    G1  detection        can this detector tell the fakes from the reals at all?
                         (if not, its explanations cannot be audited)
    G2  protocol         does the reversion test work on this detector? The
                         ORACLE explanation (C = true mask) must change the
                         score clearly more than a RANDOM region does.
    G3  finding          is there a faithfulness gap? Any of
                           - faithfulness ratio R = (CNG_model - CNG_random) / (CNG_oracle - CNG_random)
                             is at most 0.70 AND model CNG is significantly below oracle (needs G2).
                             R = 1: as good as the true edit mask; R = 0: no better than a random region
                           - decoupling rate DR high
                           - phantom evidence rate PER high
                         or the opposite: the explanations are faithful.

Decision (first rule that applies):
    G0 FAIL                    -> FIX_PROMPT      fix prompt/parsing, rerun the same samples
    almost never says "fake"   -> NO_FLAGS        (>= min_flag_n fakes, Wilson upper bound of the
                                                  flag rate < min_flag_rate): nothing to audit
    G1 FAIL                    -> SWITCH_DETECTOR this detector cannot detect these fakes
    G0 or G1 not yet PASS      -> MORE_DATA       no finding counts before both pass
    G3 PASS                    -> GO              scale up: the paper's finding exists
    G3 FAITHFUL                -> PIVOT           explanations are faithful: switch to the
                                                  pre-planned comparison-study framing
    G2 FAIL                    -> NOT_INFORMATIVE the reversion test cannot see this detector's
                                                  evidence (and DR/PER show nothing either)
    otherwise                  -> MORE_DATA       confidence intervals still too wide
    MORE_DATA at the last step -> UNDECIDED       stop and discuss before spending more
"""

from __future__ import annotations

import hashlib
import json
import math
import os

PASS, FAIL, INCONCLUSIVE, FAITHFUL, SKIP = "PASS", "FAIL", "INCONCLUSIVE", "FAITHFUL", "SKIP"

ACTIONS = {
    "GO": ("继续：发现成立", "这个检测器的解释存在可测量的不忠实。按计划扩大样本量、加入更多检测器。"),
    "PIVOT": ("转向：解释是忠实的", "这个检测器的解释通过了还原检验。先在第二个检测器上复核；如果多数检测器都忠实，"
              "论文按预案改写成“哪些训练方式带来忠实性”的对比研究。"),
    "SWITCH_DETECTOR": ("换检测器", "这个检测器在这批数据上几乎分不清真假，它的解释没有审计的意义。"
                        "不要继续跑它：换成专用鉴伪模型（如 SIDA），或先确认数据没有问题。"),
    "NOT_INFORMATIVE": ("此检测器上协议无效", "连“标准答案区域”都改变不了它的判断，说明它根本不看篡改区。"
                        "DR/PER 也没有给出发现。记录下来，换下一个检测器；如果多个检测器都这样，停下来和我讨论。"),
    "FIX_PROMPT": ("先修解析", "太多回答无法解析成判决和区域。先改提示词或解析器，再在同一批样本上重跑。"),
    "MORE_DATA": ("需要更多样本", "置信区间还太宽，无法判断。自动进入下一档样本量。"),
    "NO_FLAGS": ("检测器几乎从不判假", "它对假图几乎全部回答“真”，所以给不出可审计的证据区域（审计的对象是"
                 "“判假并指出区域”的解释）。不要继续加样本：换成专门训练过的鉴伪 MLLM（如 SIDA）。"
                 "若 AUC 不低，说明它的分数里有信号，只是阈值偏向“真”，可以记录为一个现象。"),
    "UNDECIDED": ("到上限仍未定", "已经跑到最大样本量还是判断不了。停下来，把报告发给我一起看，不要盲目加大投入。"),
}


def config_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ci(m, key):
    v = m.get(key)
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return (float("nan"),) * 3
    return tuple(float("nan") if x is None else float(x) for x in v)


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _pick_base(model: dict, min_n: int) -> str:
    bb = model.get("by_base", {})
    for b in ("fake", "raw"):
        if bb.get(b, {}).get("n", 0) >= min_n:
            return b
    return max(bb, key=lambda b: bb[b].get("n", 0)) if bb else "fake"


def _wilson_hi(p: float, n: int, z: float = 1.96) -> float:
    """Upper end of the 95% Wilson interval for a proportion p observed on n items."""
    if n <= 0:
        return float("nan")
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    return (c + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def evaluate(results: dict, cfg: dict) -> dict:
    """results: {"model": metrics_json, "oracle": metrics_json, "random": metrics_json}
    (oracle/random optional). Returns gate statuses, evidence and the decision."""
    model = results["model"]
    s1 = model.get("stage1", {})
    base = _pick_base(model, cfg["min_n"])
    mb = model.get("by_base", {}).get(base, {})
    n = mb.get("n", 0)
    gates = {}

    # G0 data & parsing
    pf = s1.get("parse_fail_rate", float("nan"))
    if not _nan(pf) and pf > cfg["max_parse_fail"]:
        gates["G0"] = (FAIL, f"解析失败率 {pf:.1%} > {cfg['max_parse_fail']:.0%}")
    elif n < cfg["min_n"]:
        gates["G0"] = (INCONCLUSIVE, f"可审计样本 n={n} < {cfg['min_n']}（模型判为“假”且给出区域的样本太少）")
    else:
        gates["G0"] = (PASS, f"解析失败率 {0 if _nan(pf) else pf:.1%}，可审计样本 n={n}（底图 {base}）")

    # flag rate: an MLLM that (almost) never answers "fake" produces no evidence to audit
    n_fake = max(s1.get("n_fake", 0), s1.get("n_raw", 0))
    acc = [x for x in (s1.get("acc_fake"), s1.get("acc_raw")) if not _nan(x)]
    flag_rate = max(acc) if acc else float("nan")
    flag_hi = _wilson_hi(flag_rate, n_fake) if not _nan(flag_rate) else float("nan")
    no_flags = (n_fake >= cfg.get("min_flag_n", 20) and not _nan(flag_hi) and flag_hi < cfg.get("min_flag_rate", 0.20))

    # G1 detection
    aucs = [x for x in (s1.get("auc_fake_vs_real"), s1.get("auc_raw_vs_real")) if not _nan(x)]
    auc = max(aucs) if aucs else float("nan")
    n_real = s1.get("n_real", 0)
    if not _nan(auc) and auc >= cfg["detect_auc_pass"] and n_real >= cfg.get("detect_min_n_for_pass", 30):
        gates["G1"] = (PASS, f"AUC={auc:.3f} ≥ {cfg['detect_auc_pass']}（n_real={n_real}）")
    elif not _nan(auc) and auc <= cfg["detect_auc_fail"] and n_real >= cfg["detect_min_n_for_fail"]:
        gates["G1"] = (FAIL, f"AUC={auc:.3f} ≤ {cfg['detect_auc_fail']}（n_real={n_real}）")
    else:
        gates["G1"] = (INCONCLUSIVE, f"AUC={auc:.3f}，n_real={n_real}")

    # G2 protocol validity
    o_res, r_res = results.get("oracle"), results.get("random")
    o = _ci(o_res.get("by_base", {}).get(base, {}), "CNG") if o_res else (float("nan"),) * 3
    r = _ci(r_res.get("by_base", {}).get(base, {}), "CNG") if r_res else (float("nan"),) * 3
    if o_res is None or r_res is None:
        gates["G2"] = (SKIP, "没有 oracle/random 对照结果")
    elif any(_nan(x) for x in (o[0], r[0])):
        gates["G2"] = (INCONCLUSIVE, "对照样本不足")
    elif n < cfg["min_n"]:
        gates["G2"] = (INCONCLUSIVE, f"oracle {o[0]:.3f} vs random {r[0]:.3f}，但只有 n={n} 个可审计样本（< {cfg['min_n']}）")
    elif o[1] > r[2] and o[0] - r[0] >= cfg["protocol_min_gap"]:
        gates["G2"] = (PASS, f"oracle CNG {o[0]:.3f} [{o[1]:.3f},{o[2]:.3f}] 明显高于 random {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]")
    elif o[2] < r[0] + cfg["protocol_min_gap"] and n >= cfg["min_n"]:
        gates["G2"] = (FAIL, f"oracle CNG 上界 {o[2]:.3f} 没有超过 random {r[0]:.3f} + {cfg['protocol_min_gap']}")
    else:
        gates["G2"] = (INCONCLUSIVE, f"oracle {o[0]:.3f} vs random {r[0]:.3f}，区间仍重叠")

    # G3 finding
    m = _ci(mb, "CNG")
    dr = _ci(model, "DR")
    dr_n = model.get("DR_n_eligible", 0)
    pr = _ci(mb, "PER")
    ratio = float("nan")
    if gates["G2"][0] == PASS and not _nan(m[0]) and o[0] - r[0] > 1e-9:
        ratio = (m[0] - r[0]) / (o[0] - r[0])
    routes = []
    if not _nan(ratio) and ratio <= cfg["cng_ratio_finding"] and m[2] < o[1]:
        routes.append(f"忠实度比例 R={ratio:.0%}（模型 CNG {m[0]:.3f}，oracle {o[0]:.3f}，random {r[0]:.3f}）："
                      f"拿掉它指的区域，效果只有拿掉真实篡改区的 {ratio:.0%}")
    if not _nan(dr[0]) and dr_n >= cfg["min_n"] and dr[0] >= cfg["dr_finding"] and dr[1] >= cfg["dr_finding_lo"]:
        routes.append(f"解耦率 DR={dr[0]:.2f} [{dr[1]:.2f},{dr[2]:.2f}]（n={dr_n}）：嘴上指着篡改区，判决却靠全局痕迹")
    if not _nan(pr[0]) and n >= cfg["min_n"] and pr[0] >= cfg["per_finding"] and pr[1] >= cfg["per_finding_lo"]:
        routes.append(f"幽灵证据率 PER={pr[0]:.2f} [{pr[1]:.2f},{pr[2]:.2f}]：已换成真像素的区域仍被指为伪影")
    faithful = (not _nan(ratio) and ratio >= cfg["cng_ratio_faithful"]
                and (_nan(dr[2]) or dr[2] < cfg["dr_faithful_max"]) and (_nan(pr[2]) or pr[2] < cfg["per_faithful_max"]))
    if routes:
        gates["G3"] = (PASS, "；".join(routes))
    elif faithful:
        gates["G3"] = (FAITHFUL, f"忠实度比例 R={ratio:.0%} ≥ {cfg['cng_ratio_faithful']:.0%}，DR 与 PER 都低")
    else:
        gates["G3"] = (INCONCLUSIVE, f"忠实度比例 R={'n/a' if _nan(ratio) else f'{ratio:.0%}'}，DR {dr[0]:.2f}（n={dr_n}），PER {pr[0]:.2f}")

    st = {k: v[0] for k, v in gates.items()}
    if st["G0"] == FAIL:
        decision = "FIX_PROMPT"
    elif no_flags:
        decision = "NO_FLAGS"
    elif st["G1"] == FAIL:
        decision = "SWITCH_DETECTOR"
    elif st["G0"] != PASS or st["G1"] != PASS:
        # no finding counts until there are enough samples AND the detector can
        # actually detect these fakes (a blind detector's "unfaithfulness" is noise)
        decision = "MORE_DATA"
    elif st["G3"] == PASS:
        decision = "GO"
    elif st["G3"] == FAITHFUL:
        decision = "PIVOT"
    elif st["G2"] == FAIL:
        decision = "NOT_INFORMATIVE"
    else:
        decision = "MORE_DATA"
    return {"base": base, "n": n, "faithfulness_ratio": ratio, "flag_rate": flag_rate, "flag_rate_hi": flag_hi, "gates": {k: {"status": v[0], "evidence": v[1]} for k, v in gates.items()},
            "decision": decision, "action": ACTIONS[decision]}


def load_results(results_dir: str, tag: str, explainers=("model", "oracle", "random")) -> dict:
    out = {}
    for ex in explainers:
        p = os.path.join(results_dir, f"{tag}_{ex}", "metrics.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[ex] = json.load(f)
    if "model" not in out:
        raise FileNotFoundError(f"no {tag}_model/metrics.json under {results_dir}")
    return out
