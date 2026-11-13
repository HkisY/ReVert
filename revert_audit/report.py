"""Self-contained Chinese HTML report: what the numbers mean, the gate
decisions, and real examples (images embedded), so a run can be judged by
looking at one page.

    python -m revert_audit.cli report --run runs/pilot --samples data/pilot/samples.jsonl --tag qwen25vl7b
"""

from __future__ import annotations

import base64
import html
import json
import math
import os

import numpy as np

from .io import load_mask, load_rgb, read_jsonl
from .pipeline import qid

COLORS = {"PASS": "#1a7f37", "FAIL": "#cf222e", "INCONCLUSIVE": "#9a6700", "FAITHFUL": "#0969da", "SKIP": "#6e7781"}
GATE_NAMES = {"G0": "数据与解析", "G1": "检测能力", "G2": "协议有效性", "G3": "有无发现"}
GATE_MEANING = {
    "G0": "流水线是否产出了足够多、可解析的回答。",
    "G1": "检测器能不能把假图和真图分开。分不开的话，它的解释没有审计意义。",
    "G2": "还原检验在这个检测器上是否灵敏：用“标准答案区域”（真实篡改区）做还原，必须比随机区域明显更能改变判断。",
    "G3": "解释是否存在可测量的不忠实（或者反过来，是否已经很忠实）。",
}
METRICS = [
    ("CNG", "必要性差", "把模型指出的区域换回真像素后，“假”分数下降了多少（已减去同形随机区域的下降）。",
     "越接近 oracle 越忠实；接近 random（≈0）说明它指的地方根本不是判决依据。"),
    ("CSG", "充分性差", "只保留模型指出的区域、其余全部换回真像素时，“假”分数比保留随机区域高多少。",
     "越高说明它指的证据单独就足以让它判假。"),
    ("GP", "定位精度", "模型指的区域有多大比例落在真实篡改区内。",
     "篡改区外的部分像素就是原图，指到那里是可以被证明的编造。"),
    ("PER", "幽灵证据率", "把它指的区域换成真像素后再问一次，它仍然指着同一处说有伪影的比例。",
     "越高说明解释越像“套话”，与图像内容无关。"),
]


def _b64(arr: np.ndarray, h: int = 220) -> str:
    import cv2

    s = h / arr.shape[0]
    t = cv2.resize(arr, (max(1, int(arr.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", t[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _outline(img, mask, color, th):
    import cv2

    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, th)
    return out


def _fmt(v, pct=False):
    if not isinstance(v, (list, tuple)):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{v:.1%}" if pct else f"{v:.3f}"
    p, lo, hi = v
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{p:.3f}"
    return f"{p:.3f}<br><small>[{lo:.3f}, {hi:.3f}]</small>"


def _examples(run, tag, base, samples, k=3):
    rec_p = os.path.join(run, "results", f"{tag}_model", "records.jsonl")
    recs = [r for r in read_jsonl(rec_p) if r["base"] == base and r.get("s_revCt")]
    if not recs:
        return ""
    for r in recs:
        r["_term"] = float(np.mean(r["s_revCt"]) - r["s_revC"])
    recs.sort(key=lambda r: r["_term"])
    def label(t):
        return "解释是判决依据" if t >= 0.2 else ("解释没起作用" if t <= 0.05 else "解释有一点作用")

    low = recs[:k]
    high = [r for r in recs[::-1][:k] if r not in low]
    picks = [(label(r["_term"]), r) for r in low + high]
    a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
    s2 = os.path.join(run, f"stage2_{tag}_model")
    cards = []
    for label, r in picks:
        sid = r["sample_id"]
        s = samples.get(sid)
        bp = os.path.join(run, "stage1", "images", f"{sid}__{base}.png")
        cp = os.path.join(s2, "regions", f"{sid}__{base}__C.png")
        rp = os.path.join(s2, "images", f"{sid}__{base}__revC0.png")
        if not (s and os.path.exists(bp) and os.path.exists(cp) and os.path.exists(rp)):
            continue
        img, M, C, rv = load_rgb(bp), load_mask(s["mask"]), load_mask(cp), load_rgb(rp)
        th = max(2, img.shape[0] // 150)
        shown = _outline(_outline(img, M, (0, 200, 0), th), C, (230, 0, 0), th)
        ans = a1.get(qid(sid, 1, base), {})
        text = html.escape((ans.get("raw_text") or "（无文字解释）")[:400])
        cards.append(f"""<div class="card"><div class="tag">{label}</div>
<div class="imgs"><figure><img src="{_b64(shown)}"><figcaption>{base}：绿=真实篡改区，红=模型指的区域</figcaption></figure>
<figure><img src="{_b64(rv)}"><figcaption>把红色区域换回真像素后</figcaption></figure></div>
<p>“假”分数：原图 <b>{r['s0']:.2f}</b> → 还原红区 <b>{r['s_revC']:.2f}</b>；还原同形随机区域平均 <b>{np.mean(r['s_revCt']):.2f}</b>；
必要性项 = {r['_term']:+.2f}；GP = {r['gp']:.2f}</p>
<p class="expl">{sid} · 模型原话：{text}</p></div>""")
    return "\n".join(cards)


def build_report(run: str, samples_path: str, tag: str, gate_result: dict, cfg_hash: str, budget: dict | None = None,
                 out_path: str | None = None) -> str:
    res = {}
    for ex in ("model", "oracle", "random"):
        p = os.path.join(run, "results", f"{tag}_{ex}", "metrics.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                res[ex] = json.load(f)
    base = gate_result["base"]
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    title, advice = gate_result["action"]
    dcol = {"GO": "#1a7f37", "PIVOT": "#0969da", "MORE_DATA": "#9a6700"}.get(gate_result["decision"], "#cf222e")

    gate_rows = "".join(
        f"<tr><td>{g} {GATE_NAMES[g]}</td><td><span class='pill' style='background:{COLORS[v['status']]}'>{v['status']}</span></td>"
        f"<td>{html.escape(v['evidence'])}</td><td class='muted'>{GATE_MEANING[g]}</td></tr>"
        for g, v in gate_result["gates"].items())

    def cell(ex, key):
        m = res.get(ex, {}).get("by_base", {}).get(base, {})
        return _fmt(m.get(key)) if m else "n/a"

    metric_rows = "".join(
        f"<tr><td><b>{k}</b> {zh}</td><td>{cell('model', k)}</td><td>{cell('oracle', k)}</td><td>{cell('random', k)}</td>"
        f"<td class='muted'>{what}<br><i>{how}</i></td></tr>" for k, zh, what, how in METRICS)
    s1 = res.get("model", {}).get("stage1", {})
    dr = res.get("model", {}).get("DR")
    other = (f"<tr><td><b>DR</b> 解耦率</td><td colspan='3'>{_fmt(dr)}（可判定样本 n={res.get('model', {}).get('DR_n_eligible', 0)}）</td>"
             "<td class='muted'>在原始修复图上判“假”且指向篡改区的样本里，把背景换回原图后改判“真”的比例。<br>"
             "<i>越高说明“嘴上说局部，实际靠全局痕迹”。</i></td></tr>")
    det = " · ".join(f"{k} = {_fmt(v, pct=k.startswith('acc') or k == 'parse_fail_rate') if isinstance(v, float) else v}"
                     for k, v in s1.items())
    bud = ""
    if budget:
        bud = (f"<h2>算力账</h2><p>到目前为止用了 <b>{budget['used_h']:.1f}</b> GPU 小时。"
               + (f"下一档（{budget['next_n']} 个样本）预计再用 <b>{budget['next_h']:.1f}</b> 小时。" if budget.get("next_n") else "")
               + "</p>")
    ex_html = _examples(run, tag, base, samples)
    rt = gate_result.get("faithfulness_ratio")
    ratio_line = ("" if rt is None or (isinstance(rt, float) and math.isnan(rt)) else
                  f"<br>忠实度比例 R = <b>{rt:.0%}</b>（100% = 和“标准答案区域”一样有用，0% = 和随机区域一样没用）")
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>ReVert 试点报告 · {html.escape(tag)}</title>
<style>
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328;background:#fff}}
h1{{font-size:22px}} h2{{font-size:18px;margin-top:32px;border-bottom:1px solid #d0d7de;padding-bottom:4px}}
.banner{{border-left:6px solid {dcol};background:#f6f8fa;padding:12px 16px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border:1px solid #d0d7de;padding:6px 8px;vertical-align:top;text-align:left}}
th{{background:#f6f8fa}} .muted{{color:#57606a;font-size:13px}} .pill{{color:#fff;border-radius:10px;padding:2px 8px;font-size:12px}}
.card{{border:1px solid #d0d7de;border-radius:8px;padding:10px;margin:12px 0}} .tag{{font-weight:600;margin-bottom:6px}}
.imgs{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:0}} figcaption{{font-size:12px;color:#57606a}}
.expl{{font-size:12px;color:#57606a;word-break:break-all}} img{{max-width:100%;height:auto}}
</style></head><body>
<h1>ReVert 试点报告 · {html.escape(tag)}</h1>
<div class="banner"><b>决定：{gate_result['decision']} · {title}</b><br>{advice}{ratio_line}</div>
<p class="muted">审计底图：{base}（fake = 只有篡改区是生成像素；raw = 编辑器原始输出，整张图都过了 VAE）· 可审计样本 n = {gate_result['n']} ·
关卡配置哈希 {cfg_hash}（预注册，跑之前固定，之后不改）</p>
<h2>关卡</h2><table><tr><th>关卡</th><th>状态</th><th>依据</th><th>它检查什么</th></tr>{gate_rows}</table>
<h2>指标：模型的解释 vs 标准答案（oracle）vs 随机区域（random）</h2>
<p class="muted">oracle 是“如果解释完全正确”能达到的水平，random 是“解释完全随便指”的水平。模型的数字落在两者之间的哪里，就是它的解释有多忠实。方括号是 95% bootstrap 置信区间。</p>
<table><tr><th>指标</th><th>模型的解释</th><th>oracle</th><th>random</th><th>含义 / 怎么读</th></tr>{metric_rows}{other}</table>
<p class="muted">检测表现：{det}</p>
{bud}
<h2>典型案例</h2>
<p class="muted">按“必要性项”（随机区域还原后的分数 − 模型所指区域还原后的分数）排序：前几个是这一项最小的样本，后几个是最大的样本。标签按数值给出：≥ 0.2 解释是判决依据，≤ 0.05 解释没起作用。</p>
{ex_html or '<p>暂无案例（模型判为“假”且给出区域的样本还没有）。</p>'}
</body></html>"""
    out = out_path or os.path.join(run, f"report_{tag}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out
