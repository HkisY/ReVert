"""Figure 3: per-inpainting-model flag rate, faithfulness ratio R (clean controls) and phantom evidence rate
PER for both detectors on the single-object set, points with 95% intervals.

Everything is recomputed from the stored records (the same ones diagnose.py reads), so the numbers match
Table 2: R uses only records whose shape-matched controls stay off the edit (overlap <= 0.1), exactly as
R_clean in diagnose.py. Intervals: flag rate = Wilson; PER = bootstrap over records; R = paired bootstrap
over images (the model, oracle and random explanations of the same images are resampled together).

    python scripts/make_fig3.py                      # writes figures/fig3_per_editor.pdf and .png
    python scripts/make_fig3.py --layout column      # one-column version: figures/fig3_per_editor_column.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from revert_audit.detectors.base import answer_verdict  # noqa: E402
from revert_audit.diagnose import _clean, load_explainer  # noqa: E402
from revert_audit.io import read_jsonl  # noqa: E402
from revert_audit.pipeline import qid  # noqa: E402

EDITORS = [("sdxl", "SDXL"), ("flux-fill", "FLUX.1-Fill"), ("kandinsky", "Kandinsky 2.2")]
DETECTORS = [("SIDA-7B", "runs/full_sida", "sida7b", (0.84, 0.15, 0.16), "o"),
             ("FakeShield", "runs/full_fakeshield", "fakeshield", (0.12, 0.47, 0.71), "s")]
TEXTWIDTH_IN = 6.875
COLUMNWIDTH_IN = 3.25  # CVPR column width is about 3.28 in
FONTS = ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


def boot_mean(vals, n=2000, seed=0):
    a = np.asarray([v for v in vals if v is not None and v == v], float)
    if not len(a):
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def r_clean(recs, n=2000, seed=0):
    """R = (CNG_model - CNG_random) / (CNG_oracle - CNG_random) on clean-control records; paired bootstrap
    over image ids."""
    by = {ex: {r["sample_id"]: r["nec"] for r in rs if _clean(r)} for ex, rs in recs.items()}
    ids = sorted(set(by["model"]) | set(by["oracle"]) | set(by["random"]))

    def ratio(sel):
        m = [by["model"][i] for i in sel if i in by["model"]]
        o = [by["oracle"][i] for i in sel if i in by["oracle"]]
        r = [by["random"][i] for i in sel if i in by["random"]]
        if not (m and o and r):
            return float("nan")
        den = np.mean(o) - np.mean(r)
        return float((np.mean(m) - np.mean(r)) / den) if den > 1e-9 else float("nan")

    point = ratio(ids)
    rng = np.random.default_rng(seed)
    arr = np.asarray(ids)
    bs = [ratio(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
    bs = [b for b in bs if b == b]
    return [point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else [point, np.nan, np.nan]


def collect(run, tag, samples_path):
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
    recs = {ex: [r for r in load_explainer(run, tag, ex, samples) if r["base"] == "fake"]
            for ex in ("model", "oracle", "random")}
    out = {}
    for ed, _ in EDITORS:
        ids = [sid for sid, s in samples.items() if s.get("editor") == ed]
        ans = [a1.get(qid(sid, 1, "fake")) for sid in ids]
        ans = [a for a in ans if a is not None]
        k = sum(answer_verdict(a) == "fake" for a in ans)
        E = {ex: [r for r in rs if samples.get(r["sample_id"], {}).get("editor") == ed] for ex, rs in recs.items()}
        out[ed] = {"flag": wilson(k, len(ans)), "n_samples": len(ans), "n_records": len(E["model"]),
                   "R": r_clean(E), "PER": boot_mean([r.get("per") for r in E["model"]])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="data/full_single/samples.jsonl")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--sida-run", default="runs/full_sida",
                    help="run folder of SIDA (or outputs/single_sida of a folder expanded by reproduce_tables.py --keep)")
    ap.add_argument("--fs-run", default="runs/full_fakeshield")
    ap.add_argument("--width", type=float, default=TEXTWIDTH_IN)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--layout", choices=["row", "column"], default="row",
                    help="row: three panels side by side (two-column figure*); column: stacked, for one column")
    a = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in FONTS if f in have] or ["serif"]
    plt.rcParams.update({"font.family": "serif", "font.serif": fam, "font.size": 7, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

    runs = {"SIDA-7B": a.sida_run, "FakeShield": a.fs_run}
    data = {name: collect(runs.get(name, run), tag, a.samples) for name, run, tag, _, _ in DETECTORS}
    plot(data, a, plt, fam)


def plot(data, a, plt, fam):
    print(f"{'detector':<11} {'editor':<14} {'n':>4} {'rec':>4}  flag [95% CI]          R (clean) [95% CI]      PER [95% CI]")
    for name, *_ in DETECTORS:
        for ed, lab in EDITORS:
            d = data[name][ed]
            f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
            print(f"{name:<11} {lab:<14} {d['n_samples']:>4} {d['n_records']:>4}  {f(d['flag'])}  {f(d['R'])}  {f(d['PER'])}")

    col = a.layout == "column"
    width = a.width if a.width != TEXTWIDTH_IN or not col else COLUMNWIDTH_IN
    height = a.height or (3.1 if col else 1.75)
    if col:
        fig, axes = plt.subplots(3, 1, figsize=(width, height), sharex=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(width, height))
    panels = [("flag", "Flag rate on forgeries"), ("R", "Faithfulness ratio R (clean controls)"),
              ("PER", "Phantom evidence rate PER")]
    xs = np.arange(len(EDITORS))
    for i, (ax, (key, title)) in enumerate(zip(axes, panels)):
        for j, (name, _, _, color, mk) in enumerate(DETECTORS):
            off = (j - 0.5) * 0.22
            v = np.array([data[name][ed][key] for ed, _ in EDITORS], float)
            ax.errorbar(xs + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=3.6, color=color,
                        mfc=color if mk == "o" else "white", mew=0.9, elinewidth=0.8, capsize=1.8, label=name)
        ax.set_xticks(xs)
        if not col or i == len(panels) - 1:
            ax.set_xticklabels([lab for _, lab in EDITORS], fontsize=6.5)
        if col:  # title inside the panel, top left, to save height
            ax.text(0.01, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        else:
            ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlim(-0.5, len(EDITORS) - 0.5)
        lo = min(0.0, np.nanmin([data[n][e][key][1] for n, *_ in DETECTORS for e, _ in EDITORS]))
        hi = max(1.0, np.nanmax([data[n][e][key][2] for n, *_ in DETECTORS for e, _ in EDITORS]))
        top = hi + (0.03 if hi > 1.0 else 0.0) + (0.22 if col else 0.0)  # column: room for the in-panel title
        ax.set_ylim(lo - 0.02, top)
        ax.set_yticks([0, 0.5, 1.0] if col else [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        if key == "R":  # R = 1: the explanation carries as much decision weight as the true edit
            ax.axhline(1, color="0.55", lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.1,
                   ncol=2 if col else 1, columnspacing=0.8)
    fig.tight_layout(pad=0.3, w_pad=1.2, h_pad=0.4)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "fig3_per_editor" + ("_column" if col else ""))
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    print(f"[fig3] font: {fam[0]}; wrote {base}.pdf and .png")


if __name__ == "__main__":
    main()
