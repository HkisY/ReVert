"""Figure 4 (qualitative examples): four real cases, each shown before and after ReVert restores the
region the model points at. Two cases where the explanation follows the evidence, two where it does not.

    (a) faithful      FakeShield, single object: it points at the edit; once that region is restored
                      the verdict flips to real.
    (b) follows       FakeShield, two objects: object A is restored as a whole; FakeShield still says
                      fake and now points at B, the object that is still forged.
    (c) phantom       SIDA, two objects: A is restored as a whole; SIDA still says tampered and still
                      points at A, now authentic, not at the remaining forgery B.
    (d) off the edit  SIDA, single object: it points at pixels that were never edited; restoring them
                      removes almost no forged pixels and SIDA keeps pointing there.

Visual language (kept minimal on purpose): the model's mask is shown as a spotlight, i.e. the image is
dimmed everywhere except where the model points, so the pointed-at content keeps its real colours and the
reader can see what it is. Forged regions have a dashed white outline, regions ReVert restored to their
original pixels a solid white outline. The verdict is a coloured tag (red = fake, green = real).

Everything shown is an experiment output: the queried images (x_f and the reverted image that was
actually sent to the model), the models' own masks (never smoothed), verdicts and fake scores. Photos are
only cropped and embedded unresampled. Before drawing, the script checks pixel by pixel that each
reverted image equals R(x_f, region) = region * x_r + (1 - region) * x_f.

Step 1, list candidates (figures/fig4_candidates.csv and figures/fig4_candidates_{a,b,c,d}.png):
    python scripts/make_fig4.py --list
Supplementary figure: one row per panel rule, K randomly drawn qualifying cases per row (seeded; ids and
the number of qualifying cases per rule go to fig4_random_seed<S>.csv and fig4_random_seed<S>_pools.csv):
    python scripts/make_fig4.py --random 3 --seed 0      -> figures/fig4_random_seed0.pdf/.png
Step 2, draw (each --pick-* defaults to the top-ranked candidate; the teaser sample is excluded from (c)):
    python scripts/make_fig4.py [--pick-a ID] [--pick-b ID] [--pick-c ID] [--pick-d ID]

Selection is for illustration only (no statistic is computed from it). Rules:
    (a) x_f flagged, coverage of M >= 0.5, GP >= 0.7, after: verdict real.          Rank: score drop.
    (b) x_f flagged on A (>= 0.3 of A, < 0.1 of B); after: fake, >= 0.3 of B, < 0.1 of A.  Rank: area.
    (c) the teaser rule (make_teaser.sida_ok).                                       Rank: area of A.
    (d) x_f flagged, GP <= 0.1, coverage of M <= 0.1, PER = 1.                       Rank: score, mask area.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from make_teaser import TEXTWIDTH_IN, Run, contour, cov, edit_desc, setup_mpl, sida_case, sida_ok  # noqa: E402
from revert_audit.detectors.base import answer_score, answer_verdict  # noqa: E402
from revert_audit.io import load_mask, load_rgb, read_jsonl  # noqa: E402
from revert_audit.pipeline import answer_region, qid  # noqa: E402
from revert_audit.regions import overlap_frac  # noqa: E402

TEASER_SID = "coco442836p_flux-fill"
NAME = {"sida7b": "SIDA", "fakeshield": "FakeShield"}
WHITE = (1.0, 1.0, 1.0)
FAKE_FC, FAKE_TC = "#fbe3e1", "#b2182b"   # verdict tags
REAL_FC, REAL_TC = "#e2f2e3", "#1b7837"
GOOD, BAD = "#1b7837", "#b2182b"           # group headers: explanation follows / does not follow evidence


# ------------------------------------------------------------------ data
def _ans(a, shape):
    if a is None:
        return None
    return {"mask": answer_region(a, *shape), "s": answer_score(a), "v": answer_verdict(a)}


class Single:
    """Single-object run: x_f, the model's region C and the answer on R(x_f, C)."""

    def __init__(self, run, tag, samples):
        self.run, self.tag, self.name = run, tag, NAME.get(tag, tag)
        self.s2 = os.path.join(run, f"stage2_{tag}_model")
        self.samples = samples
        self.a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
        self.a2 = {a["query_id"]: a for a in read_jsonl(os.path.join(self.s2, "answers.jsonl"))}
        self.meta = {m["sample_id"]: m for m in read_jsonl(os.path.join(self.s2, "meta.jsonl"))
                     if m.get("base") == "fake" and not m.get("skipped")}

    def case(self, sid):
        m, s = self.meta.get(sid), self.samples.get(sid)
        if m is None or s is None or not m.get("cmask"):
            return None
        M, C = load_mask(s["mask"]), load_mask(m["cmask"])
        x0 = _ans(self.a1.get(qid(sid, 1, "fake")), M.shape)
        x1 = _ans(self.a2.get(qid(sid, 2, "fake", "revC", 0)), M.shape)
        if x0 is None or x1 is None:
            return None
        per = bool(x1["v"] == "fake" and x1["mask"].any()
                   and (x1["mask"] & C).sum() / max(1, (x1["mask"] | C).sum()) >= 0.5)
        return {"sid": sid, "kind": "single", "det": self.name, "editor": s.get("editor"), "M": M, "C": C,
                "x0": x0, "x1": x1, "gp": overlap_frac(C, M), "cov": cov(C, M), "per": per,
                "drop": x0["s"] - x1["s"], "edit": edit_desc(s),
                "xf": os.path.join(self.run, "stage1", "images", f"{sid}__fake.png"),
                "x1p": os.path.join(self.s2, "images", f"{sid}__fake__revC0.png"), "xr": s["real"]}


def faithful_ok(k):
    return (k["x0"]["v"] == "fake" and k["cov"] >= 0.5 and k["gp"] >= 0.7 and k["x1"]["v"] == "real"
            and k["M"].mean() >= 0.03)


def offedit_ok(k):
    return (k["x0"]["v"] == "fake" and k["gp"] <= 0.1 and k["cov"] <= 0.1 and k["per"]
            and k["C"].mean() >= 0.01)


def comp_case(R, sid, name):
    """Two-object case in the same format as make_teaser.sida_case, for any detector."""
    k = sida_case(R, sid)
    if k is None:
        return None
    k.update({"kind": "comp", "det": name, "x0": k["sida0"], "x1": k["sida1"], "x1p": k["rc"]})
    return k


def follows_ok(k):
    x0, x1, A, B = k["x0"], k["x1"], k["A"], k["B"]
    return (x0["v"] == "fake" and cov(x0["mask"], A) >= 0.3 and cov(x0["mask"], B) < 0.1
            and x1["v"] == "fake" and cov(x1["mask"], B) >= 0.3 and cov(x1["mask"], A) < 0.1
            and A.mean() >= 0.02 and B.mean() >= 0.02)


def check_pixels(k):
    """The reverted image must equal region * x_r + (1 - region) * x_f, exactly."""
    xf, x1, xr = load_rgb(k["xf"]), load_rgb(k["x1p"]), load_rgb(k["xr"])
    G = k["C"] if k["kind"] == "single" else k["A"]
    if not (np.array_equal(x1[G], xr[G]) and np.array_equal(x1[~G], xf[~G])):
        raise SystemExit(f"{k['sid']}: reverted image is not R(x_f, region) pixel for pixel; not drawing it")
    return xf, x1


# ------------------------------------------------------------------ candidates
def collect(a):
    samples = {s["id"]: s for s in read_jsonl(a.samples)}
    fs1, sida1 = Single(a.single_fs, "fakeshield", samples), Single(a.single_sida, "sida7b", samples)
    Rs, Rf = Run(a.pairs_sida, "sida7b"), Run(a.pairs_fs, "fakeshield")
    out = {"a": [], "b": [], "c": [], "d": []}
    for sid in fs1.meta:
        k = fs1.case(sid)
        if k and faithful_ok(k):
            out["a"].append((k["drop"], k))
    for sid in Rf.meta:
        k = comp_case(Rf, sid, "FakeShield")
        if k and follows_ok(k):
            out["b"].append((float(min(k["A"].mean(), k["B"].mean())), k))
    for sid in Rs.meta:
        if sid == TEASER_SID:
            continue
        k = comp_case(Rs, sid, "SIDA")
        if k and sida_ok(k, a.min_area, 0.5):
            out["c"].append((float(k["A"].mean()), k))
    for sid in sida1.meta:
        k = sida1.case(sid)
        if k and offedit_ok(k):
            out["d"].append((k["x0"]["s"] + k["C"].mean(), k))
    return {p: [k for _, k in sorted(v, key=lambda t: -t[0])] for p, v in out.items()}


def write_candidates(cands, a, plt, prefix="fig4_candidates", ext=("png",)):
    """Contact sheets, one per panel: top-ranked candidates (--list) or a seeded random draw of
    qualifying cases (--random K, meant for Supp. G: examples not chosen by hand)."""
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{prefix}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "rank", "sample_id", "detector", "editor", "s_before", "s_after"])
        for p, ks in cands.items():
            for i, k in enumerate(ks[:a.top]):
                w.writerow([p, i, k["sid"], k["det"], k["editor"], f"{k['x0']['s']:.3f}", f"{k['x1']['s']:.3f}"])
    for p, ks in cands.items():
        ks = ks[:a.top]
        print(f"[fig4] ({p}) {PANELS[p][1]}: {len(cands[p])} candidates"
              + (f"; top: {', '.join(k['sid'] for k in ks[:5])}" if ks else ""))
        if not ks:
            continue
        fig, axes = plt.subplots(2, len(ks), figsize=(1.6 * len(ks), 3.5), squeeze=False)
        for j, k in enumerate(ks):
            xf, x1 = load_rgb(k["xf"]), load_rgb(k["x1p"])
            crop = crop_of(k)
            for r, (img, ans) in enumerate(((xf, k["x0"]), (x1, k["x1"]))):
                ax = axes[r][j]
                draw_image(ax, img, ans, k, after=bool(r), crop=crop)
                ax.set_title((f"{j}: {k['sid']}\n" if r == 0 else "") + f"{ans['v']} {ans['s']:.2f}", fontsize=5)
        fig.tight_layout(pad=0.2)
        for e in ext:
            fig.savefig(os.path.join(a.out, f"{prefix}_{p}.{e}"), dpi=150 if e == "png" else None)
        plt.close(fig)


# ------------------------------------------------------------------ drawing
def crop_of(k, margin=0.10):
    """Square crop around everything the panel talks about (edit, pointed-at regions)."""
    regions = [k["M"], k["C"]] if k["kind"] == "single" else [k["A"], k["B"]]
    regions += [k["x0"]["mask"], k["x1"]["mask"]]
    U = np.zeros_like(regions[0])
    for X in regions:
        U |= X
    ys, xs = np.where(U)
    h, w = U.shape
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin))
    side = min(max(side, 32), h, w)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return x0, y0, x0 + side, y0 + side


def spotlight(img, mask, dim=0.32):
    """Dim and desaturate everything outside the model's mask; the mask keeps its true pixels.
    With no mask (verdict real, or nothing pointed at) the image is shown unchanged."""
    if mask is None or not mask.any():
        return img
    f = img.astype(np.float32)
    grey = f.mean(axis=2, keepdims=True)
    dark = dim * (0.45 * f + 0.55 * grey)
    out = np.where(mask[..., None], f, dark)
    return np.clip(out, 0, 255).astype(np.uint8)


def tag(ax, M, text, placed):
    """Small text tag next to a region (white box, black text). Tries above-left, below-left, above-right
    and below-right of the region and takes the first spot that stays inside the panel and does not
    overlap a tag already placed (approximate box size in crop pixels)."""
    ys, xs = np.where(M)
    if not len(ys):
        return
    h, w = M.shape
    tw, th = (0.026 * len(text) + 0.03) * w, 0.075 * h
    spots = [(xs.min(), ys.min() - 2 - th), (xs.min(), ys.max() + 2),
             (xs.max() - tw, ys.min() - 2 - th), (xs.max() - tw, ys.max() + 2)]
    best = None
    for x, y in spots:
        x = float(np.clip(x, 1, w - tw - 1))
        if y < 1 or y + th > h - 1:
            continue
        if all(x + tw < px or px + pw_ < x or y + th < py or py + ph < y for px, py, pw_, ph in placed):
            best = (x, y)
            break
    if best is None:  # nothing free: inside the region, top-left
        best = (float(np.clip(xs.min(), 1, w - tw - 1)), float(np.clip(ys.min() + 2, 1, h - th - 1)))
    placed.append((best[0], best[1], tw, th))
    ax.text(best[0], best[1], text, fontsize=5.6, color="black", ha="left", va="top", clip_on=True,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc="white", ec="none", alpha=0.9))


def draw_image(ax, img, ans, k, after, crop):
    X0, Y0, X1, Y1 = crop
    S = lambda X: X[Y0:Y1, X0:X1]
    shown = spotlight(img, ans["mask"] if ans["v"] == "fake" else None)
    ax.imshow(S(shown))
    placed = []
    if k["kind"] == "single":
        if after and k["cov"] >= 0.5:
            # most of the edit was restored: outline what ReVert restored (a thin leftover rim of the
            # edit is not outlined, it would only double the line)
            contour(ax, S(k["C"]), WHITE, lw=1.0)
            tag(ax, S(k["C"]), "restored", placed)
        else:
            # before, or a region that is (almost) all authentic pixels: restoring it changes (almost)
            # nothing, so no "restored" outline is drawn; the edit stays forged
            contour(ax, S(k["M"]), WHITE, dashed=True, lw=0.8)
            tag(ax, S(k["M"]), "forged", placed)
    else:
        if after:
            contour(ax, S(k["A"]), WHITE, lw=1.0)
            tag(ax, S(k["A"]), "A restored", placed)
        else:
            contour(ax, S(k["A"]), WHITE, dashed=True, lw=0.8)
            tag(ax, S(k["A"]), "A forged", placed)
        contour(ax, S(k["B"]), WHITE, dashed=True, lw=0.8)
        tag(ax, S(k["B"]), "B forged", placed)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def verdict_tag(fig, x, y, k, ans):
    fake = ans["v"] == "fake"
    word = ("tampered" if fake else "real") if k["det"] == "SIDA" else ("fake" if fake else "real")
    fig.text(x, y, f"{word}  {ans['s']:.2f}", fontsize=6.6, weight="bold", color=FAKE_TC if fake else REAL_TC,
             ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.4", fc=FAKE_FC if fake else REAL_FC, ec="none"))


# key: (title, short name for messages, plain-language line under the title, good/bad)
PANELS = {
    "a": ("(a) FakeShield", "faithful", "restore what it points at:\nthe verdict flips to real", True),
    "b": ("(b) FakeShield", "follows the forgery", "A restored: it now points\nat B, which is still forged", True),
    "c": ("(c) SIDA", "phantom evidence", "A restored: it still points\nat A, now authentic", False),
    "d": ("(d) SIDA", "off the edit", "points at pixels never edited; restoring\nthem changes almost nothing", False),
}


def draw(a):
    plt, fam = setup_mpl()
    cands = collect(a)
    picks = {}
    for p in "abcd":
        want = getattr(a, f"pick_{p}")
        pool = {k["sid"]: k for k in cands[p]}
        if want:
            if want not in pool:
                raise SystemExit(f"--pick-{p} {want}: does not meet the rule of panel ({p}) {PANELS[p][1]} (see --list)")
            picks[p] = pool[want]
        elif cands[p]:
            picks[p] = cands[p][0]
        else:
            raise SystemExit(f"panel ({p}) {PANELS[p][1]}: no candidate meets the rule")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    W, n = TEXTWIDTH_IN, 4
    left, gap, mid = 0.22, 0.05, 0.16          # mid: extra space between the two groups
    pw = (W - left - gap * 2 - mid) / n
    group_h, head_h, tag_h, key_h = 0.26, 0.36, 0.20, 0.17
    H = group_h + head_h + 2 * (pw + tag_h) + key_h
    fig = plt.figure(figsize=(W, H))
    fx, fy = (lambda x: x / W), (lambda y: y / H)
    xs = [left, left + pw + gap, left + 2 * pw + gap + mid, left + 3 * pw + 2 * gap + mid]

    # group headers: the explanation follows the evidence / does not
    for (x0, x1), text, col in (((xs[0], xs[1] + pw), "Explanation follows the evidence", GOOD),
                                ((xs[2], xs[3] + pw), "Explanation does not follow the evidence", BAD)):
        yb = H - group_h + 0.06
        fig.add_artist(Line2D([fx(x0), fx(x1)], [fy(yb), fy(yb)], color=col, lw=1.4))
        fig.text(fx((x0 + x1) / 2), fy(yb + 0.025), text, ha="center", va="bottom", fontsize=7.6, color=col,
                 weight="bold")

    for j, p in enumerate("abcd"):
        k = picks[p]
        xf, x1 = check_pixels(k)
        crop = crop_of(k)
        x = xs[j]
        title, short, line, good = PANELS[p]
        ytop = H - group_h
        fig.text(fx(x), fy(ytop - 0.03), f"{title}: {short}", ha="left", va="top", fontsize=7.4, weight="bold")
        fig.text(fx(x), fy(ytop - 0.15), line, ha="left", va="top", fontsize=6.2, linespacing=1.1, color="0.25")
        for r, (img, ans) in enumerate(((xf, k["x0"]), (x1, k["x1"]))):
            y = ytop - head_h - (r + 1) * pw - r * tag_h
            ax = fig.add_axes([fx(x), fy(y), fx(pw), fy(pw)])
            draw_image(ax, img, ans, k, after=bool(r), crop=crop)
            verdict_tag(fig, fx(x + 0.025), fy(y - 0.035), k, ans)

    yr0 = H - group_h - head_h - pw / 2
    for y, t in ((yr0, "Forged image"), (yr0 - pw - tag_h, "After ReVert restores\nthe pointed-at region")):
        fig.text(fx(left * 0.42), fy(y), t, rotation=90, ha="center", va="center", fontsize=6.8, linespacing=1.1)

    handles = [Patch(fc="0.85", ec="0.3", lw=0.5, label="bright: where the model points (its mask); rest dimmed"),
               Line2D([], [], color="0.3", ls=(0, (3, 2)), lw=0.9, label="forged region"),
               Line2D([], [], color="0.3", lw=1.1, label="restored to original pixels")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(fx(left + (W - left) / 2), 0.0), ncol=3,
               frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=1.4, borderaxespad=0.15)

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "fig4_examples")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    print(f"[fig4] font: {fam}; wrote {base}.pdf and .png")
    for p in "abcd":
        k = picks[p]
        extra = f"  restored {100 * k['cov']:.0f}% of the edit" if k["kind"] == "single" else ""
        print(f"       ({p}) {PANELS[p][1]:<20} {k['sid']}  {k['det']}  before {k['x0']['v']} {k['x0']['s']:.3f}"
              f" -> after {k['x1']['v']} {k['x1']['s']:.3f}{extra}")


ROW_LINE = {
    "a": "restore what it points at: the verdict flips to real",
    "b": "A restored as a whole: it now points at B, which is still forged",
    "c": "A restored as a whole: it still points at A, now authentic",
    "d": "points at pixels that were never edited; restoring them changes almost nothing",
}


def draw_grid(drawn, pools, a):
    """Supplementary figure: one row per panel rule of Fig. 4, K randomly drawn qualifying cases per row,
    each case shown as a pair (forged image | after ReVert restores the pointed-at region). Same visual
    language as Fig. 4; every reverted image is checked pixel by pixel before it is drawn."""
    plt, fam = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    K = max(len(v) for v in drawn.values())
    W = TEXTWIDTH_IN
    left = right = 0.02
    pair_gap, case_gap = 0.03, 0.16
    tw = (W - left - right - K * pair_gap - (K - 1) * case_gap) / (2 * K)   # tile width (square tiles)
    head_h, sub_h, tag_h, row_gap, key_h = 0.20, 0.13, 0.19, 0.10, 0.20
    row_h = head_h + sub_h + tw + tag_h
    H = 4 * row_h + 3 * row_gap + key_h
    fig = plt.figure(figsize=(W, H))
    fx, fy = (lambda x: x / W), (lambda y: y / H)
    for i, p in enumerate("abcd"):
        title, short, _, good = PANELS[p]
        col = GOOD if good else BAD
        ytop = H - i * (row_h + row_gap)
        fig.text(fx(left), fy(ytop - 0.02), f"{title}: {short}", ha="left", va="top", fontsize=7.4, weight="bold",
                 color=col)
        fig.text(fx(left + 1.75), fy(ytop - 0.03), ROW_LINE[p], ha="left", va="top", fontsize=6.4, color="0.25")
        n = len(pools[p])
        fig.text(fx(W - right), fy(ytop - 0.03), f"{len(drawn[p])} of {n} qualifying cases, drawn at random",
                 ha="right", va="top", fontsize=6.0, color="0.35", style="italic")
        yl = ytop - head_h + 0.02
        fig.add_artist(Line2D([fx(left), fx(W - right)], [fy(yl), fy(yl)], color=col, lw=0.8))
        for j, k in enumerate(drawn[p]):
            xf, x1 = check_pixels(k)
            crop = crop_of(k)
            x0 = left + j * (2 * tw + pair_gap + case_gap)
            for r, (img, ans) in enumerate(((xf, k["x0"]), (x1, k["x1"]))):
                x = x0 + r * (tw + pair_gap)
                fig.text(fx(x + tw / 2), fy(ytop - head_h - 0.01), "forged" if r == 0 else "after ReVert",
                         ha="center", va="top", fontsize=5.8, color="0.3")
                y = ytop - head_h - sub_h - tw
                ax = fig.add_axes([fx(x), fy(y), fx(tw), fy(tw)])
                draw_image(ax, img, ans, k, after=bool(r), crop=crop)
                verdict_tag(fig, fx(x + 0.02), fy(y - 0.03), k, ans)
    handles = [Patch(fc="0.85", ec="0.3", lw=0.5, label="bright: where the model points (its mask); rest dimmed"),
               Line2D([], [], color="0.3", ls=(0, (3, 2)), lw=0.9, label="forged region"),
               Line2D([], [], color="0.3", lw=1.1, label="restored to original pixels")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=6.2, handlelength=1.6, columnspacing=1.4, borderaxespad=0.15)
    base = os.path.join(a.out, f"fig4_random_seed{a.seed}")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print(f"[fig4] font: {fam}; wrote {base}.pdf and .png ({W:.3f} x {H:.2f} in, {K} cases per row)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="data/full_single/samples.jsonl")
    ap.add_argument("--single-sida", default="runs/full_sida")
    ap.add_argument("--single-fs", default="runs/full_fakeshield")
    ap.add_argument("--pairs-sida", default="runs/e4b_full_sida")
    ap.add_argument("--pairs-fs", default="runs/e4b_full_fakeshield")
    ap.add_argument("--min-area", type=float, default=0.03, help="(c): both edited regions >= this image fraction")
    ap.add_argument("--list", action="store_true", help="write candidate CSV and contact sheets, do not draw")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--random", type=int, default=0, metavar="K",
                    help="supplementary figure: K qualifying cases per row drawn at random (seeded); 3 fits the page width")
    ap.add_argument("--seed", type=int, default=0)
    for p in "abcd":
        ap.add_argument(f"--pick-{p}", default=None, help=f"sample id for panel ({p}) {PANELS[p][1]}")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    if a.random:
        plt, _ = setup_mpl()
        rng = np.random.default_rng(a.seed)
        pools = collect(a)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"fig4_random_seed{a.seed}_pools.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["panel", "rule", "qualifying_cases", "drawn"])
            for p, ks in pools.items():
                w.writerow([p, PANELS[p][1], len(ks), min(a.random, len(ks))])
                print(f"[fig4] ({p}) {PANELS[p][1]}: {len(ks)} qualifying cases, drawing {min(a.random, len(ks))}")
        cands = {p: [ks[i] for i in sorted(rng.permutation(len(ks))[:a.random])] if ks else []
                 for p, ks in pools.items()}
        a.top = a.random
        with open(os.path.join(a.out, f"fig4_random_seed{a.seed}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["panel", "position", "sample_id", "detector", "editor", "verdict_before", "s_before",
                        "verdict_after", "s_after"])
            for p, ks in cands.items():
                for i, k in enumerate(ks):
                    w.writerow([p, i, k["sid"], k["det"], k["editor"], k["x0"]["v"], f"{k['x0']['s']:.3f}",
                                k["x1"]["v"], f"{k['x1']['s']:.3f}"])
        draw_grid(cands, pools, a)
    elif a.list:
        plt, _ = setup_mpl()
        write_candidates(collect(a), a, plt)
    else:
        draw(a)


if __name__ == "__main__":
    main()
