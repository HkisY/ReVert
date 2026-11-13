"""Figure 1 (teaser): the same forgery, the same pixel-exact intervention, two forensic MLLMs.

Everything shown is a real experiment output: x_f, the reverted image R(x_f, A) that was actually
queried, the detectors' own masks, verdicts and scores, and the original photo. Images are embedded
unresampled (interpolation 'none'); the PDF is vector for every line, label and text, with the photos
as raster images at their native resolution. The script checks that the restored object is
pixel-identical to the original before it writes that claim.

Step 1, list candidates (SIDA phantom evidence; with --compare also FakeShield on the same sample):
    python scripts/make_teaser.py --compare runs/e4b_full_fakeshield
Step 2, draw (default --style clean: forged | A restored, SIDA | [same image, FakeShield]):
    python scripts/make_teaser.py --pick <sample_id>                                    # SIDA only, 2 panels
    python scripts/make_teaser.py --pick <sample_id> --compare runs/e4b_full_fakeshield # + FakeShield, 3 panels
    --crop auto|none|x0,y0,x1,y1   --style full (old layout with the zoom panel)

Selection (for the figure only, not a statistic): x_f flagged by SIDA; its mask covers >= 50% of one
edited region A, < 10% of the other (B), and >= --min-share of the mask lies on A; after A is restored
SIDA still says fake, the new mask covers >= 50% of A and >= half of the originally cited part
(cPER = 1), >= --min-share of it lies on A, and < 10% of B; both regions >= --min-area of the image.
Contrast (with --compare): FakeShield also flags x_f with its mask on A (>= 30% of A, < 10% of B), and
after A is restored it says real or its mask leaves A (< 10% of A).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from revert_audit.detectors.base import answer_score, answer_verdict  # noqa: E402
from revert_audit.io import load_mask, load_rgb, read_jsonl  # noqa: E402
from revert_audit.pipeline import answer_region, qid  # noqa: E402

TEXTWIDTH_IN = 6.875  # CVPR two-column template
RED, GREEN, WHITE, CYAN = (0.92, 0.12, 0.12), (0.10, 0.78, 0.30), (1.0, 1.0, 1.0), (0.0, 0.80, 1.0)
FONTS = ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]
_THING = re.compile(r"a photo of (?:an? )?(.+?), realistic")


# ------------------------------------------------------------------ data
class Run:
    def __init__(self, run, tag):
        comp = os.path.join(run, f"component_{tag}")
        self.run, self.tag = run, tag
        self.samples = {s["id"]: s for s in read_jsonl(os.path.join(run, "pair_samples.jsonl"))}
        self.a1 = {a["query_id"]: a for a in read_jsonl(os.path.join(run, "stage1", f"answers_{tag}.jsonl"))}
        self.a2 = {a["query_id"]: a for a in read_jsonl(os.path.join(comp, "answers.jsonl"))}
        self.meta = {m["sample_id"]: m for m in read_jsonl(os.path.join(comp, "meta.jsonl"))}

    def answers(self, sid, c, shape):
        """(x_f answer, R(x_f, part c) answer) as dicts with mask, score, verdict; None if not queried."""
        a0 = self.a1.get(qid(sid, 1, "fake"))
        ac = self.a2.get(qid(sid, 2, "fake", f"comp{c + 1}", 0))
        if a0 is None:
            return None, None
        f = lambda a: None if a is None else {"mask": answer_region(a, *shape), "s": answer_score(a),
                                              "v": answer_verdict(a), "text": a.get("dte_text") or a.get("raw_text") or ""}
        return f(a0), f(ac)

    def comp_image(self, sid, c):
        return os.path.join(self.run, f"component_{self.tag}", "images", f"{sid}__comp{c + 1}.png")


def edit_desc(part):
    if part.get("edit_type") == "replace":
        m = _THING.search(part.get("prompt") or "")
        return f"replaced by {m.group(1)}" if m else "replaced"
    return "removed" if part.get("edit_type") == "remove" else (part.get("edit_type") or "?")


def cov(X, M):
    return float((X & M).sum() / max(1, M.sum()))


def share(X, M):
    return float((X & M).sum() / max(1, X.sum())) if X.any() else 0.0


def sida_case(R, sid):
    m = R.meta.get(sid)
    if not m or m.get("skipped") or not m.get("cov_ok"):
        return None
    s = R.samples[sid]
    parts = [load_mask(p["mask"]) for p in s["parts"]]
    c = m["cov_cited"]
    o = 1 - c
    x0, xc = R.answers(sid, c, parts[0].shape)
    if x0 is None or xc is None:
        return None
    A, B = parts[c], parts[o]
    target = x0["mask"] & A
    st = {"area_a": A.mean(), "area_b": B.mean(),
          "cov0_a": cov(x0["mask"], A), "cov0_b": cov(x0["mask"], B), "share0_a": share(x0["mask"], A),
          "cov1_a": cov(xc["mask"], A), "cov1_b": cov(xc["mask"], B), "share1_a": share(xc["mask"], A),
          "share1_b": share(xc["mask"], B),
          "cper": bool(xc["v"] == "fake" and (xc["mask"] & target).sum() >= 0.5 * max(1, target.sum())),
          "drop": x0["s"] - xc["s"]}
    return {"sid": sid, "editor": s.get("editor"), "c": c, "o": o, "A": A, "B": B,
            "obj_a": s["parts"][c].get("obj"), "obj_b": s["parts"][o].get("obj"),
            "edit_a": edit_desc(s["parts"][c]), "edit_b": edit_desc(s["parts"][o]),
            "sida0": x0, "sida1": xc, "st": st,
            "xf": os.path.join(R.run, "stage1", "images", f"{sid}__fake.png"), "xr": s["real"],
            "rc": R.comp_image(sid, c)}


def sida_ok(k, min_area, min_share):
    st = k["st"]
    return (k["sida0"]["v"] == "fake" and k["sida1"]["v"] == "fake" and st["cper"]
            and st["cov0_a"] >= 0.5 and st["cov0_b"] < 0.1 and st["share0_a"] >= min_share
            and st["cov1_a"] >= 0.5 and st["cov1_b"] < 0.1 and st["share1_a"] >= min_share and st["share1_b"] < 0.5
            and st["area_a"] >= min_area and st["area_b"] >= min_area)


def add_fakeshield(k, F):
    """FakeShield on the same x_f and on the same R(x_f, A) (queried only if FakeShield's own citation
    put the sample in its component set). Fills k['fs0'], k['fs1'], k['contrast']."""
    k["fs0"] = k["fs1"] = None
    k["contrast"] = k["fs_switch"] = False
    if F is None:
        return k
    f0, f1 = F.answers(k["sid"], k["c"], k["A"].shape)
    k["fs0"], k["fs1"] = f0, f1
    if f0 is None or f1 is None:
        return k
    if not np.array_equal(load_rgb(F.comp_image(k["sid"], k["c"])), load_rgb(k["rc"])):
        raise SystemExit(f"{k['sid']}: FakeShield and SIDA were queried on different restored images")
    k["contrast"] = bool(f0["v"] == "fake" and cov(f0["mask"], k["A"]) >= 0.3 and cov(f0["mask"], k["B"]) < 0.1
                         and (f1["v"] != "fake" or cov(f1["mask"], k["A"]) < 0.1))
    # the clearest contrast: FakeShield moves to the region that is still forged
    k["fs_switch"] = bool(k["contrast"] and f1["v"] == "fake" and cov(f1["mask"], k["B"]) >= 0.3)
    return k


# ------------------------------------------------------------------ drawing helpers
def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in FONTS if f in have] or ["serif"]
    plt.rcParams.update({"font.family": "serif", "font.serif": fam, "font.size": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "image.interpolation": "none", "image.resample": False})
    return plt, fam[0]


def contour(ax, M, color, dashed=False, lw=1.1, halo=True):
    """Exact outline of a region (the edited pixels), not a bounding box. A thin dark halo keeps white
    and coloured lines visible on any background."""
    Mf = M.astype(float)
    if halo:
        ax.contour(Mf, levels=[0.5], colors=[(0, 0, 0, 0.55)], linewidths=lw + 1.0)
    ax.contour(Mf, levels=[0.5], colors=[color], linewidths=lw, linestyles="dashed" if dashed else "solid")


def label(ax, M, text, color, where="top"):
    ys, xs = np.where(M)
    h, w = M.shape
    x = int(np.clip(xs.min(), 2, w - 2))
    y = ys.min() - 4 if where == "top" and ys.min() > 0.08 * h else ys.max() + 4
    va = "bottom" if y < ys.min() else "top"
    ax.text(x, y, text, color="black", fontsize=6.3, va=va, ha="left", clip_on=True,
            bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none", alpha=0.92))


def show(ax, img, sida_mask=None, fs_mask=None):
    ax.imshow(img)
    if sida_mask is not None and sida_mask.any():
        rgba = np.zeros(sida_mask.shape + (4,), float)
        rgba[sida_mask] = (*RED, 0.42)
        ax.imshow(rgba)
        ax.contour(sida_mask.astype(float), levels=[0.5], colors=[RED], linewidths=0.7)
    if fs_mask is not None and fs_mask.any():
        contour(ax, fs_mask, CYAN, lw=1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def crop_box(M, img_shape, margin=0.15):
    ys, xs = np.where(M)
    h, w = img_shape[:2]
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin))
    side = min(max(side, 16), h, w)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    X0 = int(np.clip(cx - side // 2, 0, w - side))
    Y0 = int(np.clip(cy - side // 2, 0, h - side))
    return X0, Y0, X0 + side, Y0 + side


def vtxt(ans, A=None, B=None):
    if ans is None:
        return "not queried"
    t = f"{'fake' if ans['v'] == 'fake' else 'real'} ({ans['s']:.2f})"
    if A is None:
        return t
    m = ans["mask"]
    if ans["v"] != "fake" or not m.any():
        return t + ", no region"
    return t + (", points at A" if cov(m, A) >= 0.3 else ", points at B" if cov(m, B) >= 0.3 else ", points elsewhere")


# ------------------------------------------------------------------ step 1: candidates
def list_candidates(a):
    R = Run(a.run, a.tag)
    F = Run(a.compare, a.compare_tag) if a.compare else None
    rows = []
    for sid in R.meta:
        k = sida_case(R, sid)
        if k is not None and sida_ok(k, a.min_area, a.min_share):
            rows.append(add_fakeshield(k, F))
    rows.sort(key=lambda k: k["st"]["drop"])
    print(f"[teaser] {len(rows)} SIDA phantom-evidence candidates (min region {a.min_area:.0%}, "
          f"min mask share on A {a.min_share:.0%})")
    if not rows:
        print("[teaser] none: try --min-area 0.03 or --min-share 0.4")
        return
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "teaser_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "sample_id", "editor", "A", "A_edit", "B", "B_edit", "sida_xf", "sida_restored",
                     "drop", "share_on_A_before", "share_on_A_after", "area_A", "area_B",
                     "fakeshield_xf", "fakeshield_restored", "contrast"])
        for i, k in enumerate(rows, 1):
            st = k["st"]
            wr.writerow([i, k["sid"], k["editor"], k["obj_a"], k["edit_a"], k["obj_b"], k["edit_b"],
                         f"{k['sida0']['s']:.4f}", f"{k['sida1']['s']:.4f}", f"{st['drop']:.4f}",
                         f"{st['share0_a']:.2f}", f"{st['share1_a']:.2f}", f"{st['area_a']:.3f}",
                         f"{st['area_b']:.3f}", vtxt(k["fs0"]), vtxt(k["fs1"]), int(k["contrast"])])
    contrast = sorted([k for k in rows if k["contrast"]], key=lambda k: (not k["fs_switch"], k["st"]["drop"]))
    if F is not None:
        print(f"[teaser] {len(contrast)} of them are contrast cases (FakeShield cites A on x_f and lets go after A is "
              f"restored); {sum(k['fs_switch'] for k in contrast)} where FakeShield moves to B (listed first)")
    sheet = (contrast + [k for k in rows if not k["contrast"]])[:max(a.top, len(contrast[:a.top]))]
    plt, _ = setup_mpl()
    fig, axes = plt.subplots(len(sheet), 3, figsize=(7.5, 2.5 * len(sheet)), squeeze=False)
    for i, k in enumerate(sheet):
        xf, rc, xr = load_rgb(k["xf"]), load_rgb(k["rc"]), load_rgb(k["xr"])
        fs0 = k["fs0"]["mask"] if k["fs0"] else None
        fs1 = k["fs1"]["mask"] if k["fs1"] else None
        rank = rows.index(k) + 1
        show(axes[i, 0], xf, k["sida0"]["mask"], fs0)
        contour(axes[i, 0], k["A"], WHITE, dashed=True)
        contour(axes[i, 0], k["B"], WHITE, dashed=True)
        label(axes[i, 0], k["A"], "A", WHITE)
        label(axes[i, 0], k["B"], "B", WHITE)
        tag = "  [CONTRAST: FakeShield -> B]" if k["fs_switch"] else "  [CONTRAST]" if k["contrast"] else ""
        axes[i, 0].set_title(f"#{rank} {k['sid']} ({k['editor']}){tag}\nSIDA {vtxt(k['sida0'])}"
                             + (f" | FakeShield {vtxt(k['fs0'])}" if F else ""), fontsize=6.5)
        show(axes[i, 1], rc, k["sida1"]["mask"], fs1)
        contour(axes[i, 1], k["A"], GREEN)
        contour(axes[i, 1], k["B"], WHITE, dashed=True)
        label(axes[i, 1], k["A"], "A restored", GREEN)
        axes[i, 1].set_title(f"A={k['obj_a']} ({k['edit_a']}), B={k['obj_b']} ({k['edit_b']})\nSIDA {vtxt(k['sida1'])}"
                             + (f" | FakeShield {vtxt(k['fs1'])}" if F else ""), fontsize=6.5)
        X0, Y0, X1, Y1 = crop_box(k["A"], xf.shape)
        axes[i, 2].imshow(np.concatenate([xf[Y0:Y1, X0:X1], rc[Y0:Y1, X0:X1], xr[Y0:Y1, X0:X1]], axis=1))
        axes[i, 2].set_title("A: forged | restored | original", fontsize=6.5)
        axes[i, 2].axis("off")
    fig.tight_layout()
    p = os.path.join(a.out, "teaser_candidates.png")
    fig.savefig(p, dpi=110)
    print(f"[teaser] wrote {p} and teaser_candidates.csv (contrast cases first on the sheet)")
    for k in sheet:
        st = k["st"]
        print(f"  #{rows.index(k) + 1:<3} {k['sid']:<26} {k['editor']:<10} A={k['obj_a']} ({k['edit_a']}), "
              f"B={k['obj_b']}  SIDA {k['sida0']['s']:.3f}->{k['sida1']['s']:.3f}  share on A "
              f"{st['share0_a']:.2f}/{st['share1_a']:.2f}"
              + (f"  FakeShield {vtxt(k['fs0'], k['A'], k['B'])} -> {vtxt(k['fs1'], k['A'], k['B'])}"
                 f"{'  CONTRAST' if k['contrast'] else ''}" if F else ""))


# ------------------------------------------------------------------ step 2: the figure
def draw_teaser(a):
    R = Run(a.run, a.tag)
    F = Run(a.compare, a.compare_tag) if a.compare else None
    k = sida_case(R, a.pick)
    if k is None:
        raise SystemExit(f"{a.pick}: not in SIDA's component set or answers missing")
    add_fakeshield(k, F)
    if F is not None and (k["fs0"] is None or k["fs1"] is None):
        raise SystemExit(f"{a.pick}: FakeShield was not queried on R(x_f, A) for this sample; drop --compare")
    if not sida_ok(k, 0.0, 0.0):
        print("[teaser] note: this sample does not pass the SIDA selection filter")
    xf, rc, xr = load_rgb(k["xf"]), load_rgb(k["rc"]), load_rgb(k["xr"])
    A, B = k["A"], k["B"]
    if not np.array_equal(rc[A], xr[A]):  # the claim in (c) is checked, not assumed
        raise SystemExit(f"{a.pick}: restored region is NOT pixel-identical to the original; not drawing the claim")
    X0, Y0, X1, Y1 = crop_box(A, xf.shape)
    crop_equal = bool(np.array_equal(rc[Y0:Y1, X0:X1], xr[Y0:Y1, X0:X1]))
    claim = "restored\n= original\n(pixel-identical)" if crop_equal else "restored\n= original\n(pixel-identical\ninside A)"

    plt, fam = setup_mpl()
    h, w = xf.shape[:2]
    ar = w / h
    n_lines = 3 if F else 1
    W, gap, g2 = TEXTWIDTH_IN, 0.05, 0.03
    cap = 0.13 + 0.12 * n_lines
    H = (W - 2 * gap) / (2 * ar + 1)  # (c) is a 2x2 block of square cells, as tall as (a) and (b)
    c = (H - g2) / 2
    FH = H + cap
    fig = plt.figure(figsize=(W, FH))
    ax_at = lambda x, y, ww, hh: fig.add_axes([x / W, y / FH, ww / W, hh / FH])
    axa, axb = ax_at(0, cap, ar * H, H), ax_at(ar * H + gap, cap, ar * H, H)
    x_c = 2 * ar * H + 2 * gap
    axc = [ax_at(x_c, cap + c + g2, c, c), ax_at(x_c + c + g2, cap + c + g2, c, c), ax_at(x_c, cap, c, c)]
    axt = ax_at(x_c + c + g2, cap, c, c)
    axt.axis("off")

    fs0 = k["fs0"]["mask"] if F else None
    fs1 = k["fs1"]["mask"] if F else None
    show(axa, xf, k["sida0"]["mask"], fs0)
    contour(axa, A, WHITE, dashed=True)
    contour(axa, B, WHITE, dashed=True)
    label(axa, A, "A", WHITE)
    label(axa, B, "B", WHITE)
    show(axb, rc, k["sida1"]["mask"], fs1)
    contour(axb, A, GREEN)
    contour(axb, B, WHITE, dashed=True)
    label(axb, A, "A: restored (authentic)", GREEN)
    label(axb, B, "B: still forged", WHITE)

    def caption(ax, head, x_ans, f_ans):
        y = -0.025
        ax.text(0.5, y, head, transform=ax.transAxes, ha="center", va="top", fontsize=7)
        y -= 0.12 / H
        lines = [f"SIDA (red): {vtxt(x_ans, A, B)}"] + ([f"FakeShield (cyan): {vtxt(f_ans, A, B)}"] if F else [])
        ax.text(0.5, y, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
                linespacing=1.15)
        return y

    if F:
        caption(axa, "(a) Forged image", k["sida0"], k["fs0"])
        caption(axb, "(b) Region A restored to its original pixels", k["sida1"], k["fs1"])
    else:
        axa.text(0.5, -0.025, f"(a) Forged image. SIDA: {vtxt(k['sida0'], A, B)}", transform=axa.transAxes,
                 ha="center", va="top", fontsize=7)
        axb.text(0.5, -0.025, f"(b) A restored. SIDA: {vtxt(k['sida1'], A, B)}",
                 transform=axb.transAxes, ha="center", va="top", fontsize=7)

    for ax, im, lab in zip(axc, (xf, rc, xr), ("forged", "restored", "original")):
        ax.imshow(im[Y0:Y1, X0:X1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.03, 0.95, lab, transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="white",
                bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
        for sp in ax.spines.values():
            sp.set_visible(lab == "restored")
            sp.set_edgecolor(GREEN)
            sp.set_linewidth(1.0)
    axt.text(0.5, 0.5, claim, transform=axt.transAxes, ha="center", va="center", fontsize=6.5, linespacing=1.2)
    fig.text((x_c + H / 2) / W, (cap - 0.025 * H) / FH, "(c) Zoom on A", ha="center", va="top", fontsize=7)

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"teaser_{a.pick}" + ("_vs_fakeshield" if F else ""))
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    st = k["st"]
    print(f"[teaser] font: {fam}; wrote {base}.pdf (vector, photos unresampled) and .png (preview)")
    print(f"[teaser] {a.pick} ({k['editor']}): A={k['obj_a']} ({k['edit_a']}), B={k['obj_b']} ({k['edit_b']})")
    print(f"         SIDA {vtxt(k['sida0'])} -> {vtxt(k['sida1'])}; mask share on A {st['share0_a']:.2f} -> "
          f"{st['share1_a']:.2f}; pixel-identical inside A: True, whole zoom crop: {crop_equal}")
    if F:
        print(f"         FakeShield {vtxt(k['fs0'], A, B)} -> {vtxt(k['fs1'], A, B)}; contrast: {k['contrast']}, "
              f"moves to B: {k['fs_switch']}")


# ------------------------------------------------------------------ step 2b: the clean teaser (default)
def roi(A, B, shape, aspect, margin=0.10):
    """Crop to the two edited regions (+ margin), widened to `aspect` (w/h) inside the image."""
    ys, xs = np.where(A | B)
    h, w = shape[:2]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin) + 8, int((y1 - y0) * margin) + 8
    x0, x1, y0, y1 = max(0, x0 - mx), min(w, x1 + mx), max(0, y0 - my), min(h, y1 + my)
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < aspect:
        cw = min(w, int(round(ch * aspect)))
    else:
        ch = min(h, int(round(cw / aspect)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    X0 = int(np.clip(cx - cw // 2, 0, w - cw))
    Y0 = int(np.clip(cy - ch // 2, 0, h - ch))
    return X0, Y0, X0 + cw, Y0 + ch


def fill(ax, M, color, alpha=0.45):
    if M is None or not M.any():
        return
    rgba = np.zeros(M.shape + (4,), float)
    rgba[M] = (*color, alpha)
    ax.imshow(rgba)
    ax.contour(M.astype(float), levels=[0.5], colors=[color], linewidths=0.8)


def short_edit(k, which):
    """What the edit did to the region, in a few words: 'person removed' explains why a region that
    still carries an edit can look empty (the inpainter filled it with background)."""
    obj, e = k[f"obj_{which}"] or "object", k[f"edit_{which}"]
    if e == "removed":
        return f"{obj} removed"
    if e.startswith("replaced"):
        return f"{obj} replaced"
    return "edited"


def where_txt(ans, A, B, name):
    if ans is None:
        return f"{name}: not queried"
    if ans["v"] != "fake" or not ans["mask"].any():
        return f"{name}: real ({ans['s']:.2f}), cites nothing"
    m = ans["mask"]
    loc = "A" if cov(m, A) >= 0.3 else "B" if cov(m, B) >= 0.3 else "elsewhere"
    return f"{name}: fake ({ans['s']:.2f}), cites {loc}"


def draw_clean(a):
    """One message per panel: the forged image, the same image with A restored (SIDA's mask on the
    authentic object), and optionally FakeShield on that same restored image. No zoom panel: the
    pixel identity is verified here and stated in the title of (b)."""
    R = Run(a.run, a.tag)
    F = Run(a.compare, a.compare_tag) if a.compare else None
    k = sida_case(R, a.pick)
    if k is None:
        raise SystemExit(f"{a.pick}: not in SIDA's component set or answers missing")
    add_fakeshield(k, F)
    if F is not None and k["fs1"] is None:
        raise SystemExit(f"{a.pick}: FakeShield was not queried on R(x_f, A) for this sample; drop --compare")
    xf, rc, xr = load_rgb(k["xf"]), load_rgb(k["rc"]), load_rgb(k["xr"])
    A, B = k["A"], k["B"]
    if not np.array_equal(rc[A], xr[A]):
        raise SystemExit(f"{a.pick}: restored region is NOT pixel-identical to the original")
    n = 3 if F else 2
    plt, fam = setup_mpl()
    ar = a.aspect
    if a.crop == "auto":
        X0, Y0, X1, Y1 = roi(A, B, xf.shape, ar)
    elif a.crop == "none":
        X0, Y0, X1, Y1 = 0, 0, xf.shape[1], xf.shape[0]
    else:
        X0, Y0, X1, Y1 = (int(v) for v in a.crop.split(","))
    ar = (X1 - X0) / (Y1 - Y0)
    S = lambda X: None if X is None else X[Y0:Y1, X0:X1]
    W, gap, top, bot = TEXTWIDTH_IN, 0.06, 0.20, (0.32 if F else 0.20)
    pw = (W - (n - 1) * gap) / n
    ph = pw / ar
    FH = top + ph + bot
    fig = plt.figure(figsize=(W, FH))
    axes = [fig.add_axes([(i * (pw + gap)) / W, bot / FH, pw / W, ph / FH]) for i in range(n)]
    Ac, Bc = S(A), S(B)

    def base(ax, img):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def head(ax, t):
        ax.text(0.5, 1.0 + 0.04 / ph, t, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)

    def foot(ax, t, color="black"):
        ax.text(0.5, -0.04 / ph, t, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=color,
                linespacing=1.15)

    # (a) forged
    base(axes[0], S(xf))
    fill(axes[0], S(k["sida0"]["mask"]), RED)
    contour(axes[0], Ac, WHITE, dashed=True, lw=0.9)
    contour(axes[0], Bc, WHITE, dashed=True, lw=0.9)
    label(axes[0], Ac, f"A: {short_edit(k, 'a')}", WHITE)
    label(axes[0], Bc, f"B: {short_edit(k, 'b')}", WHITE)
    head(axes[0], "(a) Forged image: two regions edited")
    foot(axes[0], where_txt(k["sida0"], A, B, "SIDA"))
    # (b) A restored, SIDA
    base(axes[1], S(rc))
    fill(axes[1], S(k["sida1"]["mask"]), RED)
    contour(axes[1], Ac, GREEN, lw=1.1)
    contour(axes[1], Bc, WHITE, dashed=True, lw=0.9)
    label(axes[1], Ac, "A: original pixels", GREEN)
    label(axes[1], Bc, f"B: {short_edit(k, 'b')}, not restored", WHITE)
    head(axes[1], "(b) A restored to its original pixels")
    foot(axes[1], where_txt(k["sida1"], A, B, "SIDA"))
    # (c) the same restored image, FakeShield
    if F:
        base(axes[2], S(rc))
        fill(axes[2], S(k["fs1"]["mask"]), CYAN)
        contour(axes[2], Ac, GREEN, lw=1.1)
        contour(axes[2], Bc, WHITE, dashed=True, lw=0.9)
        label(axes[2], Ac, "A", GREEN)
        label(axes[2], Bc, "B", WHITE)
        head(axes[2], "(c) Same image as (b), FakeShield")
        fs0 = where_txt(k["fs0"], A, B, "FakeShield").replace("FakeShield: ", "")
        foot(axes[2], where_txt(k["fs1"], A, B, "FakeShield") + f"\n(on the forged image: {fs0})")
    os.makedirs(a.out, exist_ok=True)
    base_p = os.path.join(a.out, f"teaser_clean_{a.pick}" + ("_vs_fakeshield" if F else ""))
    fig.savefig(base_p + ".pdf")
    fig.savefig(base_p + ".png", dpi=300)
    print(f"[teaser] font: {fam}; crop x {X0}-{X1}, y {Y0}-{Y1}; wrote {base_p}.pdf and .png")
    print(f"         {where_txt(k['sida0'], A, B, 'SIDA')} -> {where_txt(k['sida1'], A, B, 'SIDA')}; A pixel-identical: True")
    if F:
        print(f"         {where_txt(k['fs0'], A, B, 'FakeShield')} -> {where_txt(k['fs1'], A, B, 'FakeShield')}")


# ------------------------------------------------------------------ step 2c: storyboard teaser (default)
def sida_class(ans):
    """SIDA's predicted class (real / full synthetic / tampered). Under its paper protocol the answer
    text is the fixed template of that class, so we show the class, not a quoted sentence."""
    m = re.search(r"The image is ([a-z ]+)", ans.get("text") or "")
    return m.group(1).strip() if m else ("tampered" if ans["v"] == "fake" else "real")


def model_says(ans):
    """The model's own words, minus special tokens: SIDA answers '[CLS] The image is tampered [SEG]'."""
    t = re.sub(r"\[(CLS|SEG)\]", "", ans.get("text") or "").strip()
    t = t.split(".")[0].strip() if t else ""
    return t or ("The image is tampered" if ans["v"] == "fake" else "The image is real")


def load_stats(path):
    import json

    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return {"n": d.get("n"), "cPER": d.get("cPER"), "switch": d.get("switch")}


def draw_story(a):
    """Three parts, read left to right: (a) the forged image and the model's answer, (b) the same image
    after ReVert restores the cited region to its original pixels, with the model's new answer, and
    (c) how often this happens over all two-object forgeries (from component.json, with 95% CIs)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    R = Run(a.run, a.tag)
    k = sida_case(R, a.pick)
    if k is None:
        raise SystemExit(f"{a.pick}: not in SIDA's component set or answers missing")
    xf, rc, xr = load_rgb(k["xf"]), load_rgb(k["rc"]), load_rgb(k["xr"])
    A, B = k["A"], k["B"]
    if not np.array_equal(rc[A], xr[A]):
        raise SystemExit(f"{a.pick}: restored region is NOT pixel-identical to the original")
    stats = {"SIDA": load_stats(os.path.join(a.run, f"component_{a.tag}", "component.json"))}
    if a.compare:
        stats["FakeShield"] = load_stats(os.path.join(a.compare, f"component_{a.compare_tag}", "component.json"))
    stats = {m: v for m, v in stats.items() if v and v.get("cPER")}

    plt, fam = setup_mpl()
    if a.crop == "auto":
        X0, Y0, X1, Y1 = roi(A, B, xf.shape, a.aspect)
    elif a.crop == "none":
        X0, Y0, X1, Y1 = 0, 0, xf.shape[1], xf.shape[0]
    else:
        X0, Y0, X1, Y1 = (int(v) for v in a.crop.split(","))
    ar = (X1 - X0) / (Y1 - Y0)
    S = lambda X: X[Y0:Y1, X0:X1]
    W, arrow_w, chart_w, gap = TEXTWIDTH_IN, 0.50, (1.70 if stats else 0.0), 0.12
    top, bub = 0.22, 0.46
    pw = (W - arrow_w - chart_w - (gap if stats else 0)) / 2
    ph = pw / ar
    FH = top + ph + bub
    fig = plt.figure(figsize=(W, FH))
    fx = lambda x: x / W
    fy = lambda y: y / FH
    ax1 = fig.add_axes([fx(0), fy(bub), fx(pw), fy(ph)])
    ax2 = fig.add_axes([fx(pw + arrow_w), fy(bub), fx(pw), fy(ph)])
    Ac, Bc = S(A), S(B)

    for ax, img, ans, restored in ((ax1, S(xf), k["sida0"], False), (ax2, S(rc), k["sida1"], True)):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fill(ax, S(ans["mask"]), RED)
        if restored:
            contour(ax, Ac, GREEN, lw=1.2)
            label(ax, Ac, "A: original pixels", GREEN)
            contour(ax, Bc, WHITE, dashed=True, lw=0.9)
            label(ax, Bc, f"B: {short_edit(k, 'b')}, not restored", WHITE)
        else:
            contour(ax, Ac, WHITE, dashed=True, lw=0.9)
            label(ax, Ac, f"A: {short_edit(k, 'a')}", WHITE)
            contour(ax, Bc, WHITE, dashed=True, lw=0.9)
            label(ax, Bc, f"B: {short_edit(k, 'b')}", WHITE)
    fig.text(fx(pw / 2), fy(bub + ph + 0.05), "(a) Forged image", ha="center", va="bottom", fontsize=8)
    fig.text(fx(pw + arrow_w + pw / 2), fy(bub + ph + 0.05), "(b) Same image, A restored to its original pixels",
             ha="center", va="bottom", fontsize=8)

    # the intervention between (a) and (b)
    ym = bub + ph / 2
    fig.patches.append(FancyArrowPatch((fx(pw + 0.04), fy(ym)), (fx(pw + arrow_w - 0.04), fy(ym)),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                                       lw=1.4, color="0.15"))
    fig.text(fx(pw + arrow_w / 2), fy(ym + 0.08), "ReVert:\nrestore\nregion A", ha="center", va="bottom",
             fontsize=6.5, linespacing=1.1)
    fig.text(fx(pw + arrow_w / 2), fy(ym - 0.08), "pixel-\nexact", ha="center", va="top", fontsize=6.5,
             color="0.35", linespacing=1.1)

    # the model's answers, as speech bubbles under the images
    def bubble(x0, ans, verdict_line, mark=None):
        bx, by, bw, bh = x0 + 0.05, 0.05, pw - 0.10, bub - 0.13
        fig.patches.append(FancyBboxPatch((fx(bx), fy(by)), fx(bw), fy(bh), boxstyle="round,pad=0.004,rounding_size=0.012",
                                          transform=fig.transFigure, fc="0.96", ec="0.55", lw=0.6))
        fig.text(fx(bx + 0.06), fy(by + bh - 0.05), f"SIDA verdict: {sida_class(ans)}", ha="left", va="top",
                 fontsize=7, weight="bold")
        fig.text(fx(bx + 0.06), fy(by + 0.05), verdict_line, ha="left", va="bottom", fontsize=6.6)
        if mark:
            fig.text(fx(bx + bw - 0.06), fy(by + bh / 2), mark[0], ha="right", va="center", fontsize=11,
                     color=mark[1], family="DejaVu Sans", weight="bold")

    loc = lambda ans: ("points at A" if cov(ans["mask"], A) >= 0.3 else "points at B" if cov(ans["mask"], B) >= 0.3
                       else "points elsewhere") if ans["v"] == "fake" and ans["mask"].any() else "points at nothing"
    bubble(0, k["sida0"], f"fake score {k['sida0']['s']:.2f}; red mask {loc(k['sida0'])}")
    bubble(pw + arrow_w, k["sida1"], f"fake score {k['sida1']['s']:.2f}; still {loc(k['sida1'])}, now authentic",
           ("\u2717", RED))

    # (c) how often, over all eligible two-object forgeries
    if stats:
        cx = W - chart_w
        axc = fig.add_axes([fx(cx + 0.36), fy(bub + 0.02), fx(chart_w - 0.40), fy(ph - 0.02)])
        models = list(stats)
        xs = np.arange(len(models))
        bw = 0.36
        for j, (key, col, lab) in enumerate((("cPER", RED, "still points at the\nrestored region"),
                                             ("switch", GREEN, "moves to the\nremaining edit"))):
            vals = [stats[m][key] for m in models]
            mean = np.array([v[0] for v in vals]) * 100
            lo = mean - np.array([v[1] for v in vals]) * 100
            hi = np.array([v[2] for v in vals]) * 100 - mean
            axc.bar(xs + (j - 0.5) * bw, mean, bw * 0.92, color=col, alpha=0.85, label=lab,
                    hatch=None if key == "cPER" else "////", edgecolor="white", linewidth=0,
                    yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5, ecolor="0.25"))
            for x, m, h in zip(xs + (j - 0.5) * bw, mean, mean + hi):
                axc.text(x, h + 1.0, f"{m:.0f}%" if m >= 1 else f"{m:.1f}%", ha="center", va="bottom", fontsize=6)
        axc.set_xticks(xs)
        axc.set_xticklabels([f"{m}\n(n={stats[m]['n']})" for m in models], fontsize=6.5)
        axc.set_ylabel("% of cases (95% CI)", fontsize=6.5, labelpad=1)
        axc.tick_params(axis="y", labelsize=6, length=2, pad=1)
        axc.tick_params(axis="x", length=0, pad=2)
        ymax = max(max(v[2] for v in (stats[m]["cPER"], stats[m]["switch"])) for m in models) * 100
        axc.set_ylim(0, max(10, ymax * 1.45))
        for sp in ("top", "right"):
            axc.spines[sp].set_visible(False)
        axc.legend(fontsize=6, frameon=False, loc="upper right", handlelength=0.9, borderaxespad=0.1,
                   labelspacing=0.5)
        fig.text(fx(cx + chart_w / 2), fy(bub + ph + 0.05), "(c) Same test, all two-object cases",
                 ha="center", va="bottom", fontsize=8)

    os.makedirs(a.out, exist_ok=True)
    base_p = os.path.join(a.out, f"teaser_story_{a.pick}")
    fig.savefig(base_p + ".pdf")
    fig.savefig(base_p + ".png", dpi=300)
    print(f"[teaser] font: {fam}; crop x {X0}-{X1}, y {Y0}-{Y1}; wrote {base_p}.pdf and .png")
    for m, v in stats.items():
        print(f"         (c) {m}: n={v['n']} still cites {v['cPER'][0]:.3f} [{v['cPER'][1]:.3f}, {v['cPER'][2]:.3f}], "
              f"moves to remaining edit {v['switch'][0]:.3f} [{v['switch'][1]:.3f}, {v['switch'][2]:.3f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/e4b_full_sida")
    ap.add_argument("--tag", default="sida7b")
    ap.add_argument("--compare", default=None, help="FakeShield E4b run, e.g. runs/e4b_full_fakeshield")
    ap.add_argument("--compare-tag", default="fakeshield")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-area", type=float, default=0.05)
    ap.add_argument("--min-share", type=float, default=0.5, help="fraction of SIDA's mask that lies on A")
    ap.add_argument("--pick", default=None, help="sample id to draw")
    ap.add_argument("--style", choices=["story", "clean", "full"], default="story",
                    help="story: example + intervention + statistics (default); clean: 2-3 image panels; full: with zoom")
    ap.add_argument("--crop", default="auto", help="auto (around A and B) | none | x0,y0,x1,y1 in pixels")
    ap.add_argument("--aspect", type=float, default=4 / 3, help="panel aspect w/h for --crop auto")
    a = ap.parse_args(argv)
    if not a.pick:
        list_candidates(a)
    elif a.style == "story":
        draw_story(a)
    elif a.style == "clean":
        draw_clean(a)
    else:
        draw_teaser(a)


if __name__ == "__main__":
    main()
