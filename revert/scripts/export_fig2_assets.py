"""Thumbnails for Figure 2 (method overview), exported from real experiment outputs so that the
hand-drawn figure (PowerPoint / draw.io / Illustrator) uses the images that were actually queried.

Step 1, list single-object SIDA candidates for the partial-reversion lane:
    python scripts/export_fig2_assets.py --list
    (SIDA flagged x_f; its mask covers 30-70% of the edit, >= 70% of the mask lies on the edit, so a
     restored part and a still-forged part are both visible; the in-edit control B exists; the edit
     covers 4-25% of the image. Ranked by how close coverage is to 50%.)
Step 2, export:
    python scripts/export_fig2_assets.py --pick <single-object id> [--pair coco442836p_flux-fill]

Writes figures/fig2_assets/ : every image as <name>_raw.png (cropped photo only) and <name>_ov.png
(same crop with the overlays of Figure 1: red fill = model's mask, white dashed = forged region,
green solid = region restored in that query), plus the masks as transparent PNG layers (<name>_layer.png)
so the overlays can be restyled in the drawing tool. All single-object images share one square crop,
the two-object images another. Pixels are only cropped (no resampling). Every reverted image is checked
to equal R(x_f, region) before it is written.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from make_fig4 import Single  # noqa: E402
from make_teaser import Run, sida_case  # noqa: E402
from revert_audit.io import load_mask, load_rgb, read_jsonl, save_rgb  # noqa: E402
from revert_audit.pipeline import _seed  # noqa: E402
from revert_audit.regions import blob_in_region  # noqa: E402

RED, GREEN, WHITE = (235, 31, 31), (26, 199, 77), (255, 255, 255)


def square_crop(U, margin=0.12):
    ys, xs = np.where(U)
    h, w = U.shape
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin))
    side = min(max(side, 64), h, w)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return (slice(y0, y0 + side), slice(x0, x0 + side))


def edge(M, width=2):
    """Boundary pixels of a mask (inner band of `width` px)."""
    from scipy.ndimage import binary_erosion

    return M & ~binary_erosion(M, iterations=width)


def dashed(E, period=10):
    ys, xs = np.nonzero(E)
    keep = ((ys + xs) // (period // 2)) % 2 == 0
    out = np.zeros_like(E)
    out[ys[keep], xs[keep]] = True
    return out


def overlay(img, red=None, forged=(), restored=()):
    """Figure 1 style, burned into the pixels at native resolution (for the _ov preview)."""
    out = img.astype(np.float32).copy()
    if red is not None and red.any():
        out[red] = 0.55 * out[red] + 0.45 * np.array(RED)
    for M in forged:
        E = edge(M, 2)
        out[E] = 0
        out[dashed(E)] = WHITE
    for M in restored:
        E = edge(M, 3)
        out[E] = GREEN
    return np.clip(out, 0, 255).astype(np.uint8)


def layer(M, color, alpha=255):
    """Mask as an RGBA layer (transparent elsewhere)."""
    out = np.zeros(M.shape + (4,), np.uint8)
    out[M] = (*color, alpha)
    return out


def save_layer(rgba, path):
    from PIL import Image

    Image.fromarray(rgba, "RGBA").save(path)


def candidates(S):
    out = []
    for sid, m in S.meta.items():
        if not m.get("cin_feasible"):
            continue
        k = S.case(sid)
        if not k or k["x0"]["v"] != "fake":
            continue
        a = k["M"].mean()
        if 0.3 <= k["cov"] <= 0.7 and k["gp"] >= 0.7 and 0.04 <= a <= 0.25:
            out.append((abs(k["cov"] - 0.5), k))
    return [k for _, k in sorted(out, key=lambda t: t[0])]


def export(a):
    samples = {s["id"]: s for s in read_jsonl(a.samples)}
    S = Single(a.single_sida, "sida7b", samples)
    k = S.case(a.pick)
    if k is None:
        raise SystemExit(f"{a.pick}: no SIDA stage-2 record")
    sid, M, C = a.pick, k["M"], k["C"]
    xf, xr = load_rgb(k["xf"]), load_rgb(k["xr"])
    s2 = lambda ex: os.path.join(a.single_sida, f"stage2_sida7b_{ex}")
    img = lambda ex, kind="revC", i=0: load_rgb(os.path.join(s2(ex), "images", f"{sid}__fake__{kind}{i}.png"))
    R_rand = load_mask(os.path.join(s2("random"), "regions", f"{sid}__fake__C.png"))
    # matched in-edit control B_1, re-created with the pipeline's own seed
    rng = np.random.default_rng(_seed(sid, "fake", "cin"))
    Bk = blob_in_region(M & ~C, int((C & M).sum()), rng)
    rev = lambda region: np.where(region[..., None], xr, xf)
    queried = {"rev_C": (img("model"), C), "rev_M": (img("oracle"), M), "rev_random": (img("random"), R_rand),
               "rev_B": (img("model", "revCin", 0), Bk)}
    for name, (im, region) in queried.items():
        if region is None or not np.array_equal(im, rev(region)):
            raise SystemExit(f"{sid} {name}: queried image is not R(x_f, region); not exporting")
    out = os.path.join(a.out, "fig2_assets")
    os.makedirs(out, exist_ok=True)
    cs = square_crop(M | C | R_rand)
    w = lambda name, im: save_rgb(im[cs], os.path.join(out, name))

    w("01_xr_raw.png", xr)
    w("02_mask_M_raw.png", np.repeat(M[..., None], 3, axis=2).astype(np.uint8) * 255)
    w("03_xf_raw.png", xf)
    w("03_xf_ov.png", overlay(xf, forged=[M]))
    w("04_xf_answer_ov.png", overlay(xf, red=C, forged=[M]))
    for name, (im, region) in queried.items():
        w(f"05_{name}_raw.png", im)
        w(f"05_{name}_ov.png", overlay(im, forged=[M & ~region] if (M & ~region).any() else [], restored=[region]))
        save_layer(layer(region, GREEN, 140)[cs], os.path.join(out, f"05_{name}_layer.png"))
    save_layer(layer(C, RED, 115)[cs], os.path.join(out, "04_model_mask_C_layer.png"))
    save_layer(layer(M, WHITE, 255)[cs], os.path.join(out, "02_mask_M_layer.png"))

    # zoom on the new boundary that partial reversion leaves inside the edit (for the BA inset)
    from scipy.ndimage import binary_dilation

    seam = binary_dilation(C & M) & (M & ~C)
    ys, xs = np.nonzero(seam)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    if a.zoom:  # centre given in the coordinates of the exported (cropped) thumbnails
        zx, zy = (int(v) for v in a.zoom.split(","))
        cx, cy = cs[1].start + zx, cs[0].start + zy
    side = a.zoom_size or max(48, int(0.35 * (cs[0].stop - cs[0].start)))
    h, W_ = M.shape
    y0, x0 = int(np.clip(cy - side // 2, 0, h - side)), int(np.clip(cx - side // 2, 0, W_ - side))
    zs = (slice(y0, y0 + side), slice(x0, x0 + side))
    save_rgb(img("model")[zs], os.path.join(out, "06_boundary_zoom_raw.png"))
    save_rgb(overlay(img("model"), forged=[M & ~C], restored=[C & M])[zs], os.path.join(out, "06_boundary_zoom_ov.png"))

    # two-object forgery for the component-reversion lane (default: the Figure 1 sample)
    R = Run(a.pairs_sida, "sida7b")
    p = sida_case(R, a.pair)
    if p is None:
        raise SystemExit(f"{a.pair}: not in SIDA's component set")
    pf, pr, pxr = load_rgb(p["xf"]), load_rgb(p["rc"]), load_rgb(p["xr"])
    A, B = p["A"], p["B"]
    if not (np.array_equal(pr[A], pxr[A]) and np.array_equal(pr[~A], pf[~A])):
        raise SystemExit(f"{a.pair}: component image is not R(x_f, A)")
    ps = square_crop(A | B)
    wp = lambda name, im: save_rgb(im[ps], os.path.join(out, name))
    wp("07_pair_xf_raw.png", pf)
    wp("07_pair_xf_ov.png", overlay(pf, red=p["sida0"]["mask"], forged=[A, B]))
    wp("08_pair_revA_raw.png", pr)
    wp("08_pair_revA_ov.png", overlay(pr, red=p["sida1"]["mask"], forged=[B], restored=[A]))
    # the reverted image as it is sent to D (no response mask yet), for the method overview
    wp("08_pair_revA_input_ov.png", overlay(pr, forged=[B], restored=[A]))
    save_layer(layer(A, GREEN, 140)[ps], os.path.join(out, "08_pair_A_layer.png"))
    save_layer(layer(B, WHITE, 255)[ps], os.path.join(out, "08_pair_B_layer.png"))
    save_layer(layer(p["sida0"]["mask"], RED, 115)[ps], os.path.join(out, "07_pair_mask_before_layer.png"))
    save_layer(layer(p["sida1"]["mask"], RED, 115)[ps], os.path.join(out, "08_pair_mask_after_layer.png"))

    print(f"[fig2] wrote {out}/ (thumbnail crop: x {cs[1].start}-{cs[1].stop}, y {cs[0].start}-{cs[0].stop})")
    print(f"       single {sid} ({k['editor']}, {k['edit']}): SIDA fake {k['x0']['s']:.2f}; mask covers "
          f"{100 * k['cov']:.0f}% of the edit, {100 * k['gp']:.0f}% of it on the edit; after restoring it: "
          f"{k['x1']['v']} {k['x1']['s']:.2f}")
    print(f"       pair   {a.pair}: before {p['sida0']['v']} {p['sida0']['s']:.2f}, after A restored "
          f"{p['sida1']['v']} {p['sida1']['s']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="data/full_single/samples.jsonl")
    ap.add_argument("--single-sida", default="runs/full_sida")
    ap.add_argument("--pairs-sida", default="runs/e4b_full_sida")
    ap.add_argument("--pair", default="coco442836p_flux-fill", help="two-object sample (default: Figure 1)")
    ap.add_argument("--pick", default=None, help="single-object sample id")
    ap.add_argument("--zoom", default=None, help="x,y centre of the boundary zoom, in thumbnail pixels")
    ap.add_argument("--zoom-size", type=int, default=0, help="side of the boundary zoom in pixels")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    if a.list or not a.pick:
        samples = {s["id"]: s for s in read_jsonl(a.samples)}
        ks = candidates(Single(a.single_sida, "sida7b", samples))
        print(f"[fig2] {len(ks)} candidates (coverage 30-70%, GP >= 70%, edit 4-25% of the image)")
        for k in ks[:a.top]:
            print(f"  {k['sid']:<28} {k['editor']:<10} {k['edit']:<28} cover {100 * k['cov']:3.0f}%  "
                  f"edit area {100 * k['M'].mean():4.1f}%  fake {k['x0']['s']:.2f} -> {k['x1']['v']} {k['x1']['s']:.2f}")
        return
    export(a)


if __name__ == "__main__":
    main()
