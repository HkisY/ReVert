"""Regenerate the forgeries of the paper from the released manifest.

For every sample of data/manifest/<set>_final.jsonl.gz the COCO photograph is loaded and cropped exactly as
during the original build (revert_audit.data.sources.fit_to_grid: no resizing, centre crop to multiples of
64), the released edit mask M (COCO RLE, already dilated by 7 px) is used as is, and the inpainting model is
run with the sample's own prompt and seed. The composite forgery is x_f = M * x_raw + (1 - M) * x_r
(two-object samples: each object is edited separately from the same photograph and composed).

    python scripts/generate_from_manifest.py --set single --coco-root /path/to/coco --editor sdxl --out regen/single
    python scripts/generate_from_manifest.py --set two_object --coco-root /path/to/coco --editor kandinsky \
        --ids coco123456p_kandinsky --out regen/two_object
    # optional: compare with images you already have (pixel by pixel)
    python scripts/generate_from_manifest.py ... --compare-dir /path/to/original/images

Outputs per sample: <id>_real.png, <id>_mask.png, <id>_raw.png (single-object only), <id>_fake.png, and
samples.jsonl in the format the audit expects (python -m revert_audit.cli pilot --samples ...).
Different GPUs or library versions can change generated pixels slightly; the versions we used are listed
in environment/audit.yml.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from revert_audit.data.sources import fit_to_grid  # noqa: E402
from revert_audit.interventions import composite  # noqa: E402
from revert_audit.io import load_rgb, save_mask, save_rgb  # noqa: E402

FLAGS = {"flux-fill": {"seq_offload": True}}  # memory option only; does not change the computation


def rle_decode(r):
    from pycocotools import mask as mu

    return mu.decode({"size": r["size"], "counts": r["counts"].encode("ascii")}).astype(bool)


def read_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def coco_photo(coco_root, image_id, split="val2017"):
    from PIL import Image

    return Image.open(os.path.join(coco_root, split, f"{int(image_id):012d}.jpg")).convert("RGB")


def run_editor(editor, image, mask, prompt, seed):
    out = editor.edit(image, mask, prompt, int(seed))
    if out.size != image.size:
        out = out.resize(image.size, resample=3)
    return np.asarray(out.convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["single", "two_object"], required=True)
    ap.add_argument("--manifest-dir", default="data/manifest")
    ap.add_argument("--coco-root", required=True, help="folder with val2017/ (python -m revert_audit.data.download)")
    ap.add_argument("--editor", required=True, choices=["sdxl", "flux-fill", "kandinsky"])
    ap.add_argument("--ids", nargs="*", default=None, help="only these sample ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-dir", default=None, help="folder with <id>_fake.png etc. to compare against")
    ap.add_argument("--editor-path", default=None, help="local copy of the model (diffusers layout)")
    a = ap.parse_args()

    rows = [r for r in read_manifest(os.path.join(a.manifest_dir, f"{a.set}_final.jsonl.gz")) if r.get("editor") == a.editor]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    rows = rows[: a.limit or None]
    print(f"[regen] {len(rows)} samples ({a.set}, {a.editor})")
    from revert_audit.data.editors import build_editor

    fl = FLAGS.get(a.editor, {})
    ed = build_editor(a.editor, offload=fl.get("offload", False), seq_offload=fl.get("seq_offload", False),
                      path=a.editor_path)
    os.makedirs(a.out, exist_ok=True)
    out_rows, n_same, n_cmp = [], 0, 0
    for r in rows:
        sid = r["id"]
        img, _ = fit_to_grid(coco_photo(a.coco_root, r["coco_image_id"]), None)
        real = np.asarray(img)
        P = lambda k: os.path.join(a.out, f"{sid}_{k}.png")  # noqa: E731
        rec = {k: v for k, v in r.items() if not k.endswith("_rle") and k != "parts"}
        if r.get("parts"):
            fake = real.copy()
            parts = []
            for i, pt in enumerate(r["parts"], 1):
                m = rle_decode(pt["mask_rle"])
                raw_i = run_editor(ed, img, m, pt["prompt"], pt["seed"])
                fake[m] = raw_i[m]
                save_rgb(raw_i, P(f"raw{i}"))
                save_mask(m, P(f"mask{i}"))
                parts.append({**{k: v for k, v in pt.items() if not k.endswith("_rle")}, "raw": P(f"raw{i}"),
                              "mask": P(f"mask{i}")})
            M = rle_decode(r["mask_rle"])
            rec["parts"] = parts
        else:
            M = rle_decode(r["mask_rle"])
            raw = run_editor(ed, img, M, r["prompt"], r["seed"])
            fake = composite(raw, real, M)
            save_rgb(raw, P("raw"))
            rec["raw"] = P("raw")
        if M.shape != real.shape[:2]:
            raise SystemExit(f"{sid}: mask {M.shape} does not match the cropped photograph {real.shape[:2]}")
        save_rgb(real, P("real"))
        save_rgb(fake, P("fake"))
        save_mask(M, P("mask"))
        rec.update({"real": P("real"), "fake": P("fake"), "mask": P("mask")})
        out_rows.append(rec)
        if a.compare_dir:
            ref = os.path.join(a.compare_dir, f"{sid}_fake.png")
            if os.path.exists(ref):
                n_cmp += 1
                d = np.abs(load_rgb(ref).astype(int) - fake.astype(int))
                n_same += int(d.max() == 0)
                print(f"   {sid}: max |diff| {d.max()}, mean {d.mean():.4f}, pixels differing {(d.max(-1) > 0).mean():.4%}")
    with open(os.path.join(a.out, "samples.jsonl"), "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if n_cmp:
        print(f"[regen] identical to the reference: {n_same}/{n_cmp}")
    print(f"[regen] wrote {len(out_rows)} samples to {a.out}")


if __name__ == "__main__":
    main()
