"""Build samples.jsonl from existing folders of (real, raw-edited, mask).

Typical use: the INP-X benchmark (Kaggle: emirhanbilgic/inpainting-exchange),
Semi-Truths, COCO-Inpaint, or any dataset that ships the original image next
to the edited one. Files are matched by filename stem (or by a regex).

    python -m revert_audit.data.folder_adapter \
        --real-dir REAL --raw-dir INPAINTED --mask-dir MASKS \
        --out data/inpx_pilot --limit 500 --editor sd14 --source celebahq

If a dataset has no masks but has an INP-X style 'exchanged' image (raw
pixels inside the mask, real pixels outside), pass --exchanged-dir instead of
--mask-dir and the mask is recovered by differencing.

Check the folder layout of the dataset you download first; adjust
--real-glob/--raw-glob/--stem-regex rather than renaming files.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from ..interventions import composite, mask_from_difference
from ..io import load_mask, load_rgb, save_mask, save_rgb, write_jsonl


def _index(d: str, pattern: str, stem_re: str | None) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem_re:
            m = re.search(stem_re, stem)
            if not m:
                continue
            stem = m.group(1)
        out[stem] = p
    return out


def build(real_dir, raw_dir, out, mask_dir=None, exchanged_dir=None, limit=None, seed=0,
          real_glob="*.*", raw_glob="*.*", mask_glob="*.*", stem_regex=None,
          editor="unknown", source="unknown", edit_type="inpaint", min_mask=0.005, max_mask=0.5):
    if not (mask_dir or exchanged_dir):
        raise ValueError("need --mask-dir or --exchanged-dir")
    reals = _index(real_dir, real_glob, stem_regex)
    raws = _index(raw_dir, raw_glob, stem_regex)
    masks = _index(mask_dir or exchanged_dir, mask_glob, stem_regex)
    stems = sorted(set(reals) & set(raws) & set(masks))
    print(f"[folder_adapter] real={len(reals)} raw={len(raws)} mask={len(masks)} matched={len(stems)}")
    rng = np.random.default_rng(seed)
    if limit and len(stems) > limit:
        stems = sorted(rng.choice(stems, limit, replace=False).tolist())
    rows, skip = [], {"shape": 0, "mask_size": 0}
    for st in stems:
        real, raw = load_rgb(reals[st]), load_rgb(raws[st])
        if real.shape != raw.shape:
            skip["shape"] += 1
            continue
        if mask_dir:
            M = load_mask(masks[st])
            if M.shape != real.shape[:2]:
                skip["shape"] += 1
                continue
        else:
            M = mask_from_difference(real, load_rgb(masks[st]), threshold=3.0, dilate_px=2)
        frac = M.mean()
        if not (min_mask <= frac <= max_mask):
            skip["mask_size"] += 1
            continue
        sid = re.sub(r"[^A-Za-z0-9_\-]", "_", f"{source}_{editor}_{st}")
        P = lambda k: os.path.join(out, "images", f"{sid}_{k}.png")
        save_rgb(real, P("real"))
        save_rgb(raw, P("raw"))
        save_rgb(composite(raw, real, M), P("fake"))
        save_mask(M, P("mask"))
        rows.append({"id": sid, "real": P("real"), "raw": P("raw"), "fake": P("fake"), "mask": P("mask"),
                     "editor": editor, "source": source, "edit_type": edit_type, "mask_frac": float(frac)})
    path = os.path.join(out, "samples.jsonl")
    write_jsonl(rows, path)
    print(f"[folder_adapter] wrote {len(rows)} samples to {path}; skipped {skip}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--mask-dir")
    ap.add_argument("--exchanged-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-glob", default="*.*")
    ap.add_argument("--raw-glob", default="*.*")
    ap.add_argument("--mask-glob", default="*.*")
    ap.add_argument("--stem-regex", help="regex with one group extracting the shared id from each filename stem")
    ap.add_argument("--editor", default="unknown")
    ap.add_argument("--source", default="unknown")
    ap.add_argument("--edit-type", default="inpaint")
    a = ap.parse_args(argv)
    build(a.real_dir, a.raw_dir, a.out, a.mask_dir, a.exchanged_dir, a.limit, a.seed, a.real_glob, a.raw_glob,
          a.mask_glob, a.stem_regex, a.editor, a.source, a.edit_type)


if __name__ == "__main__":
    main()
