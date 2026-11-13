"""Build a ReVert dataset: real photo -> edit -> (x_r, x_raw, M, x_f).

    # COCO (download first: python -m revert_audit.data.download --root /path/to/coco)
    python -m revert_audit.data.build --source coco --coco-root /path/to/coco --split val2017 \\
        --editor sdxl --n 150 --out data\\pilot_sdxl
    python -m revert_audit.data.build --source coco --coco-root /path/to/coco --split val2017 \\
        --editor flux-fill --seq-offload --n 150 --out data\\pilot_flux
    python -m revert_audit.data.build --source coco --coco-root /path/to/coco --editor kontext --seq-offload ...

    # your own photos / a high-resolution folder (inpainting editors only)
    python -m revert_audit.data.build --source folder --folder /path/to/photos --name ownphotos \\
        --editor flux-fill --seq-offload --n 100 --out data\\own_flux

    # then mix several builds into one shuffled samples.jsonl for the audit
    python -m revert_audit.data.merge --inputs data\\pilot_sdxl data\\pilot_flux --out data\\pilot

Every run is resumable (already kept or rejected ids are skipped) and writes:
    samples.jsonl     kept samples (with prompt, seed, QC statistics, split)
    rejected.jsonl    rejected attempts with the reason (report the counts!)
    build_stats.json  kept / rejected per reason, seconds per image
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from ..interventions import composite, mask_from_difference
from ..io import append_jsonl, read_jsonl, save_mask, save_rgb
from .qc import check_sample

STYLES = ["a watercolor painting", "an oil painting", "a pencil sketch", "a cartoon", "Van Gogh style brushstrokes",
          "a ukiyo-e woodblock print"]
COLORS = ["bright red", "blue", "green", "golden", "purple"]
GENERIC_OBJECTS = ["a potted plant", "a cat", "a dog", "a bicycle", "a backpack", "a vase of flowers", "a chair",
                   "a traffic cone", "a teddy bear", "a lamp"]


def _article(thing: str) -> str:
    return thing if thing.startswith(("a ", "an ")) else ("an " if thing[:1].lower() in "aeiou" else "a ") + thing


def inpaint_spec(rng, obj: str | None, other: str | None):
    """(descriptive prompt, edit type, replacement thing or None). The rng draws are the same for every editor."""
    if rng.random() < 0.6:
        thing = other or str(rng.choice(GENERIC_OBJECTS))
        a = thing if thing.startswith("a ") else "a " + thing
        return f"a photo of {a}, realistic, natural lighting", "replace", thing
    return "empty background, seamless, realistic photo", "remove", None


def inpaint_prompt(rng, obj: str | None, other: str | None, instruct: bool = False):
    """Inpainting editors get the descriptive prompt. The mask-based EDIT model (Qwen-Image-Edit inpainting) gets the
    same edit as an instruction about the masked area, which it sees filled with gray (editors.py): like the
    dedicated inpainting models it does not see the original object."""
    prompt, etype, thing = inpaint_spec(rng, obj, other)
    if instruct:
        if etype == "replace":
            prompt = f"Fill the gray area with {_article(thing)} that fits the scene. Keep everything else unchanged."
        else:
            prompt = "Fill the gray area so that it continues the surrounding background. Keep everything else unchanged."
    return prompt, etype


def instruction(rng, obj: str, other: str):
    r = rng.random()
    if r < 0.35:
        return f"Replace the {obj} with a {other}. Keep everything else exactly the same.", "replace"
    if r < 0.55:
        return f"Change the color of the {obj} to {rng.choice(COLORS)}. Keep everything else exactly the same.", "color"
    if r < 0.85:
        return f"Make only the {obj} look like {rng.choice(STYLES)}. Keep everything else exactly the same.", "style"
    return f"Add a small {other} next to the {obj}. Keep everything else exactly the same.", "add"


def _edit(editor, image, mask, prompt, seed):
    try:
        return editor.edit(image, mask, prompt, seed)
    except Exception as e:  # noqa: BLE001
        if "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError":
            from ..gpu import explain_oom

            explain_oom(e)
        raise


def _build_pair(item, editor, rng, real, s_seed, out, sid, split, min_frac, max_frac, source):
    """Independent edits of the two objects from the same real photo; x_f takes
    each object's pixels from its own edit and real pixels everywhere else.
    No x_raw exists for a two-object sample (each part passed the VAE once, separately)."""
    P = lambda k: os.path.join(out, "images", f"{sid}_{k}.png")
    fake = real.copy()
    parts, part_stats = [], []
    for i, part in enumerate(item.parts):
        other = source.other_category(part["obj"]) if hasattr(source, "other_category") else None
        prompt, etype = inpaint_prompt(rng, part["obj"], other, getattr(editor, "instruct_prompts", False))
        seed_i = int(rng.integers(0, 2**31 - 1))
        out_img = _edit(editor, item.image, part["mask"], prompt, seed_i)
        if out_img.size != item.image.size:
            out_img = out_img.resize(item.image.size, resample=3)
        raw_i = np.asarray(out_img.convert("RGB"))
        ok, reason, stats = check_sample(real, raw_i, part["mask"], "inpaint", min_frac, max_frac)
        part_stats.append(stats)
        if not ok:
            return False, {"id": sid, "editor": editor.name, "split": split, "reason": f"part{i + 1}_{reason}",
                           "pair": True, **item.meta, "part_stats": part_stats}
        fake[part["mask"]] = raw_i[part["mask"]]
        parts.append({"obj": part["obj"], "ann_id": part["ann_id"], "prompt": prompt, "edit_type": etype,
                      "seed": seed_i, "raw": raw_i, "mask": part["mask"], **stats})
    save_rgb(real, P("real"))
    save_rgb(fake, P("fake"))
    save_mask(item.mask, P("mask"))
    recs = []
    for i, pt in enumerate(parts, 1):
        save_rgb(pt.pop("raw"), P(f"raw{i}"))
        save_mask(pt.pop("mask"), P(f"mask{i}"))
        recs.append({**pt, "raw": P(f"raw{i}"), "mask": P(f"mask{i}")})
    rec = {"id": sid, "editor": editor.name, "edit_type": "pair", "seed": s_seed, "split": split, "pair": True,
           **item.meta, "mask_frac": round(float(item.mask.mean()), 4), "parts": recs,
           "real": P("real"), "fake": P("fake"), "mask": P("mask")}
    return True, rec


def build(source, editor, out: str, n: int, seed: int = 0, split: str = "test", min_frac=0.01, max_frac=0.4,
          diff_threshold: float = 14.0, max_attempts: int | None = None) -> dict:
    os.makedirs(out, exist_ok=True)
    sp, rp = os.path.join(out, "samples.jsonl"), os.path.join(out, "rejected.jsonl")
    kept_ids = {r["id"] for r in read_jsonl(sp)}
    rej_ids = {r["id"] for r in read_jsonl(rp)}
    kept, reasons, secs = len(kept_ids), Counter(r["reason"] for r in read_jsonl(rp)), []
    rng = np.random.default_rng(seed)
    attempts = 0
    for item in source:
        if kept >= n or (max_attempts and attempts >= max_attempts):
            break
        sid = f"{item.sid}_{editor.name.replace(':', '-')}"
        if sid in kept_ids or sid in rej_ids:
            continue
        attempts += 1
        real = np.asarray(item.image)
        other = source.other_category(item.obj) if (item.obj and hasattr(source, "other_category")) else None
        s_seed = int(rng.integers(0, 2**31 - 1))
        t0 = time.time()
        if item.parts:
            if editor.kind != "inpaint":
                raise ValueError("two-object samples (--pairs) need an inpainting editor")
            ok, rec = _build_pair(item, editor, rng, real, s_seed, out, sid, split, min_frac, max_frac, source)
            secs.append(time.time() - t0)
            if not ok:
                append_jsonl(rec, rp)
                reasons[rec["reason"]] += 1
                continue
            append_jsonl(rec, sp)
            kept += 1
            if kept % 10 == 0:
                print(f"[build] kept {kept}/{n}  rejected {dict(reasons)}  {np.mean(secs):.0f} s/img", flush=True)
            continue
        if editor.kind == "inpaint":
            prompt, etype = inpaint_prompt(rng, item.obj, other, getattr(editor, "instruct_prompts", False))
            out_img = _edit(editor, item.image, item.mask, prompt, s_seed)
            M = item.mask
        else:
            if not item.obj:
                raise ValueError("instruction editors need object names: use --source coco")
            prompt, etype = instruction(rng, item.obj, other)
            out_img = _edit(editor, item.image, None, prompt, s_seed)
        if out_img.size != item.image.size:
            out_img = out_img.resize(item.image.size, resample=3)  # editors may change resolution
        raw = np.asarray(out_img.convert("RGB"))
        if editor.kind == "instruct":
            M = mask_from_difference(real, raw, threshold=diff_threshold, dilate_px=6)
        secs.append(time.time() - t0)
        ok, reason, stats = check_sample(real, raw, M, editor.kind, min_frac, max_frac)
        rec = {"id": sid, "editor": editor.name, "edit_type": etype, "prompt" if editor.kind == "inpaint" else "instruction": prompt,
               "seed": s_seed, "split": split, **item.meta, **stats}
        if not ok:
            rec["reason"] = reason
            append_jsonl(rec, rp)
            reasons[reason] += 1
            continue
        P = lambda k: os.path.join(out, "images", f"{sid}_{k}.png")
        save_rgb(real, P("real"))
        save_rgb(raw, P("raw"))
        save_rgb(composite(raw, real, M), P("fake"))
        save_mask(M, P("mask"))
        rec.update({"real": P("real"), "raw": P("raw"), "fake": P("fake"), "mask": P("mask")})
        append_jsonl(rec, sp)
        kept += 1
        if kept % 10 == 0:
            print(f"[build] kept {kept}/{n}  rejected {dict(reasons)}  {np.mean(secs):.0f} s/img", flush=True)
    stats = {"editor": editor.name, "kept": kept, "target": n, "rejected": dict(reasons),
             "sec_per_attempt": float(np.mean(secs)) if secs else None,
             "acceptance_rate": kept / max(1, kept + sum(reasons.values()))}
    with open(os.path.join(out, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[build] done: {stats}")
    if kept < n:
        print(f"[build] WARNING: only {kept} of {n} kept (source exhausted or --max-attempts reached)")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["coco", "folder"], required=True)
    ap.add_argument("--coco-root")
    ap.add_argument("--split", default="val2017", help="COCO split; val2017 -> test set, train2017 -> training set")
    ap.add_argument("--folder")
    ap.add_argument("--name", default="photos", help="name of a folder source (used in ids)")
    ap.add_argument("--set", default=None, choices=[None, "test", "train"], help="override the split label")
    ap.add_argument("--editor", required=True, choices=["sdxl", "flux-fill", "kontext", "qwen-edit", "qwen-image", "qwen-edit-inpaint", "kandinsky", "api-openai", "api-gemini"])
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--editor-path", default=None,
                    help="local folder of a manually downloaded model, e.g. /path/to/FLUX.1-Kontext-dev")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--allow-upscale", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="same seed = same real photos for every editor (paired design)")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--seq-offload", action="store_true")
    ap.add_argument("--nf4", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--vram-cap", type=float, default=None, help="e.g. 0.92: see revert_audit/gpu.py")
    ap.add_argument("--pairs", action="store_true", help="two-object samples for the seam-free test (E4b), COCO + inpainting")
    ap.add_argument("--min-gap", type=int, default=24, help="--pairs: min px between the two dilated object masks")
    ap.add_argument("--exclude-samples", nargs="*", default=[],
                    help="samples.jsonl / rejected.jsonl files whose COCO photos must not be used again (fresh data)")
    a = ap.parse_args(argv)

    from ..gpu import cap_vram

    cap_vram(a.vram_cap)

    from .editors import api_key_present, build_editor
    from .sources import CocoSource, FolderSource

    if not api_key_present(a.editor):
        raise SystemExit(f"{a.editor}: API key environment variable not set")
    if a.source == "coco":
        if not a.coco_root:
            raise SystemExit("--coco-root is required for --source coco")
        excl = {r.get("coco_image_id") for f in a.exclude_samples for r in read_jsonl(f) if r.get("coco_image_id")}
        if excl:
            print(f"[build] excluding {len(excl)} COCO photos used before")
        src = CocoSource(a.coco_root, a.split, a.size, a.seed, allow_upscale=a.allow_upscale, pairs=a.pairs,
                         min_gap=a.min_gap, exclude_image_ids=excl)
        split = a.set or ("test" if a.split.startswith("val") else "train")
    else:
        if a.pairs:
            raise SystemExit("--pairs needs --source coco (object masks)")
        if not a.folder:
            raise SystemExit("--folder is required for --source folder")
        src = FolderSource(a.folder, a.size, a.seed, name=a.name, allow_upscale=a.allow_upscale)
        split = a.set or "test"
    ed = build_editor(a.editor, a.offload, a.seq_offload, a.nf4, a.api_model, a.steps, a.editor_path)
    build(src, ed, a.out, a.n, a.seed, split, max_attempts=a.max_attempts)


if __name__ == "__main__":
    main()
