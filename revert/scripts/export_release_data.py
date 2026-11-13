"""Export the released data: the sample manifest and every model output, without any image.

Runs on the machine that holds data/ and runs/. Nothing in data/ or runs/ is modified. Every JSON(L)
record is copied unchanged except for file paths:
  * a path to a binary mask PNG (edit masks, the regions a model cites, controls) becomes `<key>_rle`, the
    mask as COCO RLE (pycocotools), so all analyses can be re-run without images;
  * a path to an image (forgeries, originals, reverted images) is dropped, because images are not released
    (they are regenerated from the manifest);
  * any other absolute path is dropped.
Output files are gzip-compressed JSONL, split into parts below --shard-mb.

    python scripts/export_release_data.py --out ../revert_release/data --dry-run   # what would be written, sizes
    python scripts/export_release_data.py --out ../revert_release/data

Output
  data/manifest/single.csv, two_object.csv   every built sample, including rejected ones (qc_* columns)
  data/manifest/single_final.jsonl.gz, ...   the final 1,200 samples as audited (edit masks as RLE)
  data/outputs/<run>/...                     stage-1 / stage-2 / component answers and metadata, summaries
  data/export_report.txt                     what was exported and what was skipped
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from revert_audit.io import read_jsonl  # noqa: E402

REPORT = []
DILATE_PX = 7  # revert_audit.data.sources.CocoSource default, used for every released sample
INPAINTER = {"sdxl": "sdxl", "flux-fill": "flux_fill", "kandinsky": "kandinsky22"}
RUNS = ["single_sida=runs/full_sida", "single_fakeshield=runs/full_fakeshield",
        "component_sida=runs/e4b_full_sida", "component_fakeshield=runs/e4b_full_fakeshield"]
SKIP_FILE = re.compile(r"(cache|\.failed\.|\.tmp|crosscheck)", re.I)


def note(*a):
    s = " ".join(str(x) for x in a)
    REPORT.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------ masks and paths
_mask_cache: dict = {}


def mask_rle(path):
    """COCO RLE of a binary mask PNG; None if the file is not a binary mask (e.g. an RGB image)."""
    if path in _mask_cache:
        return _mask_cache[path]
    out = None
    try:
        from PIL import Image
        from pycocotools import mask as mu

        with Image.open(path) as im:
            if im.mode in ("L", "1", "P", "LA"):
                a = np.asarray(im.convert("L"))
                vals = np.unique(a)
                if set(vals.tolist()) <= {0, 255} or len(vals) <= 2:
                    r = mu.encode(np.asfortranarray((a > 127).astype(np.uint8)))
                    out = {"size": [int(x) for x in r["size"]], "counts": r["counts"].decode("ascii")}
    except Exception:  # noqa: BLE001
        out = None
    _mask_cache[path] = out
    return out


def looks_like_path(v: str) -> bool:
    v = v.strip()
    if not v or " " in v or "\n" in v:
        return False
    return (v.startswith(("/", "~")) or bool(re.match(r"^[A-Za-z]:[\\/]", v))
            or v.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".jsonl", ".json", ".pt", ".pth", ".bin",
                                   ".safetensors", ".csv", ".xlsx", ".txt", ".log"))
            or v.startswith(("runs/", "data/", "runs\\", "data\\", "./", ".\\")))


STATS = Counter()


def scrub(obj, base_dirs):
    """Recursively replace mask paths by RLE and drop other paths."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and looks_like_path(v):
                p = resolve(v, base_dirs)
                if p and p.lower().endswith(".png"):
                    r = mask_rle(p)
                    if r is not None:
                        out[f"{k}_rle"] = r
                        STATS["mask_to_rle"] += 1
                        continue
                    STATS["image_dropped"] += 1
                    continue
                if p is None and v.lower().endswith(".png"):
                    STATS["missing_png_dropped"] += 1
                    continue
                STATS["path_dropped"] += 1
                continue
            out[k] = scrub(v, base_dirs)
        return out
    if isinstance(obj, list):
        return [scrub(v, base_dirs) for v in obj]
    return obj


def resolve(v, base_dirs):
    v = os.path.expanduser(v)
    if os.path.isabs(v):
        return v if os.path.exists(v) else None
    for b in base_dirs:
        p = os.path.join(b, v)
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ writers
class ShardWriter:
    """gzip JSONL, a new part whenever the compressed size passes the limit."""

    def __init__(self, path, limit_mb, dry):
        self.base, self.limit, self.dry = path, limit_mb * 2**20, dry
        self.part, self.n, self.f, self.raw, self.files = 0, 0, None, None, []
        self.bytes_total = 0

    def _open(self):
        self.part += 1
        name = self.base if self.part == 1 else self.base.replace(".jsonl.gz", f".part{self.part:02d}.jsonl.gz")
        self.files.append(name)
        if self.dry:
            self.raw = io.BytesIO()
        else:
            os.makedirs(os.path.dirname(name), exist_ok=True)
            self.raw = open(name, "wb")
        self.f = gzip.GzipFile(fileobj=self.raw, mode="wb", mtime=0)

    def write(self, rec):
        if self.f is None:
            self._open()
        self.f.write((json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        self.n += 1
        if self.n % 2000 == 0:
            self.f.flush()
            if self.raw.tell() > self.limit:
                self.close()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.bytes_total += self.raw.tell()
            if not self.dry:
                self.raw.close()
            self.f = None
        if len(self.files) > 1 and not self.dry:  # name the first part consistently once there are several
            first = self.files[0]
            if os.path.exists(first) and ".part" not in first:
                new = first.replace(".jsonl.gz", ".part01.jsonl.gz")
                os.replace(first, new)
                self.files[0] = new


def export_jsonl(src, dst, base_dirs, a):
    w = ShardWriter(dst, a.shard_mb, a.dry_run)
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                w.write(scrub(json.loads(line), base_dirs))
    w.close()
    return w


def export_json(src, dst, base_dirs, a):
    try:
        obj = scrub(json.load(open(src, encoding="utf-8")), base_dirs)
    except Exception as e:  # noqa: BLE001
        note(f"   skipped unreadable {src}: {e}")
        return 0
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    if not a.dry_run:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8").write(s)
    return len(s)


# ------------------------------------------------------------------ runs
def export_run(name, run, a):
    note(f"\n== run {name}: {run}")
    if not os.path.isdir(run):
        note("   not found, skipped")
        return 0
    total = 0
    base_dirs = [os.getcwd(), run]
    for src in sorted(glob.glob(os.path.join(run, "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(src, run)
        if SKIP_FILE.search(rel) or os.sep + "images" + os.sep in os.sep + rel:
            note(f"   skip {rel}")
            continue
        dst = os.path.join(a.out, "outputs", name, rel + ".gz")
        w = export_jsonl(src, dst, base_dirs, a)
        total += w.bytes_total
        note(f"   {rel}: {w.n} records -> {len(w.files)} file(s), {w.bytes_total / 2**20:.1f} MB")
    for src in sorted(glob.glob(os.path.join(run, "**", "*.json"), recursive=True)):
        rel = os.path.relpath(src, run)
        if SKIP_FILE.search(rel):
            continue
        n = export_json(src, os.path.join(a.out, "outputs", name, rel), base_dirs, a)
        total += n
    return total


# ------------------------------------------------------------------ manifest
def read_review(path):
    try:
        from revert_audit.data.qc import _read_review

        return _read_review(path)
    except Exception:  # noqa: BLE001
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))


def is_bad(v):
    return (v or "").strip().lower() in ("bad", "0", "no", "n", "x", "false", "坏", "差", "删", "删除")


def target_of(prompt):
    m = re.match(r"a photo of an? (.+?), realistic", prompt or "")
    return m.group(1) if m else ""


def image_size(rec):
    try:
        from PIL import Image

        with Image.open(rec["real"]) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return (None, None)


def build_manifest(set_name, final_path, build_dirs, a, coco_cats, coco_anns):
    note(f"\n== manifest {set_name}: final {final_path}; builds {build_dirs}")
    final = read_jsonl(final_path)
    final_ids = {r["id"] for r in final}
    rows, missing = [], []
    for d in build_dirs:
        built = {r["id"]: r for r in read_jsonl(os.path.join(d, "samples.jsonl"))}
        auto_rej = {r["id"]: r for r in read_jsonl(os.path.join(d, "rejected.jsonl"))}
        reviewed_p = os.path.join(d, "samples_reviewed.jsonl")
        reviewed = {r["id"] for r in read_jsonl(reviewed_p)} if os.path.exists(reviewed_p) else None
        marks, ext_ids, seen_in_review = {}, set(), set()
        for p in sorted(glob.glob(os.path.join(d, "qc", "qc_review*.csv")) + glob.glob(os.path.join(d, "qc", "qc_review*.xlsx"))):
            for r in read_review(p):
                sid = (r.get("id") or "").strip()
                if not sid:
                    continue
                seen_in_review.add(sid)
                if is_bad(r.get("ok")):
                    marks[sid] = (r.get("note") or "").strip() or "bad"
                if "new" in os.path.basename(p):
                    ext_ids.add(sid)
        ids = list(built) + [i for i in auto_rej if i not in built] + sorted(seen_in_review - set(built) - set(auto_rej))
        for sid in ids:
            r = built.get(sid) or auto_rej.get(sid)
            if r is None:
                missing.append(sid)
                r = {"id": sid}
            manual_keep = (sid in reviewed) if reviewed is not None else (sid not in marks)
            if sid in marks:
                manual_keep = False
            w, h = image_size(r) if r.get("real") else (None, None)
            row = {"sample_id": sid, "set": set_name, "inpainter": INPAINTER.get(r.get("editor"), r.get("editor")),
                   "build": os.path.basename(os.path.normpath(d)),
                   "batch": "extension" if sid in ext_ids else "initial",
                   "coco_image_id": r.get("coco_image_id"), "width": w, "height": h, "mask_dilate_px": DILATE_PX,
                   "crop": "centre crop to multiples of 64, no resizing", "mask_frac": r.get("mask_frac"),
                   "qc_auto_pass": sid not in auto_rej and sid in built,
                   "qc_auto_reason": (auto_rej.get(sid) or {}).get("reason", ""),
                   "qc_manual_keep": manual_keep, "qc_manual_note": marks.get(sid, ""),
                   "in_final_set": sid in final_ids, "record_found": sid not in missing}
            if r.get("parts"):
                for i, (pt, tag) in enumerate(zip(r["parts"], "AB")):
                    ann = pt.get("ann_id")
                    row.update({f"coco_ann_id_{tag}": ann,
                                f"source_category_{tag}": coco_cats.get(coco_anns.get(ann), pt.get("obj", "")),
                                f"edit_type_{tag}": pt.get("edit_type"),
                                f"target_category_{tag}": target_of(pt.get("prompt")),
                                f"prompt_{tag}": pt.get("prompt"), f"seed_{tag}": pt.get("seed"),
                                f"mask_frac_{tag}": pt.get("mask_frac")})
                row["sample_seed"] = r.get("seed")
            else:
                ann = r.get("coco_ann_id")
                row.update({"coco_ann_id": ann, "source_category": coco_cats.get(coco_anns.get(ann), ""),
                            "edit_type": r.get("edit_type"), "target_category": target_of(r.get("prompt")),
                            "prompt": r.get("prompt"), "seed": r.get("seed"),
                            "change_in_mask": r.get("change_in_mask")})
            rows.append(row)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out_csv = os.path.join(a.out, "manifest", f"{set_name}.csv")
    if not a.dry_run:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
    by = defaultdict(Counter)
    for r in rows:
        by[r["inpainter"]]["built"] += 1
        by[r["inpainter"]]["auto_pass"] += bool(r["qc_auto_pass"])
        by[r["inpainter"]]["manual_keep"] += bool(r["qc_manual_keep"])
        by[r["inpainter"]]["final"] += bool(r["in_final_set"])
        by[r["inpainter"]]["extension"] += r["batch"] == "extension"
    for ed, c in sorted(by.items()):
        note(f"   {ed}: {dict(c)}")
    if missing:
        note(f"   WARNING: {len(missing)} ids appear only in a review table (record not found, no prompt/seed): "
             f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    lost = final_ids - {r["sample_id"] for r in rows}
    if lost:
        note(f"   WARNING: {len(lost)} final samples are not in any build directory: {sorted(lost)[:5]}")
    # final samples as audited, edit masks as RLE
    w = ShardWriter(os.path.join(a.out, "manifest", f"{set_name}_final.jsonl.gz"), a.shard_mb, a.dry_run)
    for r in final:
        w.write(scrub(r, [os.getcwd()]))
    w.close()
    note(f"   final samples: {w.n} -> {w.bytes_total / 2**20:.1f} MB; manifest rows {len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="release data folder (created), e.g. ../revert_release/data")
    ap.add_argument("--single", default="data/full_single/samples.jsonl")
    ap.add_argument("--single-builds", default="data/full_single_sdxl,data/full_single_flux,data/full_single_kand")
    ap.add_argument("--pairs", default="data/full_pairs/samples.jsonl")
    ap.add_argument("--pairs-builds", default="data/full_pairs_sdxl,data/full_pairs_flux,data/full_pairs_kand")
    ap.add_argument("--runs", nargs="*", default=RUNS, help="name=path pairs")
    ap.add_argument("--extra-runs", nargs="*", default=[],
                    help="further name=path runs, e.g. pilot_sofake=runs/... pilot_qwen=runs/... pilot_seam_sida=runs/...")
    ap.add_argument("--coco-ann", default=None, help="instances_val2017.json, for the source category column")
    ap.add_argument("--shard-mb", type=float, default=45.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cats, anns = {}, {}
    if a.coco_ann and os.path.exists(a.coco_ann):
        d = json.load(open(a.coco_ann, encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        anns = {x["id"]: x["category_id"] for x in d["annotations"]}
    note(f"export to {os.path.basename(os.path.normpath(a.out))}/{' (dry run: nothing written)' if a.dry_run else ''}")
    build_manifest("single", a.single, a.single_builds.split(","), a, cats, anns)
    build_manifest("two_object", a.pairs, a.pairs_builds.split(","), a, cats, anns)
    total = 0
    for spec in a.runs + a.extra_runs:
        name, _, path = spec.partition("=")
        total += export_run(name, path, a)
    note(f"\npaths: {dict(STATS)}")
    note(f"total compressed outputs: {total / 2**20:.1f} MB")
    big = [(os.path.getsize(p), p) for p in glob.glob(os.path.join(a.out, "**", "*"), recursive=True)
           if os.path.isfile(p)] if not a.dry_run else []
    for s, p in sorted(big, reverse=True)[:5]:
        note(f"   largest: {s / 2**20:.1f} MB  {os.path.relpath(p, a.out)}")
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "export_report.txt"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
