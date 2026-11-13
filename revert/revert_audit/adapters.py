"""Convert outputs of external detectors (run in their own repos / conda
envs) into the standard answers.jsonl.

Workflow for an external detector (SIDA, FakeShield, Veritas, FakeVLM,
AIGI-Holmes, ...):
  1. stage1/stage2 write   <dir>/queries.jsonl   and   <dir>/images/*.png
  2. in the detector's own repo, loop over <dir>/images/*.png and save, per
     image stem, the raw text output to  <out>/<stem>.txt  and (if the model
     predicts a forgery mask) the mask to <out>/<stem>.png
  3. python -m revert_audit.adapters --queries <dir>/queries.jsonl \
         --text-dir <out> [--mask-dir <out>] --coord abs|rel1000|rel1 \
         --out <dir>/answers_<name>.jsonl

Hard-label models (no probability) get p_fake = None; the metrics then use
the 0/1 verdict, which is coarser but still valid (report it as such).
"""

from __future__ import annotations

import argparse
import os

from .io import load_rgb, read_jsonl, write_jsonl
from .parsing import parse_response


def _scale_box(b, coord: str, h: int, w: int):
    x1, y1, x2, y2 = b
    if coord == "abs":
        return [x1, y1, x2, y2]
    if coord == "rel1000":
        return [x1 * w / 1000.0, y1 * h / 1000.0, x2 * w / 1000.0, y2 * h / 1000.0]
    if coord == "rel1":
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    raise ValueError(coord)


def convert(queries_path: str, out_path: str, text_dir: str | None = None, mask_dir: str | None = None,
            coord: str = "abs", detector_name: str = "external", fake_word_override: dict | None = None) -> str:
    rows = []
    missing = 0
    for q in read_jsonl(queries_path):
        stem = os.path.splitext(os.path.basename(q["path"]))[0]
        text = ""
        if text_dir:
            tp = os.path.join(text_dir, stem + ".txt")
            if not os.path.exists(tp):
                missing += 1
                continue
            with open(tp, "r", encoding="utf-8") as f:
                text = f.read()
        parsed = parse_response(text)
        if fake_word_override:  # e.g. SIDA prints "tampered" / "full synthetic" / "real"
            low = text.lower()
            for k, v in fake_word_override.items():
                if k in low:
                    parsed["verdict"] = v
                    break
        h, w = load_rgb(q["path"]).shape[:2]
        mp = None
        if mask_dir:
            cand = os.path.join(mask_dir, stem + ".png")
            mp = cand if os.path.exists(cand) else None
        rows.append({
            "query_id": q["query_id"],
            "verdict": parsed["verdict"],
            "p_fake": None,
            "boxes": [_scale_box(b, coord, h, w) for b in parsed["boxes"]],
            "mask_path": mp,
            "artifacts": parsed["artifacts"],
            "raw_text": text,
            "parse_mode": parsed["parse_mode"],
            "detector": detector_name,
        })
    write_jsonl(rows, out_path)
    print(f"[adapters] wrote {len(rows)} answers, {missing} queries had no output file")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-dir")
    ap.add_argument("--mask-dir")
    ap.add_argument("--coord", default="abs", choices=["abs", "rel1000", "rel1"])
    ap.add_argument("--name", default="external")
    ap.add_argument("--sida", action="store_true", help="map SIDA class words to verdicts")
    a = ap.parse_args(argv)
    # order matters: 'tampered' / 'synthetic' are checked before 'real'
    override = {"tampered": "fake", "synthetic": "fake", "real": "real"} if a.sida else None
    convert(a.queries, a.out, a.text_dir, a.mask_dir, a.coord, a.name, override)


if __name__ == "__main__":
    main()
