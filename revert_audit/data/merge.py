"""Merge several builds into one shuffled samples.jsonl.

The progressive pilot audits the FIRST 50, then 150, then 300 samples, so the
merged file is interleaved round-robin across inputs (after shuffling each):
every prefix contains every editor in equal proportion.

    python -m revert_audit.data.merge --inputs data\\pilot_sdxl data\\pilot_flux --out data\\pilot

--cap N keeps N samples per input (the first N after the seeded shuffle, i.e. a
random subset fixed by --seed) so every editor contributes the same number;
an input with fewer than N samples is an error unless --allow-short.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import read_jsonl, write_jsonl


def merge(inputs: list[str], out: str, seed: int = 0, split: str | None = None, cap: int | None = None,
          allow_short: bool = False) -> str:
    rng = np.random.default_rng(seed)
    lists = []
    for d in inputs:
        p = d if d.endswith(".jsonl") else os.path.join(d, "samples.jsonl")
        rows = [r for r in read_jsonl(p) if split is None or r.get("split", "test") == split]
        rows = [rows[i] for i in rng.permutation(len(rows))]
        if cap is not None:
            if len(rows) < cap and not allow_short:
                raise ValueError(f"{p} has {len(rows)} samples, fewer than --cap {cap}: build more, or --allow-short")
            rows = rows[:cap]
        lists.append(rows)
        print(f"[merge] {p}: {len(rows)} samples" + (f" (cap {cap})" if cap is not None else ""))
    merged, i = [], 0
    while any(i < len(x) for x in lists):
        for x in lists:
            if i < len(x):
                merged.append(x[i])
        i += 1
    ids = [r["id"] for r in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample ids across inputs")
    path = os.path.join(out, "samples.jsonl")
    write_jsonl(merged, path)
    print(f"[merge] wrote {len(merged)} samples to {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None, choices=[None, "test", "train"])
    ap.add_argument("--cap", type=int, default=None, help="samples per input (same for every editor)")
    ap.add_argument("--allow-short", action="store_true")
    a = ap.parse_args(argv)
    merge(a.inputs, a.out, a.seed, a.split, a.cap, a.allow_short)


if __name__ == "__main__":
    main()
