"""Download and unpack COCO 2017 (resumable, pure Python, works on Windows).

    python -m revert_audit.data.download --root /path/to/coco --parts val annotations
    python -m revert_audit.data.download --root /path/to/coco --parts train   # 18 GB, only for training sets

Layout after it finishes (this is what the builder expects):
    <root>/val2017/*.jpg                         5,000 images
    <root>/train2017/*.jpg                     118,287 images (if requested)
    <root>/annotations/instances_val2017.json
    <root>/annotations/instances_train2017.json

Interrupted downloads resume from where they stopped (HTTP Range). If the
official host is slow from your network, pass --base-url pointing to a mirror
that keeps the same file names, or download the zips yourself into
<root>/_zips/ and rerun: existing complete zips are reused.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE = "http://images.cocodataset.org"
PARTS = {
    "val": ("zips/val2017.zip", "val2017", 5000),
    "train": ("zips/train2017.zip", "train2017", 118287),
    "annotations": ("annotations/annotations_trainval2017.zip", "annotations", None),
}


def _remote_size(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def download(url: str, dst: str, chunk: int = 1 << 20, timeout: float = 60.0, retries: int = 5, quiet: bool = False) -> str:
    """Resumable download. Keeps a .part file until the size matches."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    total = _remote_size(url, timeout)
    if os.path.exists(dst) and (total is None or os.path.getsize(dst) == total):
        if not quiet:
            print(f"[download] already complete: {dst}")
        return dst
    part = dst + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and have > total:
            os.remove(part)
            have = 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                mode = "ab" if have else "wb"
                done, t0, last = have, time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if not quiet and time.time() - last > 2:
                            last = time.time()
                            rate = (done - have) / max(1e-6, last - t0) / 2**20
                            pct = f"{100 * done / total:5.1f}%" if total else f"{done / 2**20:.0f} MB"
                            print(f"\r[download] {os.path.basename(dst)} {pct}  {rate:5.1f} MB/s", end="", flush=True)
            if not quiet:
                print()
            if total is None or os.path.getsize(part) == total:
                os.replace(part, dst)
                return dst
            print(f"[download] size mismatch ({os.path.getsize(part)} / {total}), retrying", file=sys.stderr)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"\n[download] attempt {attempt}/{retries} failed: {e}; resuming in 5 s", file=sys.stderr)
            time.sleep(5 if attempt < retries else 0)
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def extract(zip_path: str, root: str, quiet: bool = False) -> None:
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad} in {zip_path}; delete the zip and rerun")
        names = z.namelist()
        todo = [n for n in names if not os.path.exists(os.path.join(root, n))]
        if not quiet:
            print(f"[extract] {os.path.basename(zip_path)}: {len(todo)} of {len(names)} files to write")
        for n in todo:
            z.extract(n, root)


def verify(root: str, part: str) -> tuple[bool, str]:
    _, folder, expected = PARTS[part]
    if part == "annotations":
        need = ["instances_val2017.json", "instances_train2017.json"]
        miss = [n for n in need if not os.path.exists(os.path.join(root, "annotations", n))]
        return (not miss, "ok" if not miss else f"missing {miss}")
    n = len(glob.glob(os.path.join(root, folder, "*.jpg")))
    return (n == expected, f"{n} / {expected} images")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. /path/to/coco")
    ap.add_argument("--parts", nargs="+", default=["val", "annotations"], choices=list(PARTS))
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--keep-zips", action="store_true")
    a = ap.parse_args(argv)
    ok_all = True
    for part in a.parts:
        rel, _, _ = PARTS[part]
        good, msg = verify(a.root, part)
        if good:
            print(f"[coco] {part}: already in place ({msg})")
            continue
        z = download(f"{a.base_url.rstrip('/')}/{rel}", os.path.join(a.root, "_zips", os.path.basename(rel)))
        extract(z, a.root)
        good, msg = verify(a.root, part)
        print(f"[coco] {part}: {'OK' if good else 'INCOMPLETE'} ({msg})")
        ok_all &= good
        if good and not a.keep_zips:
            os.remove(z)
    if not ok_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
