"""Quality control for built samples.

Automatic checks (run by the builder on every sample; failures are written to
rejected.jsonl with the reason, never silently dropped):
    size_mismatch     editor output has a different size than x_r
    mask_too_small / mask_too_large
    edit_no_change    the masked region is (almost) unchanged: the edit failed
    mask_fragmented   instruction edit changed many scattered places (global
                      colour shift, misalignment) instead of one local edit

Human check (do it once per editor, ~10 minutes):
    python -m revert_audit.data.qc sheet  --samples data/pilot/samples.jsonl --out data/pilot/qc
    -> contact-sheet pages (real | fake | mask | difference) + qc_review.csv;
       write 'bad' in the 'ok' column for broken samples, then
    python -m revert_audit.data.qc apply --samples data/pilot/samples.jsonl --review data/pilot/qc/qc_review.csv
    -> samples_reviewed.jsonl (bad ones removed) + the human pass rate
"""

from __future__ import annotations

import argparse
import csv
import io
import os

import numpy as np

from ..io import load_mask, load_rgb, read_jsonl, write_jsonl


def check_sample(real: np.ndarray, raw: np.ndarray, mask: np.ndarray, kind: str, min_frac=0.01, max_frac=0.4,
                 min_change=4.0, max_components=12) -> tuple[bool, str, dict]:
    if real.shape != raw.shape:
        return False, "size_mismatch", {}
    frac = float(mask.mean())
    stats = {"mask_frac": round(frac, 4)}
    if frac < min_frac:
        return False, "mask_too_small", stats
    if frac > max_frac:
        return False, "mask_too_large", stats
    diff = np.abs(real.astype(np.int16) - raw.astype(np.int16)).mean(axis=2)
    stats["change_in_mask"] = round(float(diff[mask].mean()), 2)
    stats["change_outside_mask"] = round(float(diff[~mask].mean()) if (~mask).any() else 0.0, 2)
    if stats["change_in_mask"] < min_change:
        return False, "edit_no_change", stats
    if kind == "instruct":
        import cv2

        n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        big = int((st[1:, cv2.CC_STAT_AREA] >= 0.002 * mask.size).sum()) if n > 1 else 0
        stats["components"] = big
        if big > max_components:
            return False, "mask_fragmented", stats
    return True, "ok", stats


def _thumb(arr: np.ndarray, h: int = 220) -> np.ndarray:
    import cv2

    s = h / arr.shape[0]
    return cv2.resize(arr, (max(1, int(arr.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def contact_sheets(samples_path: str, out_dir: str, per_page: int = 10, limit: int | None = None) -> str:
    import cv2

    rows = read_jsonl(samples_path)[: limit or None]
    os.makedirs(out_dir, exist_ok=True)
    review = os.path.join(out_dir, "qc_review.csv")
    marks = {}
    if os.path.exists(review):  # a rebuilt/extended set: keep the marks already made, back up the old file
        import shutil

        prev = _read_review(review)
        marks = {(r.get("id") or "").strip(): ((r.get("ok") or "").strip(), (r.get("note") or "").strip()) for r in prev}
        shutil.copyfile(review, os.path.join(out_dir, "qc_review.bak" + (".xlsx" if open(review, "rb").read(2) == b"PK" else ".csv")))
    carried = 0
    with open(review, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["page", "row", "id", "editor", "edit_type", "prompt_or_instruction", "mask_frac", "ok", "note"])
        for page in range(0, len(rows), per_page):
            strips = []
            for r_i, s in enumerate(rows[page : page + per_page]):
                real, fake, M = load_rgb(s["real"]), load_rgb(s["fake"]), load_mask(s["mask"])
                over = fake.copy()
                cnts, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(over, cnts, -1, (0, 255, 0), max(2, real.shape[0] // 200))
                diff = np.abs(real.astype(np.int16) - fake.astype(np.int16)).max(axis=2)
                heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
                tiles = [_thumb(x) for x in (real, fake, over, heat)]
                strip = np.concatenate(tiles, axis=1)
                label = f"#{r_i}  {s['id']}  {s.get('editor', '')}/{s.get('edit_type', '')}"
                strip = cv2.copyMakeBorder(strip, 26, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(strip, label[:110], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                strips.append(strip)
                w.writerow([page // per_page, r_i, s["id"], s.get("editor", ""), s.get("edit_type", ""),
                            s.get("prompt") or s.get("instruction", "") or " | ".join(p.get("prompt", "") for p in s.get("parts") or []),
                            s.get("mask_frac", ""), *marks.get(s["id"], ("", ""))])
                carried += s["id"] in marks
            wmax = max(x.shape[1] for x in strips)
            strips = [cv2.copyMakeBorder(x, 0, 0, 0, wmax - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in strips]
            sheet = np.concatenate(strips, axis=0)
            cv2.imwrite(os.path.join(out_dir, f"sheet_{page // per_page:03d}.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("[qc] columns: real | fake | fake + mask outline (green) | |fake - real| heat map")
    print(f"[qc] wrote {(len(rows) + per_page - 1) // per_page} sheets and {review}")
    if marks:
        new_rows = [i for i, s in enumerate(rows) if s["id"] not in marks]
        print(f"[qc] kept the existing marks of {carried} samples (old file backed up as qc_review.bak.*); "
              f"{len(new_rows)} new samples to review" + (f": sheets {new_rows[0] // per_page:03d}-{new_rows[-1] // per_page:03d}" if new_rows else ""))
    return review


def _xlsx_rows(raw: bytes) -> list[list[str]]:
    """First worksheet of an .xlsx file as rows of strings (stdlib only)."""
    import re
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(io.BytesIO(raw))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    sheets = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    if not sheets:
        raise ValueError("no worksheet found in the .xlsx file")
    sheet = "xl/worksheets/sheet1.xml" if "xl/worksheets/sheet1.xml" in sheets else sheets[0]
    out = []
    for row in ET.fromstring(z.read(sheet)).iter(f"{{{ns['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", ns):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r", "A")).group(0):
                col = col * 26 + ord(ch) - 64
            t, v = c.get("t"), c.find("m:v", ns)
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{ns['m']}}}t"))
            else:
                val = v.text if v is not None and v.text is not None else ""
            cells[col - 1] = val
        if cells:
            out.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return out


def _read_review(review_csv: str) -> list[dict]:
    """Accept what spreadsheet programs produce when the review file is saved:
    UTF-8 / GBK / UTF-16 text with ',' ';' or tab separators, or an .xlsx
    workbook that kept the .csv name (Excel/WPS 'save as workbook')."""
    raw = open(review_csv, "rb").read()
    if raw[:2] == b"PK":  # a zip container = .xlsx
        table = _xlsx_rows(raw)
        if not table:
            raise ValueError(f"{review_csv}: the workbook is empty")
        header = [h.strip() for h in table[0]]
        rows = [dict(zip(header, r + [""] * (len(header) - len(r)))) for r in table[1:]]
        how = "xlsx workbook"
    else:
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            encs = ("utf-16",)
        else:
            encs = ("utf-8-sig", "gbk")
        for enc in encs:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"{review_csv}: cannot read it (first bytes {raw[:8]!r}); "
                             "in Excel use File > Save As > 'CSV UTF-8 (comma delimited)'")
        first = text.splitlines()[0] if text else ""
        delim = max([",", ";", "\t"], key=first.count)
        rows = list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))
        how = f"{enc} text, separator {delim!r}"
    if rows and ("id" not in rows[0] or "ok" not in rows[0]):
        raise ValueError(f"{review_csv}: expected columns 'id' and 'ok', found {list(rows[0])}")
    print(f"[qc] read {review_csv} as {how}")
    return rows


def apply_review(samples_path: str, review_csv: str, out_path: str | None = None) -> tuple[str, float]:
    bad = set()
    review = _read_review(review_csv)
    seen = len(review)
    for r in review:
        if (r.get("ok") or "").strip().lower() in ("bad", "no", "n", "0", "x", "false", "不行", "差", "坏"):
            bad.add((r.get("id") or "").strip())
    rows = read_jsonl(samples_path)
    unknown = bad - {r["id"] for r in rows}
    if unknown:  # e.g. Excel changed the ids: do not silently keep samples marked bad
        raise ValueError(f"{len(unknown)} ids marked bad are not in {samples_path}, e.g. {sorted(unknown)[:3]}")
    if seen != len(rows):
        print(f"[qc] note: the review file has {seen} rows, the samples file {len(rows)}")
    kept = [r for r in rows if r["id"] not in bad]
    out = out_path or os.path.join(os.path.dirname(samples_path), "samples_reviewed.jsonl")
    write_jsonl(kept, out)
    rate = 1 - len(bad) / max(1, seen)
    print(f"[qc] reviewed {seen}, marked bad {len(bad)} -> human pass rate {rate:.1%}; wrote {len(kept)} samples to {out}")
    return out, rate


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-page", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("apply")
    p.add_argument("--samples", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        contact_sheets(a.samples, a.out, a.per_page, a.limit)
    else:
        apply_review(a.samples, a.review, a.out)


if __name__ == "__main__":
    main()
