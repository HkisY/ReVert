"""Data download / build / QC / merge, end to end on CPU with a fake editor."""

import csv
import http.server
import io
import json
import os
import threading
import zipfile

import numpy as np
import pytest
from PIL import Image

from revert_audit.data import download as dl
from revert_audit.data.build import build
from revert_audit.data.masks import freeform_mask
from revert_audit.data.merge import merge
from revert_audit.data.qc import apply_review, check_sample, contact_sheets
from revert_audit.data.sources import CocoSource, FolderSource, fit_to_grid
from revert_audit.io import load_mask, load_rgb, read_jsonl


# ------------------------------------------------------------------ download
class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        path = self.translate_path(self.path)
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng and self.server.honour_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            body = data[start:]
        else:
            self.send_response(200)
            body = data
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    (root / "zips").mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(5):
            z.writestr(f"val2017/{i:012d}.jpg", os.urandom(2000))
    (root / "zips" / "val2017.zip").write_bytes(buf.getvalue())
    handler = lambda *a, **k: RangeHandler(*a, directory=str(root), **k)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.honour_range = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd, root
    httpd.shutdown()


@pytest.mark.parametrize("honour_range", [True, False])
def test_download_resumes_and_extracts(server, tmp_path, monkeypatch, honour_range):
    httpd, root = server
    httpd.honour_range = honour_range
    url = f"http://127.0.0.1:{httpd.server_address[1]}/zips/val2017.zip"
    dst = tmp_path / "coco" / "_zips" / "val2017.zip"
    dst.parent.mkdir(parents=True)
    full = (root / "zips" / "val2017.zip").read_bytes()
    (dst.parent / "val2017.zip.part").write_bytes(full[:1234])  # interrupted earlier
    dl.download(url, str(dst), chunk=512, quiet=True)
    assert dst.read_bytes() == full
    monkeypatch.setitem(dl.PARTS, "val", ("zips/val2017.zip", "val2017", 5))
    dl.extract(str(dst), str(tmp_path / "coco"), quiet=True)
    assert dl.verify(str(tmp_path / "coco"), "val") == (True, "5 / 5 images")


# ------------------------------------------------------------- masks/sources
def test_freeform_mask_area():
    rng = np.random.default_rng(0)
    for _ in range(10):
        m = freeform_mask(256, 320, rng, 0.03, 0.25)
        assert m.shape == (256, 320) and 0.02 <= m.mean() <= 0.3


def test_fit_to_grid_crops_without_upscaling():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    m = np.zeros((480, 640), bool)
    m[100:200, 100:200] = True
    out, mo = fit_to_grid(img, m, 1024)
    assert out.size == (640, 448) and mo.shape == (448, 640)
    # pure crop: pixels are the original camera pixels
    assert np.array_equal(np.asarray(out), np.asarray(img)[16:464, :, :])


def _photos(folder, n=6, size=(300, 400)):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(os.path.join(folder, f"p{i}.jpg"))


class FakeInpaint:
    """Paints the masked region grey: a deterministic stand-in for SDXL."""
    name, kind = "fake-inpaint", "inpaint"

    def edit(self, image, mask, prompt, seed):
        a = np.asarray(image).copy()
        a[mask] = 128
        a = np.clip(a.astype(int) + 1, 0, 255).astype(np.uint8)  # global change, like a VAE round trip
        return Image.fromarray(a)


class NoopEditor(FakeInpaint):
    name = "noop"

    def edit(self, image, mask, prompt, seed):
        return image


def test_build_folder_source_and_qc(tmp_path):
    _photos(tmp_path / "ph")
    src = FolderSource(str(tmp_path / "ph"), size=1024, seed=0, name="own")
    stats = build(src, FakeInpaint(), str(tmp_path / "out"), n=4)
    rows = read_jsonl(str(tmp_path / "out" / "samples.jsonl"))
    assert stats["kept"] == 4 and len(rows) == 4
    s = rows[0]
    real, raw, fake, M = load_rgb(s["real"]), load_rgb(s["raw"]), load_rgb(s["fake"]), load_mask(s["mask"])
    assert np.array_equal(fake[~M], real[~M]) and np.array_equal(fake[M], raw[M])
    assert s["split"] == "test" and s["source"] == "own"
    # resumable: a second call keeps the same 4 and does nothing new
    build(FolderSource(str(tmp_path / "ph"), seed=0, name="own"), FakeInpaint(), str(tmp_path / "out"), n=4)
    assert len(read_jsonl(str(tmp_path / "out" / "samples.jsonl"))) == 4


def test_failed_edits_are_logged_not_dropped(tmp_path):
    _photos(tmp_path / "ph", n=3)
    stats = build(FolderSource(str(tmp_path / "ph"), name="own"), NoopEditor(), str(tmp_path / "o"), n=3)
    assert stats["kept"] == 0 and stats["rejected"] == {"edit_no_change": 3}
    assert len(read_jsonl(str(tmp_path / "o" / "rejected.jsonl"))) == 3


def test_check_sample_rules():
    real = np.full((64, 64, 3), 100, np.uint8)
    raw = real.copy()
    M = np.zeros((64, 64), bool)
    M[10:30, 10:30] = True
    raw[M] = 200
    assert check_sample(real, raw, M, "inpaint")[:2] == (True, "ok")
    assert check_sample(real, real, M, "inpaint")[1] == "edit_no_change"
    assert check_sample(real, raw, np.ones((64, 64), bool), "inpaint")[1] == "mask_too_large"
    assert check_sample(real, raw[:32], M, "inpaint")[1] == "size_mismatch"


def test_coco_source_with_tiny_annotations(tmp_path):
    root = tmp_path / "coco"
    (root / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir()
    Image.fromarray(np.full((480, 640, 3), 90, np.uint8)).save(root / "val2017" / "000000000001.jpg")
    ann = {"images": [{"id": 1, "file_name": "000000000001.jpg", "width": 640, "height": 480}],
           "annotations": [{"id": 7, "image_id": 1, "category_id": 18, "iscrowd": 0, "area": 100 * 120,
                            "bbox": [100, 100, 120, 100], "segmentation": [[100, 100, 220, 100, 220, 200, 100, 200]]}],
           "categories": [{"id": 18, "name": "dog"}, {"id": 17, "name": "cat"}]}
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps(ann))
    items = list(CocoSource(str(root), "val2017", seed=0))
    assert len(items) == 1 and items[0].obj == "dog" and items[0].mask.any()
    assert items[0].image.size == (640, 448)
    with pytest.raises(FileNotFoundError):
        CocoSource(str(tmp_path / "nope"))


def test_merge_interleaves_and_review(tmp_path):
    _photos(tmp_path / "ph", n=6)
    build(FolderSource(str(tmp_path / "ph"), name="a"), FakeInpaint(), str(tmp_path / "A"), n=3)
    build(FolderSource(str(tmp_path / "ph"), name="b", seed=5), FakeInpaint(), str(tmp_path / "B"), n=3)
    p = merge([str(tmp_path / "A"), str(tmp_path / "B")], str(tmp_path / "M"))
    rows = read_jsonl(p)
    assert len(rows) == 6 and rows[0]["source"] != rows[1]["source"]
    review = contact_sheets(p, str(tmp_path / "qc"), per_page=4)
    assert os.path.exists(tmp_path / "qc" / "sheet_000.jpg")
    with open(review, encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))
    lines[1][7] = "bad"
    with open(review, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(lines)
    out, rate = apply_review(p, review)
    assert len(read_jsonl(out)) == 5 and abs(rate - 5 / 6) < 1e-9


def test_check_local_model(tmp_path):
    from revert_audit.data.editors import REQUIRED, check_local_model

    root = tmp_path / "FLUX.1-Kontext-dev"
    assert check_local_model(str(root), "kontext")[0].startswith("folder does not exist")
    root.mkdir()
    (root / "model_index.json").write_text("{}")
    for sub in REQUIRED["kontext"]:
        (root / sub).mkdir()
        if sub not in ("scheduler", "tokenizer", "tokenizer_2"):
            (root / sub / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert check_local_model(str(root), "kontext") == []
    (root / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    assert check_local_model(str(root), "kontext") == ["transformer/*.safetensors (weights)"]


def test_review_saved_by_excel_as_gbk(tmp_path):
    _photos(tmp_path / "ph", n=3)
    build(FolderSource(str(tmp_path / "ph"), name="a"), FakeInpaint(), str(tmp_path / "A"), n=3)
    p = str(tmp_path / "A" / "samples.jsonl")
    review = contact_sheets(p, str(tmp_path / "qc"))
    with open(review, encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))
    lines[2][7] = "坏"
    lines[1][8] = "备注：边缘有伪影"
    with open(review, "w", newline="", encoding="gbk") as f:  # what Chinese Excel writes
        csv.writer(f).writerows(lines)
    out, rate = apply_review(p, review)
    assert len(read_jsonl(out)) == 2 and abs(rate - 2 / 3) < 1e-9
    lines[2][2] = "999"  # an id Excel mangled must not be silently ignored
    with open(review, "w", newline="", encoding="gbk") as f:
        csv.writer(f).writerows(lines)
    with pytest.raises(ValueError):
        apply_review(p, review)


@pytest.mark.parametrize("fmt", ["xlsx", "utf16"])
def test_review_saved_as_workbook_or_unicode_text(tmp_path, fmt):
    _photos(tmp_path / "ph", n=3)
    build(FolderSource(str(tmp_path / "ph"), name="a"), FakeInpaint(), str(tmp_path / "A"), n=3)
    p = str(tmp_path / "A" / "samples.jsonl")
    review = contact_sheets(p, str(tmp_path / "qc"))
    with open(review, encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))
    lines[3][7] = "bad"
    if fmt == "xlsx":  # Excel/WPS "save as workbook" while keeping the .csv name
        openpyxl = pytest.importorskip("openpyxl")  # only the test needs it, not the reader

        wb = openpyxl.Workbook()
        for r in lines:
            wb.active.append([int(x) if x.isdigit() else x for x in r])
        wb.save(review)
    else:  # Excel "Unicode text": UTF-16 with tabs
        with open(review, "w", newline="", encoding="utf-16") as f:
            csv.writer(f, delimiter="\t").writerows(lines)
    out, rate = apply_review(p, review)
    assert len(read_jsonl(out)) == 2
