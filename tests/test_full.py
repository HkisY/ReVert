"""E5/E6 additions: the Qwen-Image inpainting editor and per-editor caps in merge."""

import contextlib
import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from revert_audit.data import editors
from revert_audit.data.merge import merge


def test_qwen_image_registered_as_inpaint():
    assert editors.KINDS["qwen-image"] == "inpaint"
    assert editors.REPOS["qwen-image"] == "Qwen/Qwen-Image"
    assert "transformer" in editors.REQUIRED["qwen-image"]


def test_qwen_image_edit_call(monkeypatch):
    calls = {}

    class Emb:
        def __init__(self, v):
            self.v = v

        def cpu(self):
            return self

        def to(self, dev):
            return self

    class Mod:
        def __init__(self, name):
            self.name = name

        def to(self, dev):
            calls.setdefault("moves", []).append((self.name, dev))
            return self

    class Pipe:
        transformer, controlnet, vae = Mod("transformer"), Mod("controlnet"), Mod("vae")
        text_encoder = Mod("text_encoder")

        def encode_prompt(self, prompt, device=None, max_sequence_length=512):
            assert self.text_encoder is not None
            calls.setdefault("encoded", []).append(prompt)
            return Emb(f"emb:{prompt}"), Emb(f"mask:{prompt}")

        def __call__(self, **kw):
            assert self.text_encoder is None  # detached: the pipeline's device is the GPU
            calls.update(kw)
            return types.SimpleNamespace(images=[Image.new("RGB", (kw["width"], kw["height"]))])

    torch_stub = types.SimpleNamespace(Generator=lambda dev: types.SimpleNamespace(manual_seed=lambda s: ("gen", s)),
                                       manual_seed=lambda s: None,
                                       cuda=types.SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0),
                                       device=lambda d: d, no_grad=contextlib.nullcontext)
    ed = editors.DiffusersEditor.__new__(editors.DiffusersEditor)
    ed.torch, ed.base_name, ed.kind, ed.name, ed.pipe, ed.steps = torch_stub, "qwen-image", "inpaint", "qwen-image-cn-nf4", Pipe(), None
    ed._qwen_place()
    img = Image.new("RGB", (128, 64))
    m = np.zeros((64, 128), bool)
    m[10:20, 10:30] = True
    out = ed.edit(img, m, "a red umbrella", 7)
    assert out.size == (128, 64)
    assert calls["controlnet_conditioning_scale"] == 1.0 and calls["true_cfg_scale"] == 4.0
    assert calls["prompt_embeds"].v == "emb:a red umbrella" and calls["negative_prompt_embeds"].v == "emb: "
    assert calls["control_image"].size == editors.qwen_work_size(128, 64)
    assert calls["num_inference_steps"] == 50 and (calls["width"], calls["height"]) == editors.qwen_work_size(128, 64)
    monkeypatch.setenv("QWEN_WORK_MP", "0")
    ed.edit(img, m, "a red umbrella", 7)
    assert (calls["width"], calls["height"]) == (128, 64) and calls["control_image"] is img
    assert np.array_equal(np.asarray(calls["control_mask"]) > 127, m)
    assert calls["encoded"] == ["a red umbrella", " "]  # cached: the second edit encoded nothing new


def test_qwen_work_size(monkeypatch):
    monkeypatch.delenv("QWEN_WORK_MP", raising=False)
    w, h = editors.qwen_work_size(640, 448)
    assert w % 64 == 0 and h % 64 == 0 and 0.9 < w * h / 2**20 < 1.15 and abs(w / h - 640 / 448) < 0.1
    assert editors.qwen_work_size(1280, 1024) == (1280, 1024)


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_merge_cap(tmp_path):
    a = str(tmp_path / "a" / "samples_reviewed.jsonl")
    b = str(tmp_path / "b" / "samples_reviewed.jsonl")
    _write(a, [{"id": f"a{i}", "editor": "sdxl"} for i in range(12)])
    _write(b, [{"id": f"b{i}", "editor": "flux"} for i in range(10)])
    p = merge([a, b], str(tmp_path / "m"), cap=8)
    rows = [json.loads(x) for x in open(p)]
    assert len(rows) == 16
    assert [r["editor"] for r in rows[:4]] == ["sdxl", "flux", "sdxl", "flux"]
    p2 = merge([a, b], str(tmp_path / "m2"), cap=8)
    assert open(p).read() == open(p2).read()  # seeded: the same subset every time
    with pytest.raises(ValueError):
        merge([a, b], str(tmp_path / "m3"), cap=11)
    assert len(open(merge([a, b], str(tmp_path / "m4"), cap=11, allow_short=True)).readlines()) == 21


def test_family_and_decide():
    from revert_audit.hypotheses import decide, family

    assert family("sdxl") == "in" and family("flux-fill") == "out" and family("qwen-image-nf4") == "out"
    assert family("qwen-image-cn-nf4") == "out" and family("qwen-edit-inpaint-nf4") == "out" and family("kandinsky") == "out"
    assert family("flux-fill-nf4") == "out" and family("kontext") is None and family(None) is None
    assert decide([0.2, 0.05, 0.3], +1, [40, 40]) == "supported"
    assert decide([-0.2, -0.3, -0.05], +1, [40, 40]) == "contradicted"
    assert decide([0.1, -0.05, 0.3], +1, [40, 40]) == "inconclusive"
    assert decide([-0.2, -0.3, -0.05], -1, [40, 40]) == "supported"
    assert decide([0.2, 0.05, 0.3], +1, [40, 12]) == "not testable"


def test_cluster_boot_resamples_photos():
    from revert_audit.hypotheses import cluster_boot

    recs = [{"photo": p, "per": float(p % 2)} for p in range(20) for _ in range(3)]
    est, lo, hi = cluster_boot(recs, lambda rs: float(np.mean([r["per"] for r in rs])), 500)
    assert abs(est - 0.5) < 1e-9 and lo < 0.5 < hi


def test_hypotheses_on_a_pilot(tmp_path):
    from revert_audit.detectors import build_detector
    from revert_audit.hypotheses import hypotheses, print_report
    from revert_audit.io import read_jsonl, write_jsonl
    from revert_audit.pilot import run_pilot
    from revert_audit.synthetic import build_synthetic_dataset

    samples = build_synthetic_dataset(str(tmp_path / "data"), 60, seed=4)
    rows = read_jsonl(samples)
    eds = ["sdxl", "flux-fill", "qwen-image-nf4"]
    for i, r in enumerate(rows):
        r["editor"], r["coco_image_id"] = eds[i % 3], i // 3
    write_jsonl(rows, samples)
    run = str(tmp_path / "run")
    run_pilot(samples, run, build_detector("mock-local"), "loc",
              os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), steps=[60], n_boot=50,
              batch_size=16, log_every=0)
    o = hypotheses(run, "loc", samples, n_boot=100)
    print_report(o)
    assert o["n_records"] > 0
    assert set(o["by_editor"]) == set(eds)
    assert all(e["flag_rate"] is not None for e in o["by_editor"].values())
    assert o["H_OOD"]["verdict"] in ("supported", "contradicted", "inconclusive", "not testable")
    assert os.path.exists(os.path.join(run, "hypotheses_loc.json"))


def test_offline_kw(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert editors.offline_kw() == {}
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert editors.offline_kw() == {"local_files_only": True}


def test_qwen_edit_inpaint_call():
    calls = {}

    class Pipe:
        def __call__(self, **kw):
            calls.update(kw)
            return types.SimpleNamespace(images=[Image.new("RGB", (1216, 832))])  # pipeline works at ~1 MP

    torch_stub = types.SimpleNamespace(Generator=lambda dev: types.SimpleNamespace(manual_seed=lambda s: ("gen", s)),
                                       manual_seed=lambda s: None)
    assert editors.KINDS["qwen-edit-inpaint"] == "inpaint" and editors.REPOS["qwen-edit-inpaint"] == "Qwen/Qwen-Image-Edit"
    ed = editors.DiffusersEditor.__new__(editors.DiffusersEditor)
    ed.torch, ed.base_name, ed.kind, ed.name, ed.pipe, ed.steps = (torch_stub, "qwen-edit-inpaint", "inpaint",
                                                                   "qwen-edit-inpaint-nf4", Pipe(), None)
    img = Image.new("RGB", (640, 448))
    m = np.zeros((448, 640), bool)
    m[100:200, 100:300] = True
    out = ed.edit(img, m, "a photo of a cat, realistic, natural lighting", 3)
    assert out.size == (640, 448)
    assert calls["strength"] == 1.0 and calls["true_cfg_scale"] == 4.0 and calls["negative_prompt"] == " "
    ref = np.asarray(calls["image"])
    assert calls["prompt"].startswith("a photo of a cat") and "height" not in calls
    assert (ref[m] == 128).all() and (ref[~m] == 0).all()  # masked area gray, rest = the photo
    assert np.array_equal(np.asarray(calls["mask_image"]) > 127, m)


def test_instruction_prompts_same_rng_stream():
    from revert_audit.data.build import inpaint_prompt

    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(20):
        d, e1 = inpaint_prompt(r1, "tie", None)
        i, e2 = inpaint_prompt(r2, "tie", None, instruct=True)
        assert e1 == e2 and i.startswith("Fill the gray area with a" if e1 == "replace" else "Fill the gray area so")
    assert r1.random() == r2.random()


def test_kandinsky_call():
    calls = {}

    class Pipe:
        def __call__(self, **kw):
            calls.update(kw)
            return types.SimpleNamespace(images=[Image.new("RGB", (kw["width"], kw["height"]))])

    torch_stub = types.SimpleNamespace(Generator=lambda dev: types.SimpleNamespace(manual_seed=lambda s: ("gen", s)))
    assert editors.KINDS["kandinsky"] == "inpaint"
    ed = editors.DiffusersEditor.__new__(editors.DiffusersEditor)
    ed.torch, ed.base_name, ed.kind, ed.name, ed.pipe, ed.steps = torch_stub, "kandinsky", "inpaint", "kandinsky", Pipe(), None
    img = Image.new("RGB", (640, 448))
    m = np.zeros((448, 640), bool)
    m[100:200, 100:300] = True
    out = ed.edit(img, m, "a photo of a cat, realistic, natural lighting", 3)
    assert out.size == (640, 448) and calls["image"] is img and calls["prompt"].startswith("a photo of a cat")
    assert calls["num_inference_steps"] == 50 and calls["guidance_scale"] == 4.0
    assert np.array_equal(np.asarray(calls["mask_image"]) > 127, m)


def test_qc_sheet_keeps_marks_on_extension(tmp_path):
    import csv as _csv

    from revert_audit.data.qc import apply_review, contact_sheets
    from revert_audit.io import read_jsonl, write_jsonl
    from revert_audit.synthetic import build_synthetic_dataset

    sp = build_synthetic_dataset(str(tmp_path / "d"), 14, seed=1)
    rows = read_jsonl(sp)
    part = str(tmp_path / "d" / "first.jsonl")
    write_jsonl(rows[:10], part)
    qc = str(tmp_path / "qc")
    rev = contact_sheets(part, qc)
    table = list(_csv.reader(open(rev, encoding="utf-8-sig")))
    table[3][7], table[5][7], table[5][8] = "bad", "bad", "mush"
    with open(rev, "w", newline="", encoding="utf-8-sig") as f:
        _csv.writer(f).writerows(table)
    contact_sheets(sp, qc)  # extended set (14): marks of the first 10 survive
    new = list(_csv.DictReader(open(rev, encoding="utf-8-sig")))
    assert len(new) == 14 and [r["ok"] for r in new].count("bad") == 2
    assert new[4]["note"] == "mush" and all(r["ok"] == "" for r in new[10:])
    assert os.path.exists(os.path.join(qc, "qc_review.bak.csv"))
    _, rate = apply_review(sp, rev)
    assert len(read_jsonl(os.path.join(os.path.dirname(sp), "samples_reviewed.jsonl"))) == 12


def test_wait_gpu_release_returns(monkeypatch):
    import subprocess as sp
    import time as _t

    from revert_audit.detectors import sida

    calls = []
    outs = iter(["123\n456\n", "456\n"])
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sp, "run", lambda *a, **k: (calls.append(1), types.SimpleNamespace(stdout=next(outs)))[1])
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    sida.wait_gpu_release(123)
    assert len(calls) == 2  # polled until pid 123 was gone
