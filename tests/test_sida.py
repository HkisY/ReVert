"""SIDA adapter: the read-out logic, the GPU/CPU split and the worker protocol
(with a stand-in worker; the real model is checked on the GPU machine with
`sida_worker.py --selftest`)."""

import os
import sys

import numpy as np
import pytest

from revert_audit.detectors.base import answer_verdict
from revert_audit.detectors.sida import SidaDetector
from revert_audit.detectors.sida_worker import cls_row_position, plan_offload, softmax

FAKE = os.path.join(os.path.dirname(__file__), "fixtures", "fake_sida_worker.py")
CLS = 32001


def test_cls_row_matches_sida_evaluate_indexing():
    # SIDA.evaluate: mask = [0]*255 + [ids[1:] == CLS]; the LAST selected row decides
    ids = [1, 5, CLS, 7, CLS, 9]
    mask = [False] * 255 + [t == CLS for t in ids[1:]]
    assert cls_row_position(ids, CLS) == max(i for i, v in enumerate(mask) if v) == 258
    assert cls_row_position([1, 5, 7], CLS) is None
    assert cls_row_position([CLS, 5, 7], CLS) is None  # ids[0] is never looked at (ids[1:])


def test_plan_offload():
    sizes = {"model.embed_tokens": 2, "lm_head": 2, "model.visual_model": 6, **{f"model.layers.{i}": 4 for i in range(4)}}
    layers = [f"model.layers.{i}" for i in range(4)]
    assert set(plan_offload(sizes, layers, None).values()) == {0}
    assert set(plan_offload(sizes, layers, 100).values()) == {0}  # fits: nothing offloaded
    dm = plan_offload(sizes, layers, 20)  # total 26 -> two last layers to CPU
    assert [k for k, v in dm.items() if v == "cpu"] == ["model.layers.2", "model.layers.3"]
    assert dm["model.visual_model"] == 0 and dm["lm_head"] == 0
    with pytest.raises(ValueError):
        plan_offload(sizes, layers, 5)


def test_softmax():
    p = softmax([2.0, 0.0, -1.0])
    assert abs(sum(p) - 1) < 1e-9 and p[0] > p[1] > p[2]


def test_native_verdict_wins_over_threshold():
    # SIDA: P(real)=0.45 is still its argmax -> 'real', although 1 - P(real) = 0.55
    assert answer_verdict({"verdict": "real", "verdict_native": True, "p_fake": 0.55}) == "real"
    assert answer_verdict({"verdict": "real", "p_fake": 0.55}) == "fake"  # other detectors unchanged


@pytest.fixture
def sida(tmp_path):
    d = SidaDetector(python=sys.executable, repo="x", weights="y", worker=FAKE, mask_dir=str(tmp_path / "masks"),
                     tmp_dir=str(tmp_path / "tmp"))
    os.makedirs(d.tmp, exist_ok=True)
    yield d
    d.close()


def test_worker_protocol(sida):
    bright = np.full((64, 64, 3), 200, np.uint8)
    bright[:10] = 20
    dark = np.full((64, 64, 3), 30, np.uint8)
    a = sida.predict([bright, dark], explain=True)
    assert [x["verdict"] for x in a] == ["fake", "real"]
    assert a[0]["mask_path"] and os.path.exists(a[0]["mask_path"]) and a[1]["mask_path"] is None
    assert os.path.basename(a[0]["mask_path"]).startswith(SidaDetector.key(bright))
    assert sida.predict([bright], explain=False)[0]["mask_path"] is None  # controls: no masks written
    assert os.listdir(sida.tmp) == []  # temporary query images are removed


def test_worker_error_is_raised_and_worker_survives(sida):
    bad = np.full((32, 32, 3), 50, np.uint8)
    bad[0, 0] = (1, 2, 3)
    with pytest.raises(RuntimeError, match="out of memory"):  # -> run_detector's OOM path
        sida.predict([bad])
    assert sida.predict([np.full((32, 32, 3), 30, np.uint8)])[0]["verdict"] == "real"


def test_missing_python_and_dead_worker(tmp_path):
    with pytest.raises(FileNotFoundError):
        SidaDetector(python=str(tmp_path / "nope.exe"), repo="x", weights="y")
    dead = tmp_path / "dead.py"
    dead.write_text("import sys; sys.exit(3)\n")
    with pytest.raises(RuntimeError, match="exited"):
        SidaDetector(python=sys.executable, repo="x", weights="y", worker=str(dead))


def test_pilot_runs_end_to_end_through_the_worker(tmp_path, monkeypatch):
    """Full pilot with SIDA-style answers (masks instead of boxes, native verdicts):
    the stand-in behaves like the faithful synthetic detector, so the audit must
    find its masks and reach the same decision as mock-local (PIVOT at n=50)."""
    from revert_audit.pilot import run_pilot
    from revert_audit.synthetic import build_synthetic_dataset

    monkeypatch.setenv("FAKE_SIDA_MODE", "mocklocal")
    samples = build_synthetic_dataset(str(tmp_path / "data"), 160, seed=7)
    d = SidaDetector(python=sys.executable, repo="x", weights="y", worker=FAKE, mask_dir=str(tmp_path / "masks"))
    try:
        r = run_pilot(samples, str(tmp_path / "run"), d, "sida",
                      os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"),
                      n_boot=100, batch_size=8, log_every=0)
    finally:
        d.close()
    assert r["n"] >= 30 and os.listdir(tmp_path / "masks")
    assert r["decision"] == "PIVOT"


def test_stand_ins_do_not_leak(tmp_path):
    """The real failure on Windows: a deepspeed stand-in left in sys.modules
    makes importlib.util.find_spec('deepspeed') raise inside transformers."""
    import importlib.util

    from revert_audit.detectors.sida_worker import import_sida_model

    repo = tmp_path / "SIDA"
    (repo / "model").mkdir(parents=True)
    (repo / "model" / "__init__.py").write_text("")
    (repo / "model" / "SIDA.py").write_text(
        "import deepspeed\nfrom torchviz import make_dot\nclass SIDAForCausalLM:\n    pass\n")
    for n in ("model", "model.SIDA"):
        sys.modules.pop(n, None)
    cls = import_sida_model(str(repo))
    assert cls.__name__ == "SIDAForCausalLM"
    assert "deepspeed" not in sys.modules and "torchviz" not in sys.modules
    assert importlib.util.find_spec("deepspeed") is None  # raised before the fix
    for n in ("model", "model.SIDA"):
        sys.modules.pop(n, None)
    sys.path.remove(str(repo))


def test_safe_decode():
    from revert_audit.detectors.sida_worker import safe_decode

    class Tok:
        def __len__(self):
            return 10

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(97 + i) for i in ids)

    assert safe_decode(Tok(), [0, 1, -200, 2, 32001, 3]) == "abc<id:32001>d"


def test_diagnose_runs_on_a_pilot(tmp_path, monkeypatch):
    from revert_audit.detectors import build_detector
    from revert_audit.diagnose import diagnose, print_report
    from revert_audit.pilot import run_pilot
    from revert_audit.synthetic import build_synthetic_dataset

    samples = build_synthetic_dataset(str(tmp_path / "data"), 60, seed=3)
    run = str(tmp_path / "run")
    run_pilot(samples, run, build_detector("mock-local"), "loc",
              os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), steps=[60], n_boot=50,
              batch_size=16, log_every=0)
    out = diagnose(run, "loc", samples)
    d = out["by_base"]["fake"]
    assert d["controls_model"]["n"] > 0 and d["controls_model"]["frac_clean"] is not None
    # the faithful synthetic detector: clean-control R near 1, reverting never raises the score
    assert d["R_clean"] is not None and d["R_clean"] > 0.8
    assert d["revert_raises_score"][0] < 0.1
    print_report(out)
    assert os.path.exists(os.path.join(run, "diagnose_loc.json"))
    assert "STAY_minus_PER" in d["MNG"]
    assert d["by_editor"] and all(e["n"] > 0 for e in d["by_editor"].values())
    late = diagnose(run, "loc", samples, skip=30)
    assert late["by_base"]["fake"]["controls_model"]["n"] < d["controls_model"]["n"]
    assert os.path.exists(os.path.join(run, "diagnose_loc_skip30.json"))


def test_matched_control_queries(tmp_path):
    """stage2 adds revCin only for model citations, inside M, outside C, same area as C∩M."""
    from revert_audit import pipeline
    from revert_audit.io import load_mask, read_jsonl, write_jsonl
    from revert_audit.synthetic import build_synthetic_dataset

    samples = build_synthetic_dataset(str(tmp_path / "data"), 6, seed=5)
    rows = read_jsonl(samples)
    s1 = str(tmp_path / "s1")
    q1 = pipeline.stage1(samples, s1)
    answers = []
    for q in read_jsonl(q1):
        s = next(r for r in rows if r["id"] == q["sample_id"])
        M = load_mask(s["mask"])
        ys, xs = np.nonzero(M)
        # cite only the left part of the edit
        box = [int(xs.min()), int(ys.min()), int((xs.min() + xs.max()) / 2), int(ys.max())]
        answers.append({"query_id": q["query_id"], "verdict": "fake", "p_fake": 0.9, "boxes": [box]})
    a1 = str(tmp_path / "a1.jsonl")
    write_jsonl(answers, a1)
    out = str(tmp_path / "s2")
    pipeline.stage2(samples, s1, a1, out, explainer="model")
    meta = [m for m in read_jsonl(str(tmp_path / "s2" / "meta.jsonl")) if not m.get("skipped")]
    qs = read_jsonl(str(tmp_path / "s2" / "queries.jsonl"))
    feas = [m for m in meta if m["cin_feasible"]]
    assert feas, "half-cited edits must allow a matched in-edit control"
    n_ctrl = len(feas[0]["ctrl_overlap"])
    assert sum(q["kind"] == "revCin" for q in qs) == n_ctrl * len(feas)
    from revert_audit.regions import area
    for m in feas:
        C, M = load_mask(m["cmask"]), load_mask(next(r for r in rows if r["id"] == m["sample_id"])["mask"])
        assert m["cin_area"] == int((C & M).sum())
    pipeline.stage2(samples, s1, a1, str(tmp_path / "s2o"), explainer="oracle")
    assert not any(q["kind"] == "revCin" for q in read_jsonl(str(tmp_path / "s2o" / "queries.jsonl")))


def test_seam_experiment_runs(tmp_path):
    from revert_audit.detectors import build_detector
    from revert_audit.pilot import run_pilot
    from revert_audit.seam import print_report, run_seam
    from revert_audit.synthetic import build_synthetic_dataset

    samples = build_synthetic_dataset(str(tmp_path / "data"), 40, seed=4)
    run = str(tmp_path / "run")
    det = build_detector("mock-local")
    run_pilot(samples, run, det, "loc", os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"),
              steps=[40], n_boot=50, batch_size=16, log_every=0)
    out = run_seam(samples, run, det, "loc", ramps=(8,), n_boot=50, batch_size=16, log_every=0)
    assert set(out["conditions"]) == {"hard", "ramp8", "dilate4"}
    assert out["conditions"]["ramp8"]["n"] == out["conditions"]["hard"]["n"] > 0
    print_report(out)


def test_seam_attraction_rebuilds_the_reverted_blobs(tmp_path):
    from revert_audit.attraction import attraction, print_report
    from revert_audit.detectors import build_detector
    from revert_audit.pilot import run_pilot
    from revert_audit.synthetic import build_synthetic_dataset

    samples = build_synthetic_dataset(str(tmp_path / "data"), 60, seed=11)
    run = str(tmp_path / "run")
    det = build_detector("mock-local")
    orig = det.predict

    def half(images, explain=True):  # cite only the left half of what mock-local finds -> uncited forged pixels exist
        out = orig(images, explain)
        for a in out:
            if a.get("boxes"):
                a["boxes"] = [[x1, y1, x1 + (x2 - x1) / 2, y2] for x1, y1, x2, y2 in a["boxes"]]
        return out

    det.predict = half
    run_pilot(samples, run, det, "loc",
              os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), steps=[60], n_boot=50,
              batch_size=16, log_every=0)
    out = attraction(run, "loc", samples, n_boot=50)
    print_report(out)
    assert out["image_check"]["checked"] > 0 and out["image_check"]["mismatch"] == 0
    assert out["n"] > 0
    # the faithful stand-in cites forged pixels: a reverted (now real) blob is not cited more than remaining forgery
    assert out["A"][0] <= 0.05


def test_gpu_preflight_waits_then_names_holders(monkeypatch):
    import shutil
    import pytest
    from revert_audit.detectors import sida

    assert sida.gpu_need_from_cmd(["w", "--gpu-mem", "18.0"]) == 18.0 + sida.GPU_MARGIN_GIB
    assert sida.gpu_need_from_cmd(["w", "--gpu-mem", "0"]) == 0.0 and sida.gpu_need_from_cmd(["w"]) == 0.0
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    free = {"v": "10240"}

    def smi(args):
        if args[0].startswith("--query-gpu"):
            return f"0, {free['v']}\n"
        return "1482579, 9380\n"

    monkeypatch.setattr(sida, "_smi", smi)
    real_run = sida.subprocess.run
    monkeypatch.setattr(sida.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "python -m revert_audit.data.build"})())
    with pytest.raises(RuntimeError) as e:
        sida.wait_gpu_free(19.5, "fakeshield-dte", wait_min=0, poll_s=0)
    assert "1482579" in str(e.value) and "data.build" in str(e.value) and "kept" in str(e.value)
    free["v"] = "23000"
    sida.wait_gpu_free(19.5, "fakeshield-dte", wait_min=0, poll_s=0)  # enough memory: returns
    monkeypatch.setattr(sida.subprocess, "run", real_run)


def test_gpu_need_with_whole_model_on_gpu():
    from revert_audit.detectors import sida
    assert sida.gpu_need_from_cmd(["w", "--gpu-mem", "0"], full_gib=14.4) == 14.4 + sida.GPU_MARGIN_GIB
    assert sida.gpu_need_from_cmd(["w", "--gpu-mem", "18"], full_gib=26.0) == 18 + sida.GPU_MARGIN_GIB


def test_worker_that_dies_while_loading_is_retried_once_then_reported(tmp_path, monkeypatch):
    import sys
    import pytest
    from revert_audit.detectors import sida

    flag = tmp_path / "second"
    w = tmp_path / "w.py"
    w.write_text(
        "import os, sys, json\n"
        f"f = {str(flag)!r}\n"
        "if not os.path.exists(f):\n"
        "    open(f, 'w').close(); sys.exit(1)\n"
        "print('@@REVERT@@' + json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    break\n")
    d = sida.WorkerDetector()
    d._start([sys.executable, str(w)], str(tmp_path), str(tmp_path), "t")  # 1st attempt dies, 2nd loads
    assert d.info["ready"]
    d.close()
    bad = tmp_path / "bad.py"
    bad.write_text("import sys; sys.exit(1)\n")
    d2 = sida.WorkerDetector()
    with pytest.raises(RuntimeError, match="exited while loading"):
        d2._start([sys.executable, str(bad)], str(tmp_path), str(tmp_path), "t")


def test_shrink_budget_only_for_whole_model_workers():
    from revert_audit.detectors import sida
    assert sida.shrink_budget(["w", "--gpu-mem", "0"], 13.9) == 11.4      # MFLM beside a 9 GiB job
    assert sida.shrink_budget(["w", "--gpu-mem", "18"], 13.9) is None     # DTE-FDM waits instead
    assert sida.shrink_budget(["w", "--gpu-mem", "0"], 7.0) is None       # too little to be worth it
    assert sida.shrink_budget(["w", "--gpu-mem", "0"], None) is None


def test_holder_hidden_from_ps_is_named_by_the_driver(monkeypatch):
    from revert_audit.detectors import sida
    monkeypatch.setattr(sida, "_smi", lambda args: "3978307, /usr/bin/python3, 9380\n")
    monkeypatch.setattr(sida.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    (pid, gib, desc), = sida.gpu_holders()
    assert pid == 3978307 and abs(gib - 9.16) < 0.01 and "python3" in desc and "not visible" in desc


def test_gpu_need_is_clamped_to_the_card():
    from revert_audit.detectors import sida
    assert sida.clamp_need(23.5, 23.99) == 23.99 - sida.GPU_RESERVE_GIB   # a cap above the card minus the desktop
    assert sida.clamp_need(16.0, 23.99) == 16.0
    assert sida.clamp_need(16.0, None) == 16.0


def test_sida_need_uses_model_size_not_the_cap():
    from revert_audit.detectors import sida
    # SIDA runs with --gpu-mem 22 but its whole model is ~18 GiB: it must start with ~22 GiB free
    need = sida.gpu_need_from_cmd(["w", "--gpu-mem", "22"], full_gib=sida.SidaDetector.full_gpu_gib)
    assert need == 18.0 + sida.GPU_MARGIN_GIB and need <= 22.2
    assert sida.gpu_need_from_cmd(["w", "--gpu-mem", "11"], full_gib=14.4) == 11 + sida.GPU_MARGIN_GIB
