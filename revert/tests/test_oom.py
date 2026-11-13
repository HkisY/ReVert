"""run_detector must trade time for memory: halve the batch on CUDA OOM, and
skip (and log) an image that does not fit even alone, never inventing a score."""

import os

import numpy as np

from revert_audit import pipeline
from revert_audit.detectors.base import Detector
from revert_audit.io import read_jsonl, save_rgb, write_jsonl


class OutOfMemoryError(RuntimeError):  # same class name torch uses
    pass


class Picky(Detector):
    name = "picky"

    def __init__(self, max_batch, poison_value):
        self.max_batch, self.poison, self.calls = max_batch, poison_value, []

    def predict(self, images, explain=True):
        self.calls.append(len(images))
        if len(images) > self.max_batch or any(int(im[0, 0, 0]) == self.poison for im in images):
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [{"verdict": "fake", "p_fake": 0.9, "boxes": [], "parse_mode": "native"} for _ in images]


def test_batch_halving_and_skip(tmp_path):
    qs = []
    for i in range(10):
        p = str(tmp_path / f"im{i}.png")
        save_rgb(np.full((8, 8, 3), i, np.uint8), p)
        qs.append({"query_id": f"q{i}", "path": p, "kind": "orig"})
    qp = str(tmp_path / "q.jsonl")
    write_jsonl(qs, qp)
    det = Picky(max_batch=2, poison_value=7)
    out = str(tmp_path / "a.jsonl")
    pipeline.run_detector(qp, det, out, batch_size=8, log_every=0)
    got = {a["query_id"] for a in read_jsonl(out)}
    assert got == {f"q{i}" for i in range(10)} - {"q7"}
    assert [r["query_id"] for r in read_jsonl(out + ".failed.jsonl")] == ["q7"]
    assert max(det.calls[3:]) <= 2  # batch size stays reduced after the first OOM


def test_non_oom_errors_propagate(tmp_path):
    p = str(tmp_path / "x.png")
    save_rgb(np.zeros((4, 4, 3), np.uint8), p)
    qp = str(tmp_path / "q.jsonl")
    write_jsonl([{"query_id": "q", "path": p}], qp)

    class Broken(Detector):
        def predict(self, images, explain=True):
            raise ValueError("bad prompt")

    try:
        pipeline.run_detector(qp, Broken(), str(tmp_path / "a.jsonl"), log_every=0)
    except ValueError:
        return
    raise AssertionError("non-OOM error was swallowed")


def test_resume_skips_done(tmp_path):
    qs = []
    for i in range(3):
        p = str(tmp_path / f"im{i}.png")
        save_rgb(np.full((4, 4, 3), i, np.uint8), p)
        qs.append({"query_id": f"q{i}", "path": p})
    qp = str(tmp_path / "q.jsonl")
    write_jsonl(qs, qp)
    out = str(tmp_path / "a.jsonl")
    write_jsonl([{"query_id": "q0", "p_fake": 0.1}], out)
    det = Picky(max_batch=8, poison_value=255)
    pipeline.run_detector(qp, det, out, batch_size=8, log_every=0)
    assert sum(det.calls) == 2 and len(read_jsonl(out)) == 3
    assert not os.path.exists(out + ".failed.jsonl")


def test_cap_vram_validation_and_noop_without_gpu():
    import pytest as _pt

    from revert_audit.gpu import cap_vram

    assert cap_vram(None) is False
    with _pt.raises(ValueError):
        cap_vram(1.5)
    assert cap_vram(0.9) in (False, True)  # False on a machine without CUDA


def test_host_memory_check(monkeypatch, capsys):
    from revert_audit import gpu

    assert gpu.check_host_memory("flux-fill", offloaded=False) is True  # nothing kept on CPU
    monkeypatch.setattr(gpu, "host_memory_gib", lambda: {"ram_total": 32, "ram_free": 20, "commit_limit": 40, "commit_free": 30})
    assert gpu.check_host_memory("flux-fill", offloaded=True) is False  # 34 + 8 > 30
    assert "page file" in capsys.readouterr().out
    assert gpu.check_host_memory("sdxl", offloaded=True) is True
    monkeypatch.setattr(gpu, "host_memory_gib", lambda: {"ram_total": 32, "ram_free": 28, "commit_limit": 96, "commit_free": 80})
    assert gpu.check_host_memory("flux-fill", offloaded=True) is True
    assert "page file will be used" in capsys.readouterr().out


def test_transient_oom_at_batch_one_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    qs = []
    for i in range(4):
        p = str(tmp_path / f"im{i}.png")
        save_rgb(np.full((8, 8, 3), i, np.uint8), p)
        qs.append({"query_id": f"q{i}", "path": p, "kind": "orig"})
    qp = str(tmp_path / "q.jsonl")
    write_jsonl(qs, qp)

    class Flaky(Detector):
        name = "flaky"

        def __init__(self):
            self.fails = {2: 2}  # image 2 runs out of memory twice, then fits

        def predict(self, images, explain=True):
            v = int(images[0][0, 0, 0])
            if len(images) > 1 or self.fails.get(v, 0) > 0:
                if len(images) == 1:
                    self.fails[v] -= 1
                raise OutOfMemoryError("CUDA out of memory")
            return [{"verdict": "fake", "p_fake": 0.9, "boxes": [], "parse_mode": "native"}]

    out = str(tmp_path / "a.jsonl")
    pipeline.run_detector(qp, Flaky(), out, batch_size=4, log_every=0)
    assert {a["query_id"] for a in read_jsonl(out)} == {"q0", "q1", "q2", "q3"}
    assert not os.path.exists(out + ".failed.jsonl")
