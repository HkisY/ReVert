import os
import sys

import numpy as np
import pytest

from revert_audit.detectors import sofake_worker as W

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_smart_resize_matches_qwen_vl_utils():
    assert W.smart_resize(1024, 1024) == (1036, 1036)
    assert W.smart_resize(1024, 1024, max_pixels=W.DM_MAX_PIXELS) == (1008, 1008)
    assert W.smart_resize(480, 640) == (476, 644)
    h, w = W.smart_resize(3000, 4000, max_pixels=W.DM_MAX_PIXELS)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= W.DM_MAX_PIXELS


def test_prompt_and_stop_tokens():
    p = W.chat_prompt("<image>\nWhat is this?")
    assert p.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
    assert "<|vision_start|><|image_pad|><|vision_end|>\nWhat is this?<|im_end|>\n<|im_start|>assistant\n" in p
    with pytest.raises(ValueError):
        W.chat_prompt("no image")
    assert W.strip_stop("<answer>REAL</answer><|im_end|>") == "<answer>REAL</answer>"
    assert W.strip_stop("a <|box_start|>(1,2),(3,4)<|box_end|><|endoftext|>x") == "a <|box_start|>(1,2),(3,4)<|box_end|>"


def test_box_frame_224_to_image():
    assert W.box_to_image([40, 50, 112, 168], 672, 448) == [120.0, 100.0, 336.0, 336.0]
    assert W.box_to_image([0, 0, 224, 224], 640, 480) == [0.0, 0.0, 640.0, 480.0]
    assert W.box_to_image(None, 10, 10) is None
    assert W.box_to_image([5, 5, 5, 9], 224, 224) is None


def test_cache_roundtrip(tmp_path):
    c = W.Cache(str(tmp_path / "c.jsonl"))
    c.put("k", {"verdict": "fake"})
    assert W.Cache(str(tmp_path / "c.jsonl")).get("k") == {"verdict": "fake"}
    (tmp_path / "c.jsonl").write_text('{"v": "old", "key": "k", "answer": {}}\n')
    assert W.Cache(str(tmp_path / "c.jsonl")).get("k") is None  # other versions are ignored


def test_runtime_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.Runtime(str(tmp_path))


def _det(tmp_path):
    from revert_audit.detectors.sofake import SoFakeDetector

    for d in ("repo", "w", "q"):
        os.makedirs(tmp_path / d, exist_ok=True)
    return SoFakeDetector(repo=str(tmp_path / "repo"), weights=str(tmp_path / "w"), qwen_base=str(tmp_path / "q"),
                          python=sys.executable, worker=os.path.join(FIX, "fake_sofake_worker.py"),
                          cache=str(tmp_path / "cache.jsonl"))


def test_detector_protocol_and_hard_scores(tmp_path):
    from revert_audit.detectors import answer_score, answer_verdict

    det = _det(tmp_path)
    try:
        img = np.zeros((64, 64, 3), np.uint8)
        a = det.predict([img, img])
        assert len(a) == 2 and a[0]["p_fake"] is None
        assert answer_score(a[0]) in (0.0, 1.0) and answer_verdict(a[0]) == a[0]["verdict"]
    finally:
        det.close()


def test_missing_paths(tmp_path):
    from revert_audit.detectors.sofake import SoFakeDetector

    with pytest.raises(FileNotFoundError):
        SoFakeDetector(repo=str(tmp_path / "nope"), weights=str(tmp_path), qwen_base=str(tmp_path))


def test_pilot_through_the_sofake_worker(tmp_path):
    """A detector with hard decisions and boxes goes through the whole pilot."""
    from revert_audit.pilot import run_pilot
    from revert_audit.synthetic import build_synthetic_dataset

    samples = build_synthetic_dataset(str(tmp_path / "data"), 40, seed=5)
    det = _det(tmp_path)
    try:
        res = run_pilot(samples, str(tmp_path / "run"), det, "sf",
                        os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), steps=[40],
                        n_boot=50, batch_size=8, log_every=0, s2_bases=("fake",))
    finally:
        det.close()
    assert res["decision"] in ("GO", "PIVOT", "MORE_DATA", "UNDECIDED", "NOT_INFORMATIVE", "SWITCH_DETECTOR")
    import json

    m = json.load(open(tmp_path / "run" / "results" / "sf_model" / "metrics.json", encoding="utf-8"))
    assert set(m["by_base"]) == {"fake"}


def test_fakeshield_worker_helpers():
    from revert_audit.detectors import fakeshield_worker as F

    assert F.verdict_from_text(F.PREFIX + " not been tampered with.") == "real"
    assert F.verdict_from_text(F.PREFIX + " been tampered with, specifically ...") == "fake"
    assert F.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and F.auc([0.5], [0.5]) == 0.5
    with pytest.raises(FileNotFoundError):
        F.find_dte("/nonexistent")


def test_fakeshield_two_stage_pilot(tmp_path):
    """DTE stand-in in predict(), MFLM stand-in in finalize(): masks reach the metrics."""
    import json

    from revert_audit.detectors.fakeshield import FakeShieldDetector
    from revert_audit.pilot import run_pilot
    from revert_audit.synthetic import build_synthetic_dataset

    for d in ("repo", "w"):
        os.makedirs(tmp_path / d, exist_ok=True)
    det = FakeShieldDetector(dte_python=sys.executable, mflm_python=sys.executable, repo=str(tmp_path / "repo"),
                             weights=str(tmp_path / "w"), mask_dir=str(tmp_path / "masks"),
                             dte_worker=os.path.join(FIX, "fake_fs_dte_worker.py"),
                             mflm_worker=os.path.join(FIX, "fake_fs_mflm_worker.py"))
    samples = build_synthetic_dataset(str(tmp_path / "data"), 40, seed=7)
    try:
        res = run_pilot(samples, str(tmp_path / "run"), det, "fs",
                        os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), steps=[40],
                        n_boot=50, batch_size=8, log_every=0, s2_bases=("fake",))
    finally:
        det.close()
    rows = [json.loads(x) for x in open(tmp_path / "run" / "stage1" / "answers_fs.jsonl")]
    masked = [r for r in rows if r.get("needs_mask")]
    assert masked and all(r.get("mflm_done") and os.path.exists(r["mask_path"]) for r in masked)
    m = json.load(open(tmp_path / "run" / "results" / "fs_model" / "metrics.json"))
    assert m["by_base"]["fake"]["n"] > 0
    assert res["decision"] in ("GO", "PIVOT", "MORE_DATA", "UNDECIDED", "NOT_INFORMATIVE", "SWITCH_DETECTOR")
