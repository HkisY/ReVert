"""The checks must be able to fail.

On the synthetic world each mock detector has a KNOWN failure mode. Every
metric must separate the faithful detector from the detector that breaks
it, and the oracle / random explainers must bracket the scale. If any of
these tests fails after a code change, no number from a real model can be
trusted.
"""

import os

import pytest

from revert_audit import pipeline
from revert_audit.detectors import build_detector
from revert_audit.synthetic import build_synthetic_dataset

N = 24


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("world"))
    samples = build_synthetic_dataset(os.path.join(root, "data"), N, seed=3)
    s1 = os.path.join(root, "stage1")
    q1 = pipeline.stage1(samples, s1)
    return root, samples, s1, q1


def _audit(world, det_name, explainer="model"):
    root, samples, s1, q1 = world
    det = build_detector(det_name)
    a1 = os.path.join(s1, f"answers_{det_name}.jsonl")
    if not os.path.exists(a1):
        pipeline.run_detector(q1, det, a1, batch_size=16, log_every=0)
    s2 = os.path.join(root, f"s2_{det_name}_{explainer}")
    q2 = pipeline.stage2(samples, s1, a1, s2, explainer=explainer, n_controls=3)
    a2 = pipeline.run_detector(q2, det, os.path.join(s2, "answers.jsonl"), batch_size=16, log_every=0)
    return pipeline.analyze(samples, a1, os.path.join(s2, "meta.jsonl"), a2, os.path.join(root, "res", f"{det_name}_{explainer}"), n_boot=100)


def _get(res, base, key):
    return res["by_base"][base][key][0]


def test_detection_sanity(world):
    r = _audit(world, "mock-local")
    assert r["stage1"]["acc_fake"] == 1.0 and r["stage1"]["acc_real"] == 1.0


def test_cng_separates_faithful_from_random_citation(world):
    local = _audit(world, "mock-local")
    rnd = _audit(world, "mock-random")
    assert _get(local, "fake", "CNG") > 0.6
    assert abs(_get(rnd, "fake", "CNG")) < 0.15
    assert _get(local, "fake", "GP") > _get(rnd, "fake", "GP") + 0.3


def test_oracle_and_random_explainers_bracket_the_scale(world):
    ora = _audit(world, "mock-local", "oracle")
    rnd = _audit(world, "mock-local", "random")
    assert _get(ora, "fake", "CNG") > 0.8 and _get(ora, "fake", "GP") == pytest.approx(1.0)
    assert abs(_get(rnd, "fake", "CNG")) < 0.15


def test_decoupling_catches_global_cheater(world):
    cheat = _audit(world, "mock-cheater")
    local = _audit(world, "mock-local")
    assert cheat["DR"][0] > 0.9 and cheat["DR_n_eligible"] > 0
    assert local["DR"][0] < 0.1
    # cheater cites the right place on x_raw, yet its decision does not depend on it
    assert _get(cheat, "raw", "GP") > 0.4
    assert abs(_get(cheat, "raw", "CNG")) < 0.15


def test_cheater_unfaithful_even_with_oracle_citation(world):
    cheat = _audit(world, "mock-cheater", "oracle")
    assert abs(_get(cheat, "raw", "CNG")) < 0.15


def test_phantom_evidence_rate(world):
    fixed = _audit(world, "mock-fixed")
    local = _audit(world, "mock-local")
    assert _get(fixed, "fake", "PER") > 0.9
    assert _get(local, "fake", "PER") < 0.1
