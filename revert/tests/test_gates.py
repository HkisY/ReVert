"""The stop rules must reach the right decision for every known situation,
and the progressive pilot must stop as soon as a decision is reached."""

import json
import os

import numpy as np
import pytest

from revert_audit.detectors.base import Detector
from revert_audit.gates import evaluate, load_config
from revert_audit.pilot import run_pilot
from revert_audit.detectors import build_detector
from revert_audit.synthetic import build_synthetic_dataset

CFG = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"))


def M(cng=(0.5, 0.4, 0.6), per=(0.0, 0.0, 0.05), n=60, dr=(0.0, 0.0, 0.05), dr_n=40, auc=0.9, pf=0.0, n_real=60):
    return {"stage1": {"parse_fail_rate": pf, "auc_fake_vs_real": auc, "auc_raw_vs_real": auc, "n_real": n_real},
            "by_base": {"fake": {"n": n, "CNG": list(cng), "PER": list(per)}}, "DR": list(dr), "DR_n_eligible": dr_n}


ORACLE, RANDOM = M(cng=(0.8, 0.75, 0.85)), M(cng=(0.0, -0.02, 0.02))


@pytest.mark.parametrize("model,expected", [
    (M(cng=(0.78, 0.72, 0.83)), "PIVOT"),                      # R ~ 0.97, DR/PER low
    (M(cng=(0.20, 0.15, 0.25)), "GO"),                         # R = 0.25
    (M(cng=(0.78, 0.72, 0.83), dr=(0.5, 0.35, 0.65)), "GO"),   # faithful CNG but decoupled
    (M(cng=(0.78, 0.72, 0.83), per=(0.4, 0.3, 0.5)), "GO"),    # phantom evidence
    (M(cng=(0.66, 0.5, 0.8)), "MORE_DATA"),                    # R ~ 0.82: in between
    (M(pf=0.3), "FIX_PROMPT"),
    (M(auc=0.5, n_real=150), "SWITCH_DETECTOR"),
    (M(auc=0.5, n_real=40, cng=(0.66, 0.5, 0.8)), "MORE_DATA"),  # too few reals to call it
    (M(n=10), "MORE_DATA"),
])
def test_decisions(model, expected):
    assert evaluate({"model": model, "oracle": ORACLE, "random": RANDOM}, CFG)["decision"] == expected


def test_protocol_not_informative():
    oracle_flat = M(cng=(0.02, -0.01, 0.05))
    r = evaluate({"model": M(cng=(0.0, -0.02, 0.02)), "oracle": oracle_flat, "random": RANDOM}, CFG)
    assert r["gates"]["G2"]["status"] == "FAIL" and r["decision"] == "NOT_INFORMATIVE"


class Blind(Detector):
    """Cannot tell real from fake: a detector whose explanations are not auditable."""
    name = "blind"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def predict(self, images, explain=True):
        out = []
        for im in images:
            p = float(self.rng.uniform(0.3, 0.7))
            h, w = im.shape[:2]
            out.append({"verdict": "fake" if p >= 0.5 else "real", "p_fake": p, "boxes": [[0, 0, w / 4, h / 4]],
                        "parse_mode": "native"})
        return out


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pilot"))
    return root, build_synthetic_dataset(os.path.join(root, "data"), 160, seed=7)


def test_pilot_stops_early_when_decided(data):
    root, samples = data
    r = run_pilot(samples, os.path.join(root, "local"), build_detector("mock-local"), "local",
                  os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), n_boot=100, batch_size=16, log_every=0)
    assert r["decision"] == "PIVOT" and r["step_n"] == 50
    assert not os.path.exists(os.path.join(root, "local", "gates_step150.json"))
    html = open(os.path.join(root, "local", "report_local.html"), encoding="utf-8").read()
    assert "PIVOT" in html and "data:image/jpeg;base64," in html


def test_pilot_grows_then_switches_detector(data):
    root, samples = data
    r = run_pilot(samples, os.path.join(root, "blind"), Blind(), "blind",
                  os.path.join(os.path.dirname(__file__), "..", "configs", "gates.json"), n_boot=100, batch_size=16, log_every=0)
    assert r["decision"] == "SWITCH_DETECTOR" and r["step_n"] == 150
    with open(os.path.join(root, "blind", "gates_step50.json"), encoding="utf-8") as f:
        assert json.load(f)["decision"] == "MORE_DATA"


def _never_fake(n_fake, acc=0.0):
    m = M(n=0)
    m["stage1"].update({"n_fake": n_fake, "n_raw": n_fake, "acc_fake": acc, "acc_raw": acc, "n_real": n_fake})
    return m


def test_never_flags_fake():
    # the smoke-test situation: zero-shot model answers "real" for every fake
    assert evaluate({"model": _never_fake(50), "oracle": ORACLE, "random": RANDOM}, CFG)["decision"] == "NO_FLAGS"
    # 10 fakes are too few to call it (0/10 is still compatible with a 30% flag rate)
    assert evaluate({"model": _never_fake(10), "oracle": ORACLE, "random": RANDOM}, CFG)["decision"] == "MORE_DATA"
    # flagging 30% of 50 fakes is enough to go on collecting
    assert evaluate({"model": _never_fake(50, acc=0.3), "oracle": ORACLE, "random": RANDOM}, CFG)["decision"] != "NO_FLAGS"


def test_g1_needs_enough_reals_to_pass():
    r = evaluate({"model": M(n_real=10), "oracle": ORACLE, "random": RANDOM}, CFG)
    assert r["gates"]["G1"]["status"] == "INCONCLUSIVE" and r["decision"] == "MORE_DATA"


def test_g2_needs_enough_samples():
    r = evaluate({"model": M(n=2), "oracle": ORACLE, "random": RANDOM}, CFG)
    assert r["gates"]["G2"]["status"] == "INCONCLUSIVE"
