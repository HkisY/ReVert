import math

from revert_audit.budget import Design, audit_calls, estimate, format_estimate


def test_audit_call_counts_match_pipeline():
    # 3 stage-1 explains + revC on 2 bases; keepC + 3 revCt + 3 keepCt on 2 bases
    assert audit_calls(3) == (5, 14)
    assert audit_calls(5) == (5, 22)


def test_estimate_scales_linearly_and_keeps_design():
    d = Design()
    a = estimate(10.0, 1.0, {"sdxl": 10, "flux-fill": 100, "kontext": 100}, d)
    b = estimate(20.0, 2.0, {"sdxl": 20, "flux-fill": 200, "kontext": 200}, d)
    assert math.isclose(a["per_sample_model_sec"], 5 * 10 + 14 * 1)
    assert math.isclose(a["audit_h"], d.n_test * d.n_models * 64 / 3600)
    for k in ("pilot_h", "data_generation_h", "audit_h", "training_h"):
        assert math.isclose(b[k], 2 * a[k])
    assert "TOTAL" in format_estimate(a)
