import numpy as np
import pytest

from revert_audit.interventions import composite, keep, mask_from_difference, revert
from revert_audit.metrics import auc, bootstrap_ci, cng, csg, decoupling_rate
from revert_audit.parsing import normalize_verdict, parse_response
from revert_audit.regions import boxes_to_mask, iou, overlap_frac, translate_mask


def _imgs(h=32, w=40, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    raw = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    m = np.zeros((h, w), bool)
    m[5:15, 10:30] = True
    return real, raw, m


# ------------------------------------------------------------- interventions
def test_composite_outside_mask_is_bit_exact_real():
    real, raw, m = _imgs()
    f = composite(raw, real, m)
    assert np.array_equal(f[~m], real[~m])
    assert np.array_equal(f[m], raw[m])


def test_revert_introduces_only_real_pixels():
    real, raw, m = _imgs()
    f = composite(raw, real, m)
    C = np.zeros_like(m)
    C[0:20, 0:20] = True
    r = revert(f, real, C)
    assert np.array_equal(r[C], real[C])
    assert np.array_equal(r[~C], f[~C])
    # every pixel of the output equals either the base or the real image
    assert np.all((r == f).all(-1) | (r == real).all(-1))


def test_keep_full_mask_on_raw_equals_composite():
    # K(x_raw, M) == x_f : the decoupling test needs no extra query
    real, raw, m = _imgs()
    assert np.array_equal(keep(raw, real, m), composite(raw, real, m))


def test_revert_whole_mask_restores_real():
    real, raw, m = _imgs()
    assert np.array_equal(revert(composite(raw, real, m), real, m), real)


def test_shape_mismatch_raises():
    real, raw, m = _imgs()
    with pytest.raises(ValueError):
        revert(raw[:-1], real, m)


def test_mask_from_difference_recovers_edit():
    real, raw, m = _imgs(64, 64)
    real = np.full((64, 64, 3), 120, np.uint8)
    m = np.zeros((64, 64), bool)
    m[20:40, 10:30] = True
    edited = real.copy()
    edited[m] = 200
    est = mask_from_difference(real, edited, dilate_px=0)
    assert iou(est, m) > 0.8


# ------------------------------------------------------------------ regions
def test_boxes_to_mask_clips_and_orders():
    m = boxes_to_mask([[30, 20, -5, 2], [100, 100, 200, 200]], 25, 40)
    assert m.shape == (25, 40)
    assert m[2:20, 0:30].all() and m.sum() == 18 * 30


def test_translate_mask_preserves_shape_and_area():
    rng = np.random.default_rng(0)
    m = np.zeros((100, 120), bool)
    m[10:30, 20:35] = True
    m[50:55, 60:90] = True  # two components
    t, ov = translate_mask(m, rng)
    assert t.sum() == m.sum()
    assert ov <= 0.1
    # rigid shift: the pixel offsets are identical
    ys, xs = np.nonzero(m)
    ys2, xs2 = np.nonzero(t)
    d = {(int(a - b), int(c - e)) for a, b, c, e in zip(ys2, ys, xs2, xs)}
    assert len(d) == 1


def test_overlap_frac_and_iou():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[:, :5] = True
    assert overlap_frac(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(25 / 75)


# ------------------------------------------------------------------ parsing
def test_parse_json_in_code_fence():
    t = 'Sure!\n```json\n{"verdict": "fake", "evidence": [{"region": [1, 2, 3, 4], "artifact": "warped hand"}], "explanation": "x"}\n```'
    p = parse_response(t)
    assert p["verdict"] == "fake" and p["boxes"] == [[1.0, 2.0, 3.0, 4.0]] and p["parse_mode"] == "json"


def test_parse_regex_fallback():
    p = parse_response("Verdict: real. No regions.")
    assert p["verdict"] == "real" and p["parse_mode"] == "regex"


def test_parse_failure_is_reported():
    assert parse_response("I cannot tell.")["parse_mode"] == "failed"


@pytest.mark.parametrize("s,v", [("FAKE", "fake"), ("AI-generated", "fake"), ("not real", "fake"), ("authentic photo", "real"), ("not manipulated", "real"), ("??", None)])
def test_normalize_verdict(s, v):
    assert normalize_verdict(s) == v


# ------------------------------------------------------------------ metrics
def test_auc_known_values():
    assert auc([1, 2, 3], [0, 0, 0]) == 1.0
    assert auc([0, 0], [1, 1]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5


def test_cng_sign():
    faithful = [{"s_revC": 0.1, "s_revCt": [0.9, 0.8], "s_keepC": 0.9, "s_keepCt": [0.1, 0.2]}] * 5
    unfaithful = [{"s_revC": 0.9, "s_revCt": [0.9, 0.9], "s_keepC": 0.1, "s_keepCt": [0.1, 0.1]}] * 5
    assert cng(faithful) > 0.7 and abs(cng(unfaithful)) < 1e-9
    assert csg(faithful) > 0.7 and abs(csg(unfaithful)) < 1e-9


def test_decoupling_rate():
    pairs = [{"v_raw": "fake", "gp_raw": 0.9, "v_fake": "real"}, {"v_raw": "fake", "gp_raw": 0.9, "v_fake": "fake"}, {"v_raw": "fake", "gp_raw": 0.1, "v_fake": "real"}]
    dr, n = decoupling_rate(pairs)
    assert n == 2 and dr == 0.5


def test_bootstrap_ci_contains_point():
    xs = list(np.random.default_rng(0).normal(0, 1, 200))
    p, lo, hi = bootstrap_ci(xs, lambda v: float(np.mean(v)), 300)
    assert lo <= p <= hi


def test_translate_mask_avoids_given_region():
    rng = np.random.default_rng(0)
    h = w = 100
    C = np.zeros((h, w), bool)
    C[10:30, 10:30] = True
    M = np.zeros((h, w), bool)
    M[40:90, 0:100] = True  # a big true edit below the cited region
    for _ in range(20):
        t, ov = translate_mask(C, rng, avoid=C | M)
        assert t.sum() == C.sum() and ov <= 0.1
        assert overlap_frac(t, M) <= 0.1  # never removes the real evidence


def test_blob_in_region_and_matched_metric():
    from revert_audit.metrics import mng
    from revert_audit.regions import blob_in_region

    rng = np.random.default_rng(1)
    region = np.zeros((50, 50), bool)
    region[10:40, 5:20] = True
    b = blob_in_region(region, 100, rng)
    assert b.sum() == 100 and not (b & ~region).any()
    assert blob_in_region(region, 10_000, rng) is None
    # failure control: a citation that does not matter gives MNG ~ 0, one that does gives > 0
    inert = [{"s_revC": 0.8, "s_revCin": [0.8, 0.78, 0.82]}]
    causal = [{"s_revC": 0.1, "s_revCin": [0.8, 0.78, 0.82]}]
    assert abs(mng(inert)) < 1e-9 and mng(causal) > 0.6


def test_per_counts_flips_to_real_for_mask_only_detectors():
    from revert_audit.pipeline import _explained

    assert _explained({"boxes": [], "verdict": "real"})
    assert _explained({"boxes": None, "mask_path": None, "verdict": "real", "verdict_native": True, "parse_mode": "native"})
    assert not _explained({"boxes": None, "p_fake": 0.2, "parse_mode": "score_only", "verdict": "real"})
    assert not _explained({"boxes": None, "parse_mode": "failed", "verdict": None})


def test_seam_alpha_modes():
    import numpy as np

    from revert_audit.interventions import revert_seam, seam_alpha

    r = np.zeros((40, 40), bool)
    r[15:25, 15:25] = True
    h = seam_alpha(r, "hard")
    ramp = seam_alpha(r, "ramp:8")
    dil = seam_alpha(r, "dilate:4")
    assert (ramp[r] == 1).all() and (dil[r] == 1).all() and (h[~r] == 0).all()
    assert 0 < ramp[15, 12] < 1 and ramp[15, 5] == 0 and dil[15, 12] == 1 and dil[15, 5] == 0
    extra_ramp, extra_dil = ramp[~r].sum(), dil[~r].sum()
    assert 0.6 < extra_ramp / extra_dil < 1.6  # about the same extra reverted area
    base = np.full((40, 40, 3), 200, np.uint8)
    real = np.zeros((40, 40, 3), np.uint8)
    out = revert_seam(base, real, r, "ramp:8")
    assert (out[r] == 0).all() and out[0, 0, 0] == 200


def test_component_test_on_synthetic_pairs(tmp_path):
    """Faithful stand-in: cites the strong object; restoring it removes the 'fake'
    verdict (cPER 0, cMNG > 0); restoring the weak object changes nothing (cSTAY 1)."""
    import os

    from revert_audit.component import print_report, run_component
    from revert_audit.detectors import build_detector
    from revert_audit.io import load_rgb, load_mask, read_jsonl
    from revert_audit.synthetic import build_synthetic_pairs

    sp = build_synthetic_pairs(str(tmp_path / "data"), 30, seed=2, weak=0.05)
    res = run_component(sp, str(tmp_path / "run"), build_detector("mock-local"), "loc", n_boot=50, batch_size=16,
                        log_every=0)
    print_report(res)
    assert res["n"] >= 20
    from revert_audit.component import eligibility_summary

    es = eligibility_summary(str(tmp_path / "run" / "pair_samples.jsonl"), str(tmp_path / "run" / "stage1" / "answers_loc.jsonl"))
    assert es["flagged"] == 30 and es["coverage_rule cited>=0.5,other<0.1"] == 30
    assert res["cPER"][0] < 0.1 and res["cSTAY"][0] > 0.8 and res["cMNG"][0] > 0.3
    # whole-object reversion leaves exactly a one-object forgery: compC == x_f with the cited object real
    out = tmp_path / "run" / "component_loc"
    meta = [m for m in read_jsonl(str(out / "meta.jsonl")) if not m.get("skipped")]
    m = meta[0]
    s = {r["id"]: r for r in read_jsonl(sp)}[m["sample_id"]]
    c = m["cov_cited"]
    img = load_rgb(str(out / "images" / f"{m['sample_id']}__comp{c + 1}.png"))
    real, fake = load_rgb(s["real"]), load_rgb(s["fake"])
    mc = load_mask(s["parts"][c]["mask"])
    assert (img[mc] == real[mc]).all() and (img[~mc] == fake[~mc]).all()


def test_pair_source_rejects_touching_objects():
    import numpy as np

    from revert_audit.data.masks import dilate

    a = np.zeros((100, 100), bool)
    b = np.zeros((100, 100), bool)
    a[10:30, 10:30] = True
    b[10:30, 45:65] = True  # 15 px apart
    assert (dilate(a, 24) & b).any()  # would be rejected with min_gap 24
    assert not (dilate(a, 10) & b).any()
