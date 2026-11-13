import json
import os

import numpy as np

from revert_audit import rationale as R
from revert_audit.io import save_mask, write_jsonl
from revert_audit.pipeline import qid


def test_terms_and_matching():
    t = R.terms("dining table")
    assert "table" in t and "tables" in t
    assert R.names("The table on the left has been tampered with.", t)
    assert not R.names("The tablecloth looks odd.", t)  # word boundary
    assert R.thing_of("a photo of a vase of flowers, realistic, natural lighting") == "vase of flowers"
    assert "vase" in R.terms("vase of flowers")
    assert R.names("Two women stand nearby", R.terms("person"))


def test_template_detected():
    ans = [{"query_id": "a|s1|fake|orig|0", "verdict": "fake", "raw_text": "[CLS] The image is tampered [SEG]"}]
    assert R.is_template(ans)
    assert not R.is_template([{"raw_text": "The picture has been tampered with. The dog ..."}])


def _fs(v, text="", mask=None):
    return {"verdict": v, "p_fake": 0.9 if v == "fake" else 0.1, "parse_mode": "native",
            "dte_text": text, "raw_text": text, "mask_path": mask}


def test_single_and_pairs(tmp_path):
    run = tmp_path / "full"
    s1, s2 = run / "stage1", run / "stage2_fakeshield_model"
    os.makedirs(s1), os.makedirs(s2)
    C = np.zeros((8, 8), bool)
    C[2:6, 2:6] = True
    cpath = str(tmp_path / "C.png")
    save_mask(C, cpath)
    samples = [{"id": f"x{i}", "coco_image_id": i, "coco_ann_id": 100 + i, "editor": "sdxl",
                "edit_type": "replace", "prompt": "a photo of a dog, realistic, natural lighting"} for i in range(4)]
    sp = str(tmp_path / "samples.jsonl")
    write_jsonl(samples, sp)
    a1 = [{"query_id": qid(s["id"], 1, "fake"), **_fs("fake", "The picture has been tampered with. The cat and the dog look edited.")}
          for s in samples]
    write_jsonl(a1, str(s1 / "answers_fakeshield.jsonl"))
    write_jsonl([{"sample_id": s["id"], "base": "fake", "skipped": None, "cmask": cpath, "ctrl_overlap": [0.0],
                  "cin_feasible": False} for s in samples], str(s2 / "meta.jsonl"))
    a2 = [{"query_id": qid("x0", 2, "fake", "revC", 0), **_fs("fake", "The cat was tampered with.", cpath)},
          {"query_id": qid("x1", 2, "fake", "revC", 0), **_fs("real")},
          {"query_id": qid("x2", 2, "fake", "revC", 0), **_fs("fake", "The sky was edited.", cpath)},
          {"query_id": qid("x3", 2, "fake", "revC", 0), **_fs("fake", "A dog was inserted.")}]
    write_jsonl(a2, str(s2 / "answers.jsonl"))
    out, rows = R.single(str(run), "fakeshield", sp, {100 + i: "cat" for i in range(4)}, n_boot=50)
    m = out["model"]
    assert m["named0"]["mean"] == 1.0
    assert m["rPER"]["mean"] == 0.5  # x0 (cat), x3 (dog); x1 real, x2 names neither
    assert m["rPER_obj"]["mean"] == 0.25 and m["rPER_thing"]["mean"] == 0.25
    assert m["joint_with_mask_PER"] == {"mask&text": 1, "mask_only": 1, "text_only": 1, "neither": 1}

    # two-object set
    pr = tmp_path / "e4b"
    ps1, comp = pr / "stage1", pr / "component_fakeshield"
    os.makedirs(ps1), os.makedirs(comp)
    write_jsonl([{"id": "p0", "coco_image_id": 9, "editor": "sdxl",
                  "parts": [{"obj": "cat", "edit_type": "remove"}, {"obj": "car", "edit_type": "remove"}]},
                 {"id": "p1", "coco_image_id": 10, "editor": "sdxl",
                  "parts": [{"obj": "person", "edit_type": "remove"}, {"obj": "person", "edit_type": "remove"}]}],
                str(pr / "pair_samples.jsonl"))
    write_jsonl([{"query_id": qid(p, 1, "fake"), **_fs("fake", "The cat is fake.")} for p in ("p0", "p1")],
                str(ps1 / "answers_fakeshield.jsonl"))
    write_jsonl([{"sample_id": p, "skipped": None, "cov_ok": True, "cov_cited": 0} for p in ("p0", "p1")],
                str(comp / "meta.jsonl"))
    write_jsonl([{"query_id": qid("p0", 2, "fake", "comp1", 0), **_fs("fake", "The car is fake.")},
                 {"query_id": qid("p0", 2, "fake", "comp2", 0), **_fs("fake", "The cat is fake.")},
                 {"query_id": qid("p1", 2, "fake", "comp1", 0), **_fs("real")},
                 {"query_id": qid("p1", 2, "fake", "comp2", 0), **_fs("real")}], str(comp / "answers.jsonl"))
    res, prow = R.pairs(str(pr), "fakeshield", n_boot=50)
    assert res["n_excluded_shared_terms"] == 1 and res["n_eligible_distinct_terms"] == 1
    assert res["rcPER"]["mean"] == 0.0 and res["rswitch"]["mean"] == 1.0 and res["rcSTAY"]["mean"] == 1.0
