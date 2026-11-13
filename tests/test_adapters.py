import os

import numpy as np

from revert_audit.adapters import convert
from revert_audit.detectors.qwen_vl import QwenVLDetector
from revert_audit.io import read_jsonl, save_rgb, write_jsonl


def test_text_dir_adapter_rel1000(tmp_path):
    img = np.zeros((200, 400, 3), np.uint8)
    p = str(tmp_path / "images" / "s1__fake.png")
    save_rgb(img, p)
    q = str(tmp_path / "queries.jsonl")
    write_jsonl([{"query_id": "s1|s1|fake|orig|0", "path": p}], q)
    td = tmp_path / "out"
    os.makedirs(td)
    (td / "s1__fake.txt").write_text('{"verdict":"fake","evidence":[{"region":[500,500,1000,1000],"artifact":"x"}]}')
    out = convert(q, str(tmp_path / "ans.jsonl"), text_dir=str(td), coord="rel1000")
    a = read_jsonl(out)[0]
    assert a["verdict"] == "fake" and a["boxes"] == [[200.0, 100.0, 400.0, 200.0]] and a["p_fake"] is None


def test_sida_word_mapping(tmp_path):
    img = np.zeros((10, 10, 3), np.uint8)
    p = str(tmp_path / "a.png")
    save_rgb(img, p)
    q = str(tmp_path / "q.jsonl")
    write_jsonl([{"query_id": "a", "path": p}], q)
    td = tmp_path / "t"
    os.makedirs(td)
    (td / "a.txt").write_text("[CLS] The image is tampered. [SEG]")
    a = read_jsonl(convert(q, str(tmp_path / "o.jsonl"), text_dir=str(td),
                           fake_word_override={"tampered": "fake", "synthetic": "fake", "real": "real"}))[0]
    assert a["verdict"] == "fake"


def test_qwen_coordinate_mapping_without_model():
    d = object.__new__(QwenVLDetector)  # skip model loading
    d.patch = 14
    d.coord_mode = "abs"
    # resized input 28x14 patches -> 392 x 196 px; original 800 x 400
    assert d._to_orig([196, 98, 392, 196], 400, 800, [14, 28]) == [400.0, 200.0, 800.0, 400.0]
    d.coord_mode = "rel1000"
    assert d._to_orig([0, 0, 500, 1000], 400, 800, [14, 28]) == [0.0, 0.0, 400.0, 400.0]
