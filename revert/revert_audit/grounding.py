"""Map text-only explanations (FakeVLM, Veritas, AIGI-Holmes, closed APIs)
to image regions with an open-vocabulary detector (Grounding DINO).

    python -m revert_audit.grounding --queries <dir>/queries.jsonl \
        --answers <dir>/answers_fakevlm.jsonl --out <dir>/answers_fakevlm_grounded.jsonl

Claims are taken from `artifacts` (if the model produced structured
evidence) or split from `raw_text` into sentences. Each claim keeps at most
one box (the highest scoring above `box_threshold`). Claims with no box are
counted as GLOBAL claims ("overall too smooth") and reported separately;
they never enter the region interventions.

The mapping is noisy by construction. The paper must report its accuracy on
a hand-checked subset (plan: 300 samples) and show that the main findings
also hold on detectors that output regions natively (Qwen-VL, SIDA,
FakeShield).
"""

from __future__ import annotations

import argparse
import re

from .io import load_rgb, read_jsonl, write_jsonl

_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")
_SKIP = ("the image is", "this image is", "overall", "in conclusion", "therefore", "verdict")


def claims_from_answer(ans: dict, max_claims: int = 6) -> list[str]:
    arts = [a for a in (ans.get("artifacts") or []) if a]
    if arts:
        return arts[:max_claims]
    sents = [s.strip() for s in _SPLIT.split(ans.get("raw_text") or "") if len(s.strip()) > 8]
    sents = [s for s in sents if not s.lower().startswith(_SKIP)]
    return [s[:200] for s in sents[:max_claims]]


class Grounder:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda",
                 box_threshold: float = 0.3, text_threshold: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.bt, self.tt = box_threshold, text_threshold

    def ground(self, img, claim: str):
        from PIL import Image

        im = Image.fromarray(img)
        text = claim.lower().rstrip(".") + "."
        inputs = self.processor(images=im, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.bt, text_threshold=self.tt, target_sizes=[im.size[::-1]]
        )[0]
        if len(res["scores"]) == 0:
            return None
        i = int(res["scores"].argmax())
        return [float(v) for v in res["boxes"][i].tolist()]


def ground_answers(queries_path: str, answers_path: str, out_path: str, **kw) -> str:
    paths = {q["query_id"]: q["path"] for q in read_jsonl(queries_path)}
    g = Grounder(**kw)
    out = []
    for a in read_jsonl(answers_path):
        a = dict(a)
        if a.get("boxes") or a.get("mask_path") or a.get("verdict") != "fake":
            out.append(a)
            continue
        img = load_rgb(paths[a["query_id"]])
        boxes, glob = [], []
        for c in claims_from_answer(a):
            b = g.ground(img, c)
            (boxes if b else glob).append(b or c)
        a["boxes"] = boxes
        a["global_claims"] = glob
        a["region_source"] = "grounding-dino"
        out.append(a)
    write_jsonl(out, out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    a = ap.parse_args(argv)
    ground_answers(a.queries, a.answers, a.out, model_id=a.model, box_threshold=a.box_threshold)


if __name__ == "__main__":
    main()
