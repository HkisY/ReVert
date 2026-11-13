"""A deliberately simple 'known-cheater' calibration detector on REAL data.

A linear probe on frozen CLIP features, trained on x_raw (fake) vs x_r
(real), is expected to pick up the global VAE trace of inpainting (INP-X).
Paired with the ORACLE citation (C = M), it is a detector whose explanation
is perfectly placed but whose decision does not depend on it, so the audit
must report a high DR and a near-zero CNG for it. The same probe trained on
the twin set (x_f vs x_r) cannot use the global trace and should show a
lower DR. If the audit cannot tell these two apart on your data, stop and
debug before auditing real MLLMs.

Train:
    python -m revert_audit.detectors.clip_probe --samples train/samples.jsonl --mode raw --out probe_raw.pkl
Use:
    python -m revert_audit.cli run --detector clip-probe --probe probe_raw.pkl ...
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from ..io import load_rgb, read_jsonl
from .base import Detector

CLIP_ID = "openai/clip-vit-large-patch14"


class _Feat:
    def __init__(self, model_id=CLIP_ID, device="cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        x = self.proc(images=[Image.fromarray(i) for i in images], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            f = self.model.get_image_features(pixel_values=x["pixel_values"].half())
        f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()


class ClipProbeDetector(Detector):
    def __init__(self, probe_path: str, model_id: str = CLIP_ID):
        with open(probe_path, "rb") as fh:
            self.clf = pickle.load(fh)
        self.feat = _Feat(model_id)
        self.name = f"clip-probe:{probe_path}"

    def predict(self, images, explain=True):
        p = self.clf.predict_proba(self.feat(images))[:, 1]
        return [{"verdict": "fake" if q >= 0.5 else "real", "p_fake": float(q), "boxes": None, "mask_path": None,
                 "artifacts": [], "raw_text": "", "parse_mode": "native"} for q in p]


def train(samples_path: str, mode: str, out: str, batch: int = 32):
    from sklearn.linear_model import LogisticRegression

    rows = read_jsonl(samples_path)
    pos_key = "raw" if mode == "raw" else "fake"
    fx = _Feat()
    X, y = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        X.append(fx([load_rgb(r[pos_key]) for r in chunk])); y += [1] * len(chunk)
        X.append(fx([load_rgb(r["real"]) for r in chunk])); y += [0] * len(chunk)
    X = np.concatenate(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, np.asarray(y))
    print(f"[clip_probe] train acc ({mode}) = {clf.score(X, y):.3f} on {len(y)} images")
    with open(out, "wb") as fh:
        pickle.dump(clf, fh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--mode", choices=["raw", "twin"], default="raw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train(a.samples, a.mode, a.out)
