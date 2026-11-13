"""Stand-in for sida_worker.py speaking the same protocol (no torch, no SIDA).
Calls an image 'tampered' when its mean is > 100 and cites its bright pixels;
an image whose top-left pixel is (1, 2, 3) makes it report an error."""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
for k in ("--repo", "--weights", "--mask-dir", "--max-new-tokens", "--clip", "--gpu-mem", "--protocol"):
    ap.add_argument(k)
a = ap.parse_args()
out = sys.stdout
print("some library noise on stdout", flush=True)
out.write("@@REVERT@@" + json.dumps({"ready": True, "n_offloaded_layers": 0, "weights_gib": 0.0}) + "\n")
out.flush()
while True:
    line = sys.stdin.readline()
    if not line:
        break
    req = json.loads(line)
    if req["cmd"] == "quit":
        break
    answers = []
    err = None
    for it in req["items"]:
        img = cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2RGB)
        if os.environ.get("FAKE_SIDA_MODE") == "mocklocal":  # behave like the faithful synthetic detector
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from revert_audit.detectors import build_detector
            from revert_audit.regions import boxes_to_mask

            r = build_detector("mock-local").predict([img])[0]
            ans = {"verdict": r["verdict"], "verdict_native": True, "p_fake": r["p_fake"], "boxes": [],
                   "parse_mode": "native", "mask_path": None}
            if r["verdict"] == "fake" and r["boxes"] and req.get("explain", True):
                os.makedirs(a.mask_dir, exist_ok=True)
                mp = os.path.join(a.mask_dir, it["key"] + ".png")
                cv2.imwrite(mp, boxes_to_mask(r["boxes"], *img.shape[:2]).astype(np.uint8) * 255)
                ans["mask_path"] = mp
            answers.append(ans)
            continue
        if tuple(img[0, 0]) == (1, 2, 3):
            err = "OutOfMemoryError: CUDA out of memory (simulated)"
            break
        m = float(img.mean())
        tampered = m > 100
        ans = {"verdict": "fake" if tampered else "real", "verdict_native": True, "p_fake": min(1.0, m / 255),
               "sida_class": "tampered" if tampered else "real", "boxes": [], "parse_mode": "native", "mask_path": None}
        if tampered and req.get("explain", True):
            os.makedirs(a.mask_dir, exist_ok=True)
            mp = os.path.join(a.mask_dir, it["key"] + ".png")
            cv2.imwrite(mp, ((img.mean(-1) > 100) * 255).astype(np.uint8))
            ans["mask_path"] = mp
        answers.append(ans)
    out.write("@@REVERT@@" + json.dumps({"error": err} if err else {"answers": answers}) + "\n")
    out.flush()
