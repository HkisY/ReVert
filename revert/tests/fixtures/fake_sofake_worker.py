"""Stand-in for sofake_worker.py speaking the same protocol (no torch, no model).
Answers like the faithful synthetic detector, but with hard decisions only
(p_fake None) and boxes, as So-Fake-R1 does."""
import argparse
import json
import os
import sys

import cv2

ap = argparse.ArgumentParser()
for k in ("--repo", "--weights", "--qwen-base", "--gpu-mem", "--cpu-mem", "--cache", "--gen-batch"):
    ap.add_argument(k)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from revert_audit.detectors import build_detector  # noqa: E402

det = build_detector("mock-local")
print("library noise", flush=True)
sys.stdout.write("@@REVERT@@" + json.dumps({"ready": True, "detector": "so-fake-r1"}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    if req["cmd"] == "quit":
        break
    answers = []
    for it in req["items"]:
        img = cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2RGB)
        r = det.predict([img])[0]
        fake = r["verdict"] == "fake"
        answers.append({"verdict": r["verdict"], "verdict_native": True, "p_fake": None,
                        "boxes": r["boxes"] if fake else [], "mask_path": None, "parse_mode": "native",
                        "sofake_scope": "TAMPERED" if fake else "REAL"})
    sys.stdout.write("@@REVERT@@" + json.dumps({"answers": answers}) + "\n")
    sys.stdout.flush()
