"""Stand-in for fakeshield_worker.py (DTE-FDM): mock-local scores, text only for fakes."""
import argparse, json, os, sys
import cv2
ap = argparse.ArgumentParser()
for k in ("--repo", "--weights", "--gpu-mem", "--cpu-mem", "--cache", "--clip"):
    ap.add_argument(k)
ap.add_argument("--load-8bit", action="store_true")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from revert_audit.detectors import build_detector  # noqa: E402
det = build_detector("mock-local")
sys.stdout.write("@@REVERT@@" + json.dumps({"ready": True}) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    if req["cmd"] == "quit":
        break
    out = []
    for it in req["items"]:
        r = det.predict([cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2RGB)])[0]
        p = r["p_fake"]
        fake = p >= 0.5
        a_ = {"p_fake": p, "verdict": "fake" if fake else "real", "verdict_native": True, "boxes": None,
              "mask_path": None, "parse_mode": "native", "needs_mask": bool(fake and req.get("explain", True))}
        if a_["needs_mask"]:
            a_["dte_text"] = "The picture has been tampered with. BOXES=" + json.dumps(r["boxes"])
        out.append(a_)
    sys.stdout.write("@@REVERT@@" + json.dumps({"answers": out}) + "\n"); sys.stdout.flush()
