"""Stand-in for fakeshield_mflm_worker.py: draws the boxes named in the DTE text as the mask."""
import argparse, json, os, sys
import cv2
import numpy as np
ap = argparse.ArgumentParser()
for k in ("--repo", "--weights", "--gpu-mem", "--mask-dir", "--clip"):
    ap.add_argument(k)
a = ap.parse_args()
sys.stdout.write("@@REVERT@@" + json.dumps({"ready": True}) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    if req["cmd"] == "quit":
        break
    out = []
    for it in req["items"]:
        h, w = cv2.imread(it["path"]).shape[:2]
        m = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in json.loads(it["text"].split("BOXES=")[1]):
            m[int(y1):int(y2), int(x1):int(x2)] = 255
        os.makedirs(a.mask_dir, exist_ok=True)
        mp = os.path.join(a.mask_dir, it["key"] + ".png")
        cv2.imwrite(mp, m)
        out.append({"mask_path": mp, "mflm_text": "Sure, [SEG].", "n_seg": 1, "mask_frac": float((m > 0).mean())})
    sys.stdout.write("@@REVERT@@" + json.dumps({"answers": out}) + "\n"); sys.stdout.flush()
