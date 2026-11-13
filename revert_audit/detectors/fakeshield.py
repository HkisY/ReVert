"""FakeShield (ICLR 2025) as a ReVert detector: two worker processes, two envs.

  DTE-FDM (fakeshield_worker.py, env 'fsdte', transformers 4.37.2)
      p_fake, verdict and the textual explanation, during predict()
  MFLM    (fakeshield_mflm_worker.py, env 'fsmflm', transformers 4.28.0)
      the mask = the region FakeShield cites, in finalize()

Both models do not fit on a 16 GB GPU together, so predict() keeps DTE-FDM
loaded, and finalize() (called by pipeline.run_detector after each query file)
stops it, starts MFLM for every answer that needs a mask, writes the mask paths
into the answers file, and stops MFLM; DTE-FDM is restarted on the next predict().
"""

from __future__ import annotations

import json
import os

from ..io import load_rgb, read_jsonl
from .sida import WorkerDetector

HERE = os.path.dirname(os.path.abspath(__file__))
DTE_WORKER = os.path.join(HERE, "fakeshield_worker.py")
MFLM_WORKER = os.path.join(HERE, "fakeshield_mflm_worker.py")


MFLM_FULL_GIB = 14.4  # MFLM entirely on the GPU (--fs-mflm-gpu-mem 0), measured on the 4090
DTE_FULL_GIB = 26.0   # DTE-FDM (13B fp16) entirely on the GPU: does not fit 24 GB; use --gpu-mem 18


class _Client(WorkerDetector):
    def __init__(self, cmd, tag, tmp_dir=None, full_gib=None):
        self._start(cmd, os.getcwd(), tmp_dir, tag, full_gib=full_gib)

    def request(self, items):
        self.proc.stdin.write(json.dumps({"cmd": "mask", "items": items}) + "\n")
        self.proc.stdin.flush()
        resp = self._read()
        if "error" in resp:
            raise RuntimeError(f"{self._tag} worker: {resp['error']}")
        return resp["answers"]


class FakeShieldDetector(WorkerDetector):
    name = "fakeshield"

    def __init__(self, dte_python: str, mflm_python: str, repo: str, weights: str, clip: str | None = None,
                 gpu_mem: float | None = 11.0, mflm_gpu_mem: float | None = 9.5, cpu_mem: float | None = 40.0,
                 mask_dir: str = os.path.join("runs", "fakeshield_masks"), cache: str | None = None,
                 load_8bit: bool = False, load_4bit: bool = False, mflm_batch: int = 16,
                 dte_worker: str | None = None, mflm_worker: str | None = None, tmp_dir: str | None = None):
        for flag, p in (("--fs-dte-python", dte_python), ("--fs-mflm-python", mflm_python),
                        ("--fs-repo", repo), ("--fs-weights", weights)):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{flag} {p} not found")
        self.mask_dir = os.path.abspath(mask_dir)
        self.mflm_batch = mflm_batch
        self.tmp_dir = tmp_dir
        self.dte_cmd = [dte_python, "-u", dte_worker or DTE_WORKER, "--repo", repo, "--weights", weights,
                        "--gpu-mem", str(gpu_mem or 0), "--cpu-mem", str(cpu_mem or 40)]
        if cache:
            self.dte_cmd += ["--cache", os.path.abspath(cache)]
        if load_8bit:
            self.dte_cmd.append("--load-8bit")
        if load_4bit:
            self.dte_cmd.append("--load-4bit")
        self.mflm_cmd = [mflm_python, "-u", mflm_worker or MFLM_WORKER, "--repo", repo, "--weights", weights,
                         "--gpu-mem", str(mflm_gpu_mem or 0), "--mask-dir", self.mask_dir]
        if clip:
            self.dte_cmd += ["--clip", clip]
            self.mflm_cmd += ["--clip", clip]
        self.proc = None

    def predict(self, images, explain: bool = True):
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            self._start(self.dte_cmd, self.mask_dir, self.tmp_dir, "fakeshield-dte", full_gib=DTE_FULL_GIB)
        answers = super().predict(images, explain)
        for img, a in zip(images, answers):
            a["image_key"] = self.key(img)
        return answers

    def finalize(self, queries_path: str, answers_path: str):
        rows = read_jsonl(answers_path)
        todo = [r for r in rows if r.get("needs_mask") and not r.get("mflm_done")]
        if not todo:
            return
        paths = {q["query_id"]: q["path"] for q in read_jsonl(queries_path)}
        items, seen = [], {}
        for r in todo:
            k = r.get("image_key") or self.key(load_rgb(paths[r["query_id"]]))
            r["image_key"] = k
            if k not in seen:
                seen[k] = len(items)
                items.append({"key": k, "path": os.path.abspath(paths[r["query_id"]]), "text": r.get("dte_text") or r.get("raw_text", "")})
        print(f"[fakeshield] {len(items)} image(s) need a mask: stopping DTE-FDM, starting MFLM", flush=True)
        self.close()
        self.proc = None
        mflm = _Client(self.mflm_cmd, "fakeshield-mflm", self.tmp_dir, full_gib=MFLM_FULL_GIB)
        self.mflm_cmd = mflm.cmd_used  # keep a reduced GPU budget for the next stage (no failed load first)
        try:
            out = []
            for i in range(0, len(items), self.mflm_batch):
                out += mflm.request(items[i:i + self.mflm_batch])
        finally:
            mflm.close()
        res = {it["key"]: o for it, o in zip(items, out)}
        for r in rows:
            if r.get("needs_mask") and not r.get("mflm_done") and r.get("image_key") in res:
                o = res[r["image_key"]]
                r.update({"mask_path": o["mask_path"], "mflm_text": o["mflm_text"], "n_seg": o["n_seg"],
                          "mask_frac": o["mask_frac"], "mflm_done": True})
        tmp = answers_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, answers_path)
        print(f"[fakeshield] masks written for {len(res)} image(s)", flush=True)
