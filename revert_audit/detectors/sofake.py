"""So-Fake-R1 (Qwen2.5-VL-7B + two LoRA adapters + FETA evidence provider) as a
ReVert detector. The model runs in a worker process (sofake_worker.py) in the
same 'audit' env, so its sys.path and GPU memory stay separate from the audit.

Answers: verdict = final decision (FULL_SYNTHETIC / TAMPERED -> 'fake'),
verdict_native=True, p_fake=None (no end-to-end probability; the audit score is
the decision), boxes = the final TAMPERED box (224x224 frame scaled to the
image) = the region it cites. Observer / provider / route fields are kept.
"""

from __future__ import annotations

import os
import sys

from .sida import WorkerDetector

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sofake_worker.py")


class SoFakeDetector(WorkerDetector):
    name = "so-fake-r1"

    def __init__(self, repo: str, weights: str, qwen_base: str, python: str | None = None, gpu_mem: float | None = 11.0,
                 cpu_mem: float | None = 40.0, cache: str | None = os.path.join("runs", "sofake_cache.jsonl"),
                 worker: str | None = None, tmp_dir: str | None = None, gen_batch: int = 8):
        python = python or sys.executable
        for flag, path in (("--sofake-repo", repo), ("--sofake-weights", weights), ("--qwen-base", qwen_base)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{flag} {path} not found")
        cmd = [python, "-u", worker or WORKER, "--repo", repo, "--weights", weights, "--qwen-base", qwen_base,
               "--gpu-mem", str(gpu_mem or 0), "--cpu-mem", str(cpu_mem or 40), "--gen-batch", str(gen_batch)]
        if cache:
            cmd += ["--cache", os.path.abspath(cache)]
        self._start(cmd, os.path.dirname(os.path.abspath(cache or "runs/x")), tmp_dir, "sofake")
