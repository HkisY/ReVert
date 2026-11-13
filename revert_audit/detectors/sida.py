"""SIDA-7B (CVPR 2025) as a ReVert detector.

SIDA needs transformers 4.31 and cannot live in the audit environment, so it
runs in its own conda env as a persistent worker process (sida_worker.py) and
this class talks to it over stdin/stdout. From the pipeline's point of view it
is an ordinary Detector: same resumable runs, same pilot and gates.

What the answers contain
  verdict      SIDA's own decision (argmax of real / full synthetic / tampered);
               'fake' = full synthetic or tampered. verdict_native=True makes the
               pipeline use this decision instead of thresholding p_fake.
  p_fake       1 - P(real) from SIDA's classification head (softmax of its logits)
  mask_path    SIDA's predicted tampering mask (only when it says 'tampered') =
               the region it cites, i.e. C in the audit
  sida_class, probs, n_seg_masks, mask_frac, raw_text
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from .base import Detector

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sida_worker.py")
PROTO = "@@REVERT@@"


def wait_gpu_release(pid: int, timeout: float = 120.0) -> None:
    """Block until the driver no longer lists `pid` as a GPU user (a process that has exited can hold its
    memory for a few seconds; the next worker, e.g. FakeShield MFLM after DTE-FDM, needs that memory)."""
    import shutil
    import time

    if not shutil.which("nvidia-smi"):
        time.sleep(3)
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001
            return
        if str(pid) not in {x.strip() for x in out.splitlines()}:
            return
        time.sleep(2)
    print(f"[worker] warning: process {pid} still holds GPU memory after {timeout:.0f}s", flush=True)


def _smi(args: list[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20).stdout


def gpu_holders() -> list[tuple[int, float, str]]:
    """(pid, GiB, command line) of every process the driver lists on the GPU. A process that `ps` cannot see
    (another user's job with /proc hidden, or another container) is shown by the driver's process name."""
    rows = []
    for line in _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            pid, name, gib = int(parts[0]), ",".join(parts[1:-1]), float(parts[-1]) / 1024
        except (ValueError, IndexError):
            continue
        try:
            cmdline = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True,
                                     timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            cmdline = ""
        rows.append((pid, gib, cmdline[:160] or f"{name or '?'} [not visible to ps: other user or container]"))
    return rows


def _gpu_query_gib(field: str) -> float | None:
    dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    try:
        out = _smi([f"--query-gpu=index,{field}", "--format=csv,noheader,nounits"])
    except Exception:  # noqa: BLE001
        return None
    vals = {}
    for line in out.splitlines():
        try:
            i, mib = (x.strip() for x in line.split(","))
            vals[i] = float(mib) / 1024
        except ValueError:
            continue
    return vals.get(dev, next(iter(vals.values()), None))


def gpu_free_gib() -> float | None:
    return _gpu_query_gib("memory.free")


def gpu_total_gib() -> float | None:
    return _gpu_query_gib("memory.total")


GPU_RESERVE_GIB = 2.5  # desktop, remote-desktop tools and other small resident GPU users


def clamp_need(need: float, total: float | None, reserve: float = GPU_RESERVE_GIB) -> float:
    """--gpu-mem is a CAP on what a worker may place on the GPU (SIDA runs with 22 on a 24 GB card and
    actually uses ~16), not what it needs: a need above the card minus a small reserve could never be met
    (the desktop alone holds a few hundred MB). Clamp it so the wait only blocks on other GPU jobs."""
    if total and need > total - reserve:
        return max(0.0, total - reserve)
    return need


GPU_MARGIN_GIB = 1.5  # CUDA context + activations on top of the weights a worker is allowed to put on the GPU


def wait_gpu_free(need_gib: float, tag: str, wait_min: float | None = None, poll_s: float = 30.0,
                  stable: int = 2) -> None:
    """Before loading a worker: wait until `need_gib` is free on the GPU (another job, e.g. a data build,
    may be using it). Waits up to REVERT_GPU_WAIT_MIN minutes (default 60), then stops with a short
    message naming the processes that hold the memory, instead of a page of CUDA OOM tracebacks."""
    import shutil
    import time

    if not need_gib or not shutil.which("nvidia-smi"):
        return
    wait_min = float(os.environ.get("REVERT_GPU_WAIT_MIN", 60)) if wait_min is None else wait_min
    t0, last_note, ok_streak = time.time(), -1e9, 0
    while True:
        free = gpu_free_gib()
        if free is None:
            return
        if free >= need_gib:
            ok_streak += 1
            if ok_streak >= stable:  # free on consecutive checks: not just a gap between two runs of another job
                return
            time.sleep(min(poll_s, 10.0))
            continue
        ok_streak = 0
        holders = gpu_holders()
        held = "; ".join(f"pid {p} {g:.1f} GiB: {c}" for p, g, c in holders) or "no process listed"
        waited = (time.time() - t0) / 60
        if waited >= wait_min:
            raise RuntimeError(f"[{tag}] GPU busy: needs {need_gib:.1f} GiB, {free:.1f} GiB free after waiting "
                               f"{waited:.0f} min. Held by: {held}. Stop that job (or let it finish) and rerun the "
                               f"same command; answers already written are kept.")
        if time.time() - last_note >= 600:
            print(f"[{tag}] waiting for GPU memory: needs {need_gib:.1f} GiB, {free:.1f} GiB free "
                  f"(waited {waited:.0f}/{wait_min:.0f} min). Held by: {held}", flush=True)
            last_note = time.time()
        time.sleep(poll_s)


def gpu_need_from_cmd(cmd: list[str], full_gib: float | None = None) -> float:
    """The worker's --gpu-mem budget (GiB) + margin. --gpu-mem 0 means 'whole model on the GPU': then the
    caller's estimate `full_gib` is used (e.g. FakeShield MFLM: 14.4 GiB measured on the 4090)."""
    if "--gpu-mem" in cmd:
        try:
            g = float(cmd[cmd.index("--gpu-mem") + 1])
        except (IndexError, ValueError):
            g = 0.0
        if g > 0:
            # --gpu-mem is a cap: a worker never puts more than its whole model on the GPU
            return (min(g, full_gib) if full_gib else g) + GPU_MARGIN_GIB
    return full_gib + GPU_MARGIN_GIB if full_gib else 0.0


def shrink_budget(cmd: list[str], free_gib: float | None, reserve: float = 2.5, floor: float = 6.0) -> float | None:
    """A whole-model worker (--gpu-mem 0) that could not load beside another job: GPU budget that fits the
    memory free now (the other job still running), so the rest of the model is offloaded to CPU RAM.
    Workers with an explicit budget (DTE-FDM: 18) are not shrunk: they wait instead."""
    if "--gpu-mem" not in cmd or not free_gib:
        return None
    try:
        g = float(cmd[cmd.index("--gpu-mem") + 1])
    except (IndexError, ValueError):
        return None
    b = round(free_gib - reserve, 1)
    return b if g == 0 and b >= floor else None


def describe_gpu_holders(min_gib: float = 0.5) -> str:
    try:
        rows = [r for r in gpu_holders() if r[1] >= min_gib]
    except Exception:  # noqa: BLE001
        return "(nvidia-smi not available)"
    return "; ".join(f"pid {p} {g:.1f} GiB: {c}" for p, g, c in rows) or "none"


class WorkerDetector(Detector):
    """A detector that lives in a separate process (own conda env and/or own
    model code) and answers over stdin/stdout lines prefixed with PROTO."""

    name = "worker"

    full_gpu_gib: float | None = None  # GPU need when the command says --gpu-mem 0 (whole model on the GPU)

    def _start(self, cmd: list[str], mask_dir: str, tmp_dir: str | None, tag: str, full_gib: float | None = None):
        self.mask_dir = os.path.abspath(mask_dir)
        self.tmp = tmp_dir or tempfile.mkdtemp(prefix=f"revert_{tag}_")
        self._tag = tag
        cmd = list(cmd)
        need = gpu_need_from_cmd(cmd, full_gib if full_gib is not None else self.full_gpu_gib)
        if need:
            need = clamp_need(need, gpu_total_gib())
        if os.environ.get("REVERT_GPU_NEED_GIB"):  # manual override; 0 switches the pre-start wait off
            need = float(os.environ["REVERT_GPU_NEED_GIB"])
        tries = max(1, int(os.environ.get("REVERT_LOAD_RETRIES", 4)))
        for attempt in range(1, tries + 1):
            wait_gpu_free(need, tag)
            print(f"[{tag}] starting worker: {' '.join(cmd)}", flush=True)
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                                         text=True, encoding="utf-8", bufsize=1)
            try:
                hello = self._read()
                break
            except RuntimeError:
                # the worker died while loading (usually CUDA OOM because another job took memory meanwhile)
                try:
                    self.proc.wait(timeout=60)
                except Exception:  # noqa: BLE001
                    self.proc.kill()
                wait_gpu_release(self.proc.pid, timeout=60)
                held = describe_gpu_holders()
                print(f"[{tag}] worker failed to load (attempt {attempt}/{tries}). GPU now held by: {held}", flush=True)
                if attempt == tries:
                    raise RuntimeError(f"[{tag}] worker exited while loading ({tries} attempts). GPU held by: {held}. "
                                       f"If a process above is not ours, stop it and rerun; answers are kept.")
                budget = shrink_budget(cmd, gpu_free_gib())
                if budget:
                    cmd[cmd.index("--gpu-mem") + 1] = f"{budget:g}"
                    need = budget + GPU_MARGIN_GIB
                    print(f"[{tag}] retrying with --gpu-mem {budget:g}: part of the model stays in CPU RAM "
                          f"(slower, same outputs), so it fits beside the other job", flush=True)
        self.cmd_used = cmd
        if not hello.get("ready"):
            raise RuntimeError(f"{tag} worker did not start: {hello}")
        self.info = hello

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise RuntimeError(f"{self._tag} worker exited (code {code}); see its messages above")
            if line.startswith(PROTO):
                return json.loads(line[len(PROTO):])
            sys.stderr.write(line)  # stray output: show it, do not parse it

    @staticmethod
    def key(img: np.ndarray) -> str:
        h = hashlib.sha1(np.ascontiguousarray(img).tobytes())
        h.update(str(img.shape).encode())
        return h.hexdigest()[:20]

    def predict(self, images: list[np.ndarray], explain: bool = True) -> list[dict]:
        import cv2

        items = []
        for img in images:
            k = self.key(img)
            p = os.path.join(self.tmp, k + ".png")
            if not os.path.exists(p):
                cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # lossless: the model sees the exact pixels
            items.append({"path": p, "key": k})
        self.proc.stdin.write(json.dumps({"cmd": "predict", "items": items, "explain": explain}) + "\n")
        self.proc.stdin.flush()
        resp = self._read()
        if "error" in resp:
            raise RuntimeError(f"{self._tag} worker: {resp['error']}")  # 'out of memory' text triggers the OOM path
        for it in items:
            try:
                os.remove(it["path"])
            except OSError:
                pass
        return resp["answers"]

    def close(self):
        proc = getattr(self, "proc", None)
        if not proc:
            return
        if proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=120)
            except Exception:  # noqa: BLE001
                proc.kill()
                try:
                    proc.wait(timeout=60)
                except Exception:  # noqa: BLE001
                    pass
            wait_gpu_release(proc.pid)  # only after stopping a live worker (not again from __del__)

    def __del__(self):
        self.close()


class SidaDetector(WorkerDetector):
    name = "sida-7b"
    full_gpu_gib = 18.0  # SIDA-7B (LLaVA-7B + SAM ViT-H, bf16) entirely on the GPU, with headroom

    def __init__(self, python: str, repo: str, weights: str, clip: str | None = None, gpu_mem: float | None = 12.0,
                 mask_dir: str = os.path.join("runs", "sida_masks"), max_new_tokens: int = 512,
                 worker: str | None = None, tmp_dir: str | None = None, protocol: str = "paper"):
        if not os.path.exists(python):
            raise FileNotFoundError(f"--sida-python {python} not found (the python.exe of the 'sida' conda env)")
        cmd = [python, "-u", worker or WORKER, "--repo", repo, "--weights", weights, "--mask-dir",
               os.path.abspath(mask_dir), "--max-new-tokens", str(max_new_tokens), "--protocol", protocol]
        if clip:
            cmd += ["--clip", clip]
        if gpu_mem:
            cmd += ["--gpu-mem", str(gpu_mem)]
        self._start(cmd, mask_dir, tmp_dir, "sida")
