"""Software guard against Windows "sysmem fallback".

On Windows the NVIDIA driver can silently spill GPU allocations into system
RAM when VRAM is full: nothing errors, everything gets 5-10x slower, and our
automatic batch halving never triggers. The clean fix is the driver setting
"CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback". If that setting
cannot be applied, cap PyTorch's own allocator below the physical VRAM: then
PyTorch raises a normal out-of-memory error before the driver would spill,
and the batch is halved as intended.
"""

from __future__ import annotations


def cap_vram(fraction: float | None, device: int = 0) -> bool:
    """Limit this process's PyTorch allocations to `fraction` of the GPU's
    memory. Returns True if a cap was set."""
    if not fraction:
        return False
    if not 0.1 <= fraction <= 1.0:
        raise ValueError("--vram-cap must be between 0.1 and 1.0 (e.g. 0.92)")
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"[gpu] PyTorch allocations capped at {fraction:.0%} of {total:.1f} GiB = {fraction * total:.1f} GiB")
    return True


# ---------------------------------------------------------------- host memory
# Windows-specific failure seen with FLUX + sequential CPU offload: the bf16
# weights (~34 GB for FLUX.1-Fill incl. the T5 text encoder) live in system RAM.
# When RAM + page file (the Windows "commit limit") runs out, two confusing
# errors appear: an access violation (exit -1073741819 = 0xC0000005) while
# loading, or "CUDA out of memory" although almost all VRAM is free (under
# WDDM every GPU allocation needs system commit). Fix: a larger page file.

# host RAM (GiB) the bf16 weights need when kept on the CPU (offload modes)
HOST_GIB = {"sdxl": 8, "flux-fill": 34, "kontext": 34, "qwen-edit": 42, "qwen-image": 58, "qwen-edit-inpaint": 58, "kandinsky": 8}


def host_memory_gib() -> dict | None:
    """{'ram_total', 'ram_free', 'commit_limit', 'commit_free'} in GiB, or None
    if unknown. commit_* = RAM + page file (Windows only)."""
    import sys

    if sys.platform == "win32":
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        g = 2**30
        return {"ram_total": m.ullTotalPhys / g, "ram_free": m.ullAvailPhys / g,
                "commit_limit": m.ullTotalPageFile / g, "commit_free": m.ullAvailPageFile / g}
    try:
        import psutil
    except ImportError:
        return None
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    g = 2**30
    return {"ram_total": vm.total / g, "ram_free": vm.available / g,
            "commit_limit": (vm.total + sw.total) / g, "commit_free": (vm.available + sw.free) / g}


def check_host_memory(editor: str, offloaded: bool, nf4: bool = False, margin_gib: float = 8.0) -> bool:
    """Warn before loading if RAM + page file cannot hold the offloaded
    weights. Returns False if a warning was printed."""
    if not offloaded or editor not in HOST_GIB:
        return True
    need = HOST_GIB[editor] * (0.45 if nf4 and editor != "sdxl" else 1.0) + margin_gib
    mem = host_memory_gib()
    if mem is None:
        return True
    print(f"[mem] RAM {mem['ram_free']:.0f}/{mem['ram_total']:.0f} GiB free, "
          f"RAM+pagefile {mem['commit_free']:.0f}/{mem['commit_limit']:.0f} GiB free; {editor} needs ~{need:.0f} GiB")
    if mem["commit_free"] < need:
        print(f"[mem] WARNING: not enough RAM + page file for {editor} (need ~{need:.0f} GiB, "
              f"{mem['commit_free']:.0f} free). Expect a crash (exit -1073741819) or 'CUDA out of memory' with free VRAM.\n"
              f"      Fix: enlarge the Windows page file on an SSD (System > Advanced system settings > Performance > "
              f"Advanced > Virtual memory, e.g. 64 GB), reboot, and close other programs.")
        return False
    if mem["ram_total"] < HOST_GIB[editor] * (0.45 if nf4 and editor != "sdxl" else 1.0):
        print(f"[mem] note: weights exceed physical RAM; the page file will be used (slower, but results are identical)")
    return True


def explain_oom(exc: BaseException) -> None:
    """If CUDA reports OOM while most VRAM is free, the cause is host memory."""
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return
    if free > 0.5 * total:
        mem = host_memory_gib() or {}
        print(f"[mem] 'CUDA out of memory' with {free / 2**30:.1f} of {total / 2**30:.1f} GiB VRAM free: this is not VRAM.\n"
              f"      On Windows it means RAM + page file is exhausted (free now: {mem.get('commit_free', float('nan')):.0f} GiB).\n"
              f"      Enlarge the page file (e.g. 64 GB on an SSD), reboot, close other programs, and rerun (the build resumes).")
