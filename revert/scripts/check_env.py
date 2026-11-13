"""Environment pre-flight check. Run this FIRST on a new machine:

    python scripts/check_env.py

It catches the problems that otherwise show up as confusing tracebacks deep
inside transformers, e.g. "operator torchvision::nms does not exist", which
means torch and torchvision come from different builds.
"""

import importlib
import os
import platform
import shutil
import sys

OK, WARN, BAD = "[ ok ]", "[warn]", "[FAIL]"
problems = []


def line(tag, msg):
    print(f"{tag} {msg}")
    if tag == BAD:
        problems.append(msg)


def ram_gib():
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    if platform.system() == "Windows":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / 2**30
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return float("nan")


def version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"IMPORT ERROR: {e.__class__.__name__}: {str(e).splitlines()[0][:160]}"


def main():
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()} ({sys.executable})")
    if sys.version_info >= (3, 13):
        line(WARN, "Python >= 3.13: several GPU packages (bitsandbytes, flash-attn, deepspeed) lag behind on new "
                   "Python versions. Recommended: a separate Python 3.11 or 3.12 environment for this project.")
    elif sys.version_info < (3, 10):
        line(BAD, "Python < 3.10 is not supported")
    else:
        line(OK, "Python version")

    r = ram_gib()
    line(OK if r >= 32 else WARN, f"system RAM {r:.0f} GiB (>= 32 recommended; >= 64 for bf16 FLUX with offload)")
    free = shutil.disk_usage(os.getcwd()).free / 2**30
    line(OK if free >= 200 else WARN, f"free disk here {free:.0f} GiB (models ~60 GB + data ~100 GB)")

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        line(BAD, f"cannot import torch: {e}")
        return finish()
    line(OK, f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    if torch.version.cuda is None:
        line(BAD, "this torch is a CPU-only build; reinstall the CUDA build (see README, Windows setup)")
    if not torch.cuda.is_available():
        line(BAD, "torch.cuda.is_available() is False (driver / CUDA build problem)")
    else:
        p = torch.cuda.get_device_properties(0)
        line(OK, f"GPU {p.name}, {p.total_memory / 2**30:.1f} GiB, bf16 supported={torch.cuda.is_bf16_supported()}")
        free_b, _ = torch.cuda.mem_get_info()
        line(OK if free_b / 2**30 > p.total_memory / 2**30 - 1.5 else WARN,
             f"GPU memory free now {free_b / 2**30:.1f} GiB (the desktop / other programs use the rest)")

    tv = version("torchvision")
    if tv.startswith("IMPORT ERROR"):
        line(BAD, f"torchvision: {tv}\n       -> torch and torchvision are from different builds. Uninstall BOTH and "
                  "reinstall them together in ONE pip command from the same PyTorch index (README, Windows setup).")
    else:
        try:
            import torchvision

            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
            scores = torch.tensor([0.9, 0.8])
            torchvision.ops.nms(boxes, scores, 0.5)
            if torch.cuda.is_available():
                torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5)
            line(OK, f"torchvision {tv} (compiled ops work, matches torch)")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"torchvision {tv} ops fail: {str(e).splitlines()[0][:160]}\n       -> torch/torchvision "
                      "mismatch: reinstall both together from the same PyTorch index (README, Windows setup).")

    for mod, need in [("transformers", True), ("accelerate", True), ("diffusers", False), ("bitsandbytes", False),
                      ("numpy", True), ("cv2", True), ("PIL", True), ("sklearn", False), ("pycocotools", False)]:
        v = version(mod)
        bad = v.startswith("IMPORT ERROR")
        line(BAD if (bad and need) else (WARN if bad else OK), f"{mod} {v}")

    if not problems:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401

            line(OK, "transformers can import AutoProcessor / AutoModelForImageTextToText")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"transformers import failed: {str(e).splitlines()[0][:200]}")
    return finish()


def finish():
    print()
    if problems:
        print(f"{len(problems)} problem(s) to fix before running the audit.")
        sys.exit(1)
    print("Environment looks good.")


if __name__ == "__main__":
    main()
