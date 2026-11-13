"""Project compute budget from MEASURED per-image times.

The workload is fixed by the experimental design (it must not shrink when the
GPU gets smaller); only the per-call times depend on the hardware. Feed the
times printed by scripts/bench_gpu.py into `estimate()`.

Per (sample, detector) in the audit:
    explain calls  = 3 (stage 1: x_f, x_raw, x_r) + 2 (R(x,C) re-explained, for PER)
    score-only     = 2 bases x (1 keep(C) + 2 x n_ctrl controls)
"""

from __future__ import annotations

from dataclasses import dataclass, field


def audit_calls(n_ctrl: int = 3) -> tuple[int, int]:
    return 5, 2 * (1 + 2 * n_ctrl)


@dataclass
class Design:
    """Experimental scale (the 4090 plan; restoring the original A100 plan
    means n_test=6000, n_models=12, n_ctrl=5)."""
    n_test: int = 4000
    n_models: int = 8
    n_ctrl: int = 3
    n_pilot: int = 300
    pilot_explainers: int = 3  # model / oracle / random
    gen_images: dict = field(default_factory=lambda: {"sdxl": 9000, "flux-fill": 1000, "kontext": 1400})
    rvpo_runs: int = 3          # seeds of the proposed method
    baseline_runs: int = 4      # SFT, SFT+INP-X, mask-SFT, twin-SFT (1 seed each)
    train_samples: int = 8000
    dpo_pairs: int = 6000
    dpo_epochs: int = 2
    rvpo_samples_per_image: int = 4
    rvpo_sample_images: int = 4000
    misc_hours: float = 20.0    # attacks, video appendix, re-runs of small pieces


def estimate(t_explain: float, t_score: float, gen_sec: dict, d: Design = Design(),
             train_factor: float = 3.0) -> dict:
    """All times in seconds per image at the best measured batch size.

    Training is NOT measured by the benchmark; it is approximated as
    `train_factor` x a forward pass per sequence (forward + backward ~ 3x
    forward), two sequences per DPO pair. Treat it as the roughest line.
    """
    ne, ns = audit_calls(d.n_ctrl)
    per_sm = ne * t_explain + ns * t_score
    audit = d.n_test * d.n_models * per_sm / 3600
    pilot = d.n_pilot * (3 * t_explain + d.pilot_explainers * (2 * t_explain + ns * t_score)) / 3600
    gen = sum(n * gen_sec.get(k, float("nan")) for k, n in d.gen_images.items()) / 3600
    sft = d.train_samples * train_factor * t_score / 3600
    sampling = d.rvpo_sample_images * (d.rvpo_samples_per_image * t_explain + d.rvpo_samples_per_image * t_score) / 3600
    dpo = d.dpo_pairs * d.dpo_epochs * 2 * train_factor * t_score / 3600
    train = d.rvpo_runs * (sft + sampling + dpo) + d.baseline_runs * sft
    total = pilot + gen + audit + train + d.misc_hours
    return {
        "per_sample_model_sec": per_sm,
        "pilot_h": pilot,
        "data_generation_h": gen,
        "audit_h": audit,
        "training_h": train,
        "misc_h": d.misc_hours,
        "total_h": total,
        "total_days_24_7": total / 24,
    }


def format_estimate(e: dict) -> str:
    rows = [
        ("pilot (1 model, 3 explainers)", e["pilot_h"]),
        ("data generation", e["data_generation_h"]),
        ("audit (all models)", e["audit_h"]),
        ("training (RVPO + baselines, rough)", e["training_h"]),
        ("misc", e["misc_h"]),
        ("TOTAL", e["total_h"]),
    ]
    lines = [f"per (sample, model): {e['per_sample_model_sec']:.1f} s", "", "| item | GPU hours |", "| --- | --- |"]
    lines += [f"| {k} | {v:.0f} |" for k, v in rows]
    lines.append(f"\n= {e['total_days_24_7']:.0f} days of 24/7 running")
    return "\n".join(lines)
