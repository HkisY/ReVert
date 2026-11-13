"""Command line entry point:  python -m revert_audit.cli <command> ...

Commands
  synth     build the synthetic sanity dataset (CPU, seconds)
  stage1    write original-image queries for a samples.jsonl
  run       run a built-in detector on a query file (resumable)
  stage2    write intervention queries from stage-1 answers
  analyze   compute CNG / CSG / GP / PER / DR with bootstrap CIs
  pilot     progressive run 50 -> 150 -> 300 with fixed stop rules (used for all audits)
  gates     evaluate the gates on existing results
  report    write the Chinese HTML report for existing results
  demo      synth + full pipeline with every mock detector (CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import pipeline


def _detector(args):
    from .detectors import build_detector
    from .gpu import cap_vram

    cap_vram(getattr(args, "vram_cap", None))

    kw = {}
    if args.detector.startswith("qwen"):
        kw = {"model_id": args.model, "coord_mode": args.coord_mode, "max_new_tokens": args.max_new_tokens,
              "max_pixels": args.max_pixels, "load_in_4bit": args.load_in_4bit,
              "gpu_mem_gib": args.gpu_mem, "cpu_mem_gib": args.cpu_mem, "offload_folder": args.offload_folder,
              "offload_kv": args.offload_kv}
        if args.attn:
            kw["attn_implementation"] = args.attn
    elif args.detector in ("sida", "sida-7b"):
        missing = [f for f in ("sida_python", "sida_repo", "sida_weights") if not getattr(args, f)]
        if missing:
            raise SystemExit("--detector sida needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
        kw = {"python": args.sida_python, "repo": args.sida_repo, "weights": args.sida_weights, "clip": args.clip_path,
              "gpu_mem": args.gpu_mem if args.gpu_mem is not None else 12.0, "max_new_tokens": args.max_new_tokens,
              "protocol": args.sida_protocol}
        if args.mask_dir:
            kw["mask_dir"] = args.mask_dir
    elif args.detector in ("sofake", "so-fake-r1", "sofake-r1"):
        missing = [f for f in ("sofake_repo", "sofake_weights", "qwen_base") if not getattr(args, f)]
        if missing:
            raise SystemExit("--detector sofake needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
        kw = {"repo": args.sofake_repo, "weights": args.sofake_weights, "qwen_base": args.qwen_base,
              "gpu_mem": args.gpu_mem if args.gpu_mem is not None else 11.0,
              "cpu_mem": args.cpu_mem if args.cpu_mem is not None else 40.0}
        if args.sofake_cache:
            kw["cache"] = args.sofake_cache
        kw["gen_batch"] = args.gen_batch
    elif args.detector in ("fakeshield", "fakeshield-v1-22b"):
        missing = [f for f in ("fs_dte_python", "fs_mflm_python", "fs_repo", "fs_weights") if not getattr(args, f)]
        if missing:
            raise SystemExit("--detector fakeshield needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
        kw = {"dte_python": args.fs_dte_python, "mflm_python": args.fs_mflm_python, "repo": args.fs_repo,
              "weights": args.fs_weights, "clip": args.clip_path,
              "gpu_mem": args.gpu_mem if args.gpu_mem is not None else 11.0,
              "cpu_mem": args.cpu_mem if args.cpu_mem is not None else 40.0,
              "mflm_gpu_mem": args.fs_mflm_gpu_mem, "cache": args.fs_cache, "load_8bit": args.fs_8bit}
        if args.mask_dir:
            kw["mask_dir"] = args.mask_dir
    elif args.detector == "clip-probe":
        kw = {"probe_path": args.probe}
    return build_detector(args.detector, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="revert_audit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("stage1")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)

    def add_detector_args(p):
        p.add_argument("--detector", required=True, help="qwen-vl | sida | sofake | fakeshield | clip-probe | mock-local | mock-cheater | mock-random | mock-fixed")
        p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
        p.add_argument("--coord-mode", default="auto", choices=["auto", "abs", "rel1000"])
        p.add_argument("--max-new-tokens", type=int, default=512)
        p.add_argument("--attn", default=None, help="e.g. flash_attention_2 (or sdpa)")
        p.add_argument("--max-pixels", type=int, default=None,
                       help="cap model input pixels (e.g. 802816 = 1024*28*28) to save memory")
        p.add_argument("--load-in-4bit", action="store_true",
                       help="NF4 weights; changes the audited model - only for models that do not fit in bf16")
        p.add_argument("--gpu-mem", type=float, default=None,
                       help="GiB of bf16 weights to keep on the GPU; the rest is streamed from CPU RAM")
        p.add_argument("--cpu-mem", type=float, default=None, help="GiB of CPU RAM allowed for offloaded weights")
        p.add_argument("--offload-folder", default=None, help="spill to disk if CPU RAM is too small (very slow)")
        p.add_argument("--offload-kv", action="store_true", help="KV cache in CPU RAM -> larger batches (experimental)")
        p.add_argument("--probe", default=None)
        p.add_argument("--sida-python", default=None, help="python.exe of the SIDA conda env")
        p.add_argument("--sida-repo", default=None, help="folder of the SIDA git repository")
        p.add_argument("--sida-weights", default=None, help="folder of saberzl/SIDA-7B")
        p.add_argument("--sida-protocol", default="paper", choices=["paper", "readme"],
                       help="paper = SIDA test.py protocol (default); readme = chat.py with the README prompt")
        p.add_argument("--clip-path", default=None, help="CLIP vision tower for SIDA (default openai/clip-vit-large-patch14)")
        p.add_argument("--mask-dir", default=None, help="where SIDA's predicted masks are stored (default runs/sida_masks)")
        p.add_argument("--sofake-repo", default=None, help="So-Fake repository folder (contains sofake_r1/runtime)")
        p.add_argument("--sofake-weights", default=None, help="folder of saberzl/So-Fake-R1")
        p.add_argument("--qwen-base", default=None, help="local Qwen2.5-VL-7B-Instruct folder (So-Fake-R1 base)")
        p.add_argument("--sofake-cache", default=None, help="answer cache jsonl (default runs/sofake_cache.jsonl)")
        p.add_argument("--gen-batch", type=int, default=8, help="So-Fake-R1: images per generate() call")
        p.add_argument("--fs-dte-python", default=None, help="python.exe of the FakeShield DTE-FDM env (fsdte)")
        p.add_argument("--fs-mflm-python", default=None, help="python.exe of the FakeShield MFLM env (fsmflm)")
        p.add_argument("--fs-repo", default=None, help="FakeShield repository folder")
        p.add_argument("--fs-weights", default=None, help="fakeshield-v1-22b folder")
        p.add_argument("--fs-mflm-gpu-mem", type=float, default=9.5)
        p.add_argument("--fs-cache", default=None, help="DTE-FDM answer cache jsonl")
        p.add_argument("--fs-8bit", action="store_true", help="DTE-FDM with the release's --load-8bit")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--vram-cap", type=float, default=None,
                       help="cap PyTorch at this fraction of VRAM (e.g. 0.92) so Windows cannot silently spill to RAM")

    p = sub.add_parser("run")
    p.add_argument("--queries", required=True)
    p.add_argument("--out", required=True)
    add_detector_args(p)

    p = sub.add_parser("stage2")
    p.add_argument("--samples", required=True)
    p.add_argument("--s1-dir", required=True)
    p.add_argument("--s1-answers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--explainer", default="model", choices=["model", "oracle", "random"])
    p.add_argument("--n-controls", type=int, default=5)
    p.add_argument("--feather", type=int, default=0)
    p.add_argument("--bases", default="fake,raw")
    p.add_argument("--all-verdicts", action="store_true", help="also intervene when the model said 'real'")

    p = sub.add_parser("analyze")
    p.add_argument("--samples", required=True)
    p.add_argument("--s1-answers", required=True)
    p.add_argument("--s2-dir", required=True)
    p.add_argument("--s2-answers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--oracle-citation", action="store_true", help="DR for detectors without regions: assume C = M")

    p = sub.add_parser("pilot")
    p.add_argument("--samples", required=True)
    p.add_argument("--run", required=True, help="output folder, e.g. runs/pilot")
    p.add_argument("--tag", default=None, help="short name of the detector, used in file names")
    p.add_argument("--gates", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "gates.json"))
    p.add_argument("--steps", default=None, help="e.g. 50,150,300 (default: from the gates file)")
    p.add_argument("--n-controls", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--s2-bases", default="fake,raw",
                   help="bases that get stage-2 interventions; 'fake' alone saves time for slow detectors")
    add_detector_args(p)

    p = sub.add_parser("seam", help="E4: rerun the model's stage-2 queries with soft/grown seams")
    p.add_argument("--samples", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ramps", default="8,24", help="ramp widths N (each with dilate:N/2)")
    p.add_argument("--n-controls", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)
    add_detector_args(p)

    p = sub.add_parser("component", help="E4b: seam-free whole-object reversion on two-object samples")
    p.add_argument("--samples", required=True, help="samples.jsonl containing two-object samples (data.build --pairs)")
    p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ref-run", default=None, help="single-object pilot run of the same detector (oracle PER reference)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    add_detector_args(p)


    for name in ("gates", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--samples", required=True)
        p.add_argument("--tag", required=True)
        p.add_argument("--gates", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "gates.json"))

    p = sub.add_parser("demo")
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--n", type=int, default=60)

    a = ap.parse_args(argv)

    if a.cmd == "synth":
        from .synthetic import build_synthetic_dataset

        print(build_synthetic_dataset(a.out, a.n, a.seed))
    elif a.cmd == "stage1":
        print(pipeline.stage1(a.samples, a.out))
    elif a.cmd == "run":
        print(pipeline.run_detector(a.queries, _detector(a), a.out, a.batch_size))
    elif a.cmd == "stage2":
        print(
            pipeline.stage2(
                a.samples, a.s1_dir, a.s1_answers, a.out, explainer=a.explainer, n_controls=a.n_controls,
                bases=tuple(a.bases.split(",")), feather_px=a.feather, require_fake=not a.all_verdicts,
            )
        )
    elif a.cmd == "analyze":
        res = pipeline.analyze(a.samples, a.s1_answers, os.path.join(a.s2_dir, "meta.jsonl"), a.s2_answers, a.out, a.n_boot, oracle_citation=a.oracle_citation)
        print(pipeline.format_table(res))
    elif a.cmd == "pilot":
        from .pilot import run_pilot

        tag = a.tag or (a.model.split("/")[-1] if a.detector.startswith("qwen") else a.detector)
        steps = [int(x) for x in a.steps.split(",")] if a.steps else None
        run_pilot(a.samples, a.run, _detector(a), tag, a.gates, steps, n_controls=a.n_controls,
                  batch_size=a.batch_size, n_boot=a.n_boot, s2_bases=tuple(a.s2_bases.split(",")))
    elif a.cmd == "seam":
        from .seam import print_report, run_seam

        out = run_seam(a.samples, a.run, _detector(a), a.tag, [int(x) for x in a.ramps.split(",")],
                       n_controls=a.n_controls, batch_size=a.batch_size, n_boot=a.n_boot, limit=a.limit)
        print_report(out)
    elif a.cmd == "component":
        from .component import print_report as cprint, run_component

        res = run_component(a.samples, a.run, _detector(a), a.tag, n_boot=a.n_boot, batch_size=a.batch_size,
                            limit=a.limit, ref_run=a.ref_run)
        cprint(res, a.tag)
    elif a.cmd in ("gates", "report"):
        from .gates import config_hash, evaluate, load_config, load_results
        from .report import build_report

        r = evaluate(load_results(os.path.join(a.run, "results"), a.tag), load_config(a.gates))
        print(json.dumps(r, indent=2, ensure_ascii=False))
        if a.cmd == "report":
            print(build_report(a.run, a.samples, a.tag, r, config_hash(a.gates)))
    elif a.cmd == "demo":
        run_demo(a.out, a.n)


def run_demo(out: str, n: int = 60, explainers=("model", "oracle", "random"), n_boot: int = 300) -> dict:
    """Full pipeline on the synthetic world for every mock detector."""
    from .detectors import build_detector
    from .synthetic import MOCKS, build_synthetic_dataset

    samples = build_synthetic_dataset(os.path.join(out, "data"), n)
    s1 = os.path.join(out, "stage1")
    q1 = pipeline.stage1(samples, s1)
    summary = {}
    for name in MOCKS:
        det = build_detector(name)
        a1 = pipeline.run_detector(q1, det, os.path.join(s1, f"answers_{name}.jsonl"), batch_size=16, log_every=0)
        for ex in explainers:
            s2 = os.path.join(out, f"stage2_{name}_{ex}")
            q2 = pipeline.stage2(samples, s1, a1, s2, explainer=ex, n_controls=3)
            a2 = pipeline.run_detector(q2, det, os.path.join(s2, "answers.jsonl"), batch_size=16, log_every=0)
            res = pipeline.analyze(samples, a1, os.path.join(s2, "meta.jsonl"), a2, os.path.join(out, "results", f"{name}_{ex}"), n_boot)
            summary[f"{name}/{ex}"] = {
                "acc_fake": res["stage1"].get("acc_fake"),
                "acc_raw": res["stage1"].get("acc_raw"),
                "acc_real": res["stage1"].get("acc_real"),
                **{f"{b}_{k}": round(v[k][0], 3) if isinstance(v[k], (list, tuple)) else v[k] for b, v in res["by_base"].items() for k in ("n", "CNG", "CSG", "GP", "PER")},
                "DR": res["DR"][0],
                "DR_n": res["DR_n_eligible"],
            }
    with open(os.path.join(out, "demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(k, v)
    return summary


if __name__ == "__main__":
    main()
