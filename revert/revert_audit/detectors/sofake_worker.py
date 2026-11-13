"""So-Fake-R1 (inference release v5.64) as a persistent worker process.

Runs in the 'audit' env (transformers + peft; no ms-swift). Started by
revert_audit.detectors.sofake.SoFakeDetector; talks over stdin/stdout lines
prefixed with PROTO, like the SIDA worker.

What is reproduced, and how
  The official release (sofake_r1/scripts/infer_one_image.sh) chains 9 steps
  and reloads the 7B model several times per image. Here the Qwen2.5-VL-7B
  base is loaded ONCE with both LoRA adapters attached (peft), and every step
  runs on a whole batch of images:

    1 observer input      official builder (make-observer-input prompt)
    2 Visual Observer     Qwen2.5-VL + observer LoRA, greedy, 384 new tokens   <- re-implemented
    3 L19 features        the SAME weights with the adapters disabled, official
                          m0 hooks / processor settings (max_pixels 224*224)
    4 evidence provider   official m60 main()          (in-process, unchanged)
    5 provider assets     official v544 main()         (in-process, unchanged)
    6 evidence gate       official v564 builder main() (in-process, unchanged)
    7 Decision Maker      Qwen2.5-VL + DM LoRA, greedy, 512 new tokens,
                          max_pixels 1048576 (only routed rows)                  <- re-implemented
    8 merge               official merge main()
    9 export              official export main()

  Steps 2 and 7 replace `swift infer`. What ms-swift does there and is copied:
  default system prompt 'You are a helpful assistant.', '<image>' replaced by
  <|vision_start|><|image_pad|><|vision_end|>, image resized with
  qwen_vl_utils' smart_resize + PIL bicubic and passed with do_resize=False,
  greedy decoding with the model's generation_config (repetition_penalty
  1.05), special tokens kept in the decoded text (<|box_start|> ...).
  --selftest on the release's examples/example.jpg checks the documented
  result (observer FULL_SYNTHETIC, provider REAL, TARGETED_RECHECK, final REAL).

Coordinates
  Provider boxes are explicitly written on a 224x224 frame (grid_bbox_to_224)
  and the Decision Maker's GRPO reward clamps boxes to 0..224 and resizes the
  GT mask to 224x224, so the final box is read on a 224x224 frame and scaled
  to the image. (The release's own export draws it in raw pixels, which is
  only right for 224x224 images.) The raw box text is kept in every answer.

Score
  So-Fake-R1 has no end-to-end probability (lazy-gate rows never reach a VLM),
  so the audit score is its final decision (fake = FULL_SYNTHETIC or TAMPERED).
  The provider's class probabilities are recorded as p_fake_provider.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback

PROTO = "@@REVERT@@"
FRAME = 224
OBS_MAX_NEW, DM_MAX_NEW = 384, 512
OBS_MAX_PIXELS = 16384 * 28 * 28  # qwen_vl_utils default (swift infer without --max_pixels)
DM_MAX_PIXELS = 1048576  # infer_one_image.sh: --max_pixels 1048576
MIN_PIXELS = 4 * 28 * 28
FEAT_MAX_PIXELS = 224 * 224
SYSTEM = "You are a helpful assistant."
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
GATE_TAG = "v5_60_prompt_clean_phrase_gate"
EXAMPLE_EXPECT = {"observer": "FULL_SYNTHETIC", "provider": "REAL", "route": "TARGETED_RECHECK", "final": "REAL"}
CACHE_VERSION = "sofake-r1-v5.64-w1"
DEVICE = os.environ.get("SOFAKE_DEVICE", "cuda")  # provider heads; "cpu" only for tests


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(obj):
    sys.stdout.write(PROTO + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------- qwen_vl_utils
def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = OBS_MAX_PIXELS) -> tuple[int, int]:
    """qwen_vl_utils.smart_resize (the resize ms-swift applies before the processor)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too large: {height}x{width}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def fetch_image(path: str, max_pixels: int):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    h, w = smart_resize(img.height, img.width, max_pixels=max_pixels)
    return img.resize((w, h), Image.BICUBIC)


def chat_prompt(user_content: str) -> str:
    if "<image>" not in user_content:
        raise ValueError("prompt without <image> tag")
    query = user_content.replace("<image>", VISION, 1)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n")


def strip_stop(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text[: text.index(stop)]
    return text


def box_to_image(box, w: int, h: int, frame: int = FRAME):
    """[x1,y1,x2,y2] on a frame x frame canvas -> original image pixels."""
    if not box:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    sx, sy = w / float(frame), h / float(frame)
    out = [max(0.0, x1 * sx), max(0.0, y1 * sy), min(float(w), x2 * sx), min(float(h), y2 * sy)]
    return out if out[2] > out[0] and out[3] > out[1] else None


# ------------------------------------------------------------------- runtime
class Runtime:
    """Imports the release's own modules from sofake_r1/runtime."""

    SUBDIRS = ["tools", "tools/v544_grpo_provider_bbox_complete_20260706",
               "tools/v547_external_provider_interface_repair_20260706",
               "tools/v551_unified_vlm_recheck_sft_2465_20260707",
               "tools/v564_new_provider_external_20260711", "plugins"]

    def __init__(self, repo: str):
        cands = [repo, os.path.join(repo, "runtime"), os.path.join(repo, "sofake_r1", "runtime"),
                 os.path.join(repo, "So-Fake-main", "sofake_r1", "runtime")]
        root = next((c for c in cands if os.path.isdir(os.path.join(c, "tools"))), None)
        if root is None:
            raise FileNotFoundError(f"sofake_r1/runtime not found under {repo}")
        self.root = os.path.abspath(root)
        os.environ.setdefault("SOFAKE_R1_RELEASE", self.root)
        for sub in reversed(self.SUBDIRS):
            p = os.path.join(self.root, *sub.split("/"))
            if p not in sys.path:
                sys.path.insert(0, p)
        try:  # matplotlib >= 3.9 removed cm.get_cmap; the release uses it for the topology PNG only
            import matplotlib
            import matplotlib.cm as mcm

            if not hasattr(mcm, "get_cmap"):
                mcm.get_cmap = lambda name=None, lut=None: matplotlib.colormaps[name or "viridis"]
        except ImportError:
            pass
        self.m0 = importlib.import_module("m0_extract_features")
        self.m60 = importlib.import_module("m60_export_locked_evidence_provider")
        self.v544 = importlib.import_module("export_locked_provider_v5_7_assets_from_h5_reference")
        self.builder = importlib.import_module("build_v564_new_provider_external_gate")
        self.merge = importlib.import_module("merge_v551_gate_aware_infer")
        self.export = importlib.import_module("export_sofake_r1_infer_outputs")

    def run_main(self, mod, argv: list[str]):
        """Call an official script's main() with argv; its prints go to stderr."""
        old = sys.argv
        sys.argv = [mod.__file__] + [str(a) for a in argv]
        try:
            with contextlib.redirect_stdout(sys.stderr):
                mod.main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"{os.path.basename(mod.__file__)} exited: {e.code}") from None
        finally:
            sys.argv = old


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- model
def load(args) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    try:  # same as the release's m0: avoid the large warm-up allocation
        import transformers.modeling_utils as mu

        mu.caching_allocator_warmup = lambda *a, **k: None
    except Exception:  # noqa: BLE001
        pass
    rt = Runtime(args.repo)
    w = args.weights
    need = [os.path.join(w, "observer_adapter", "adapter_model.safetensors"),
            os.path.join(w, "decision_maker_adapter", "adapter_model.safetensors")] + \
           [os.path.join(w, "provider", f) for f in ("npr_internalized_feta_mono_blur_only05_e1.pt",
                                                     "feta_v0_L19_cls_only10d_L19_u1_e1.pt",
                                                     "npr_feta_cls_npr_residual_feta10d_e1.pt",
                                                     "l19_heatmap_bbox_head_best.pt")]
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("missing So-Fake-R1 weights: " + ", ".join(miss))
    kw = {"dtype": torch.bfloat16}
    if DEVICE != "cuda":
        kw["device_map"] = DEVICE
    elif args.gpu_mem:
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: f"{args.gpu_mem}GiB", "cpu": f"{args.cpu_mem}GiB"}
    else:
        kw["device_map"] = "cuda"
    if DEVICE == "cuda" and args.vram_cap:
        # Windows silently spills CUDA allocations into shared RAM (very slow) when the card is full;
        # capping PyTorch makes it raise out-of-memory instead, and batches are then split
        torch.cuda.set_per_process_memory_fraction(args.vram_cap, 0)
    t0 = time.time()
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_base, **kw).eval()
    model = PeftModel.from_pretrained(base, os.path.join(w, "observer_adapter"), adapter_name="observer")
    model.load_adapter(os.path.join(w, "decision_maker_adapter"), adapter_name="dm")
    model.eval()
    moved = place_lora_on_gpu(model, "cuda:0" if DEVICE == "cuda" else DEVICE)
    if moved:
        log(f"[sofake] moved {moved} LoRA tensors of offloaded layers to the GPU ({moved_mb(model):.0f} MB of LoRA total)")
    proc = AutoProcessor.from_pretrained(args.qwen_base)
    # the release's feature processor (m0.load_model): separate instance, 224*224 cap
    fproc = AutoProcessor.from_pretrained(args.qwen_base)
    fproc.image_processor.max_pixels = FEAT_MAX_PIXELS
    fproc.image_processor.size = {"longest_edge": FEAT_MAX_PIXELS, "shortest_edge": max(28 * 28, FEAT_MAX_PIXELS // 16)}
    dmap = getattr(base, "hf_device_map", {}) or {}
    n_cpu = sum(1 for d in dmap.values() if str(d) in ("cpu", "disk"))
    gpu = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    log(f"[sofake] loaded in {time.time() - t0:.0f}s; modules offloaded to CPU: {n_cpu}; GPU {gpu:.1f} GiB")
    return {"rt": rt, "model": model, "core": model.get_base_model(), "proc": proc, "fproc": fproc,
            "weights": w, "qwen_base": args.qwen_base, "dev": "cuda:0" if DEVICE == "cuda" else DEVICE,
            "offloaded": n_cpu}


def place_lora_on_gpu(model, dev: str) -> int:
    """A second adapter loaded after accelerate dispatched the base model keeps its
    LoRA tensors on the CPU for offloaded layers (the offload hooks only stream the
    tensors they knew at dispatch time). LoRA tensors are small: keep them on the
    GPU. Tensors on 'meta' are managed by the hooks and are left alone."""
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name and p.device.type == "cpu" and dev != "cpu":
            p.data = p.data.to(dev)
            n += 1
    return n


def moved_mb(model) -> float:
    return sum(p.numel() * p.element_size() for n, p in model.named_parameters() if "lora_" in n) / 2**20


def _encode(proc, prompts, images):
    try:
        return proc(text=prompts, images=images, do_resize=False, padding=True, return_tensors="pt")
    except TypeError:
        return proc(text=prompts, images=images, images_kwargs={"do_resize": False}, padding=True,
                    return_tensors="pt")


def generate_batch(S, adapter: str, contents: list[str], image_paths: list[str], max_pixels: int, max_new: int,
                   batch: int = 1) -> list[str]:
    """Greedy generation, `batch` images per call (left padding). Batching only
    amortizes the CPU->GPU weight streaming; the answers are the same as one by
    one up to bf16 kernel differences (check with --selftest-batch)."""
    import torch

    model, proc = S["model"], S["proc"]
    model.set_adapter(adapter)
    proc.tokenizer.padding_side = "left"
    out_texts: list[str] = []

    def run(cs, ps):
        imgs = [fetch_image(p, max_pixels) for p in ps]
        inputs = _encode(proc, [chat_prompt(c) for c in cs], imgs).to(S["dev"])
        # no_grad, not inference_mode: with CPU offload, inference_mode turns the streamed (LoRA) weights into
        # inference tensors, and peft's later enable_adapter_layers() (requires_grad_) then fails
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            if len(cs) == 1:
                raise
            h = len(cs) // 2  # same answers, smaller batch
            log(f"[sofake] {adapter}: batch of {len(cs)} did not fit in VRAM, splitting")
            return run(cs[:h], ps[:h]) + run(cs[h:], ps[h:])
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [strip_stop(proc.tokenizer.decode(row, skip_special_tokens=False)) for row in gen]

    for i in range(0, len(contents), max(1, batch)):
        out_texts += run(contents[i:i + batch], image_paths[i:i + batch])
    return out_texts


def generate(S, adapter: str, user_content: str, image_path: str, max_pixels: int, max_new: int) -> str:
    return generate_batch(S, adapter, [user_content], [image_path], max_pixels, max_new, 1)[0]


def extract_features(S, samples: list[dict], h5_path: str):
    """Official m0 main() loop for --layers 19 --max_pixels 224*224, on the shared
    weights with both adapters disabled (= the frozen base model)."""
    import h5py
    import numpy as np
    import torch
    from PIL import Image

    m0, core = S["rt"].m0, S["core"]
    visual = core.model.visual
    store: dict = {}
    handles = m0.attach_hooks(visual, store, [19])
    try:
        with torch.no_grad(), S["model"].disable_adapter(), h5py.File(h5_path, "w") as h5:
            meta = h5.create_group("meta")
            meta.attrs["model"] = S["qwen_base"]
            meta.attrs["layers"] = np.array([19], dtype=np.int32)
            meta.attrs["hidden_dim"] = 1280
            meta.attrs["max_pixels"] = FEAT_MAX_PIXELS
            for samp in samples:
                img = Image.open(samp["image"]).convert("RGB")
                grid_thw = m0.extract_one(core, S["fproc"], img, store, S["dev"])
                grp = h5.create_group(samp["sample_id"])
                grp.attrs["image_path"] = samp["image"]
                grp.attrs["gt_cls"] = int(samp.get("gt_cls", -1))
                grp.attrs["grid_thw"] = grid_thw.astype(np.int32)
                for k, v in samp.items():
                    if k in ("image", "gt_cls", "sample_id"):
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        grp.attrs[k] = v
                    elif v is None:
                        grp.attrs[k] = ""
                grp.create_dataset("patch_mask", data=m0.compute_patch_mask(None, grid_thw), compression="lzf")
                for layer_idx, feat in store.items():
                    grp.create_dataset(f"layer_{layer_idx}", data=feat.numpy(), compression="lzf")
    finally:
        m0.detach_hooks(handles)


def run_batch(S, items: list[dict], work: str) -> dict[str, dict]:
    """items: [{'key', 'path'}] with unique keys. Returns key -> raw So-Fake-R1 record."""
    rt, w = S["rt"], S["weights"]
    prov = os.path.join(w, "provider")
    os.makedirs(work, exist_ok=True)
    pdir = os.path.join(work, "provider")
    old_gate, gate = os.path.join(work, "v5_51_gate_aware"), os.path.join(work, GATE_TAG)
    map_dir = os.path.join(pdir, "topology_maps", "single")
    for d in (pdir, old_gate, gate, map_dir):
        os.makedirs(d, exist_ok=True)
    f = {k: os.path.join(work, v) for k, v in {
        "samples": "samples.jsonl", "obs_in": "observer_input.jsonl", "obs": "observer_infer.jsonl",
        "h5": "features_L19.h5", "deep_pred": "dm_deep_pred.jsonl", "merged": "merged_prediction.jsonl",
        "final": "sofake_r1_result.jsonl"}.items()}
    locked, compat = os.path.join(pdir, "rerun_locked_provider.jsonl"), os.path.join(pdir, "compatible_evidence.jsonl")
    loc, summ = os.path.join(pdir, "localization.jsonl"), os.path.join(pdir, "provider_export_summary.json")
    manifest = os.path.join(gate, f"judge_val_bfree_{GATE_TAG}_manifest.jsonl")
    deep_in = os.path.join(gate, f"judge_val_bfree_{GATE_TAG}_deep_vlm_infer.jsonl")

    # gt_cls is a dummy (0/REAL), exactly as in infer_one_image.sh; the gate never reads it
    samples = [{"sample_id": f"s_{it['key']}", "image": os.path.realpath(it["path"]), "gt_cls": 0, "label": 0,
                "gt_label_name": "REAL", "source": "single_image", "subsource": "single_image",
                "dataset": "single_image"} for it in items]
    write_jsonl(f["samples"], samples)
    tm = {}

    t = time.time()  # [1] observer input + [2] observer
    rt.run_main(rt.builder, ["make-observer-input", "--samples", f["samples"], "--out", f["obs_in"],
                             "--dataset", "single_image"])
    rows = read_jsonl(f["obs_in"])
    resps = generate_batch(S, "observer", [r["messages"][0]["content"] for r in rows], [r["images"][0] for r in rows],
                           OBS_MAX_PIXELS, OBS_MAX_NEW, S.get("gen_batch", 1))
    obs_rows = [{"response": resp, "images": [{"path": r["images"][0]}], "label": r.get("label"),
                 "sample_id": r.get("sample_id"), "meta": r.get("meta")} for r, resp in zip(rows, resps)]
    write_jsonl(f["obs"], obs_rows)
    tm["observer"] = time.time() - t
    log(f"[sofake] observer: {len(obs_rows)} image(s) in {tm['observer']:.0f}s")

    t = time.time()  # [3] features
    extract_features(S, samples, f["h5"])
    tm["features"] = time.time() - t
    log(f"[sofake] L19 features in {tm['features']:.0f}s")

    t = time.time()  # [4] provider, [5] assets, [6] gate
    rt.run_main(rt.m60, ["--h5", f["h5"], "--ckpt", os.path.join(prov, "npr_internalized_feta_mono_blur_only05_e1.pt"),
                         "--out", locked, "--transform", "log_scaled_a2", "--dataset_name", "single_image",
                         "--qwen_ckpt", os.path.join(prov, "feta_v0_L19_cls_only10d_L19_u1_e1.pt"),
                         "--npr_ckpt", os.path.join(prov, "npr_feta_cls_npr_residual_feta10d_e1.pt"),
                         "--device", DEVICE, "--eval_batch_size", "1", "--batch_size", "1", "--num_workers", "0"])
    rt.run_main(rt.v544, ["--samples", f["samples"], "--h5", f["h5"], "--locked_evidence", locked,
                          "--map_dir", map_dir, "--compatible_evidence_out", compat, "--localization_out", loc,
                          "--summary_out", summ, "--qwen_feta_ckpt", os.path.join(prov, "feta_v0_L19_cls_only10d_L19_u1_e1.pt"),
                          "--prompt_ckpt", os.path.join(prov, "l19_heatmap_bbox_head_best.pt"), "--device", DEVICE])
    rt.run_main(rt.builder, ["build", "--bfree-samples", f["samples"], "--bfree-observer", f["obs"],
                             "--bfree-provider", compat, "--old-gate-dir", old_gate, "--phrase-dir", gate,
                             "--tag", GATE_TAG])
    tm["provider_gate"] = time.time() - t
    log(f"[sofake] provider + gate in {tm['provider_gate']:.0f}s")

    t = time.time()  # [7] decision maker on routed rows
    rows = read_jsonl(deep_in)
    paths = [(r["images"][0]["path"] if isinstance(r["images"][0], dict) else r["images"][0]) for r in rows]
    resps = generate_batch(S, "dm", [r["messages"][0]["content"] for r in rows], paths, DM_MAX_PIXELS, DM_MAX_NEW,
                           S.get("gen_batch", 1))
    deep = [{"response": resp, "images": [{"path": p}], "label": r.get("label"), "meta": r.get("meta")}
            for r, p, resp in zip(rows, paths, resps)]
    write_jsonl(f["deep_pred"], deep)
    tm["decision_maker"] = time.time() - t
    tm["n_deep"] = len(deep)

    t = time.time()  # [8] merge, [9] export
    rt.run_main(rt.merge, ["--manifest", manifest, "--deep-pred", f["deep_pred"], "--out", f["merged"]])
    rt.run_main(rt.export, ["--manifest", manifest, "--merged", f["merged"], "--out", f["final"],
                            "--mask-dir", os.path.join(work, "masks"), "--observer-jsonl", f["obs"],
                            "--provider-compatible", compat, "--provider-localization", loc,
                            "--topology-map-dir", map_dir, "--mask-mode", "box"])
    tm["merge_export"] = time.time() - t

    provider_raw = {r.get("sid"): r for r in read_jsonl(locked)}
    out = {}
    for row in read_jsonl(f["final"]):
        sid = str(row.get("sample_id"))
        row["_provider_raw"] = provider_raw.get(sid, {})
        out[sid[2:] if sid.startswith("s_") else sid] = row
    missing = [it["key"] for it in items if it["key"] not in out]
    if missing:
        raise RuntimeError(f"So-Fake-R1 produced no result for {len(missing)} image(s)")
    out["_timing"] = tm
    return out


def to_answer(rec: dict, path: str) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    final = rec.get("final") or {}
    scope = str(final.get("scope") or "PARSE_FAIL").upper()
    verdict = {"REAL": "real", "FULL_SYNTHETIC": "fake", "TAMPERED": "fake"}.get(scope)
    box224 = final.get("bbox")
    box = box_to_image(box224, w, h) if scope == "TAMPERED" else None
    pr = rec.get("_provider_raw") or {}
    probs = pr.get("probs")
    obs, prov = rec.get("observer") or {}, rec.get("provider") or {}
    dm = rec.get("decision_maker") or {}
    return {
        "verdict": verdict,
        "verdict_native": verdict is not None,
        "p_fake": None,  # no end-to-end probability: the audit score is the final decision
        "p_fake_provider": (1.0 - float(probs[0])) if probs else None,
        "provider_probs": probs,
        "boxes": [box] if box else [],
        "mask_path": None,
        "box224": box224,
        "sofake_scope": scope,
        "route_action": rec.get("route_action"),
        "route_detail": rec.get("route_detail"),
        "output_source": rec.get("output_source"),
        "observer_answer": obs.get("answer"),
        "observer_bbox224": obs.get("bbox"),
        "provider_answer": prov.get("answer"),
        "provider_bbox224": prov.get("bbox"),
        "artifacts": [],
        "raw_text": dm.get("raw_response") or "",
        "observer_text": obs.get("raw_response") or obs.get("report") or "",
        "parse_mode": "native" if verdict else "failed",
    }


class Cache:
    """Answers keyed by a hash of the exact pixels: identical images (many
    interventions are no-ops on x_f) are run once. Greedy decoding makes the
    answer a function of the pixels."""

    def __init__(self, path: str | None):
        self.path, self.d = path, {}
        if path and os.path.exists(path):
            for r in read_jsonl(path):
                if r.get("v") == CACHE_VERSION:
                    self.d[r["key"]] = r["answer"]
            log(f"[sofake] cache: {len(self.d)} answers from {path}")

    def get(self, k):
        return self.d.get(k)

    def put(self, k, ans):
        self.d[k] = ans
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"v": CACHE_VERSION, "key": k, "answer": ans}, ensure_ascii=False) + "\n")


def predict(S, cache: Cache, items: list[dict], keep_work: str | None = None) -> list[dict]:
    todo, seen = [], set()
    for it in items:
        if cache.get(it["key"]) is None and it["key"] not in seen:
            todo.append(it)
            seen.add(it["key"])
    if todo:
        work = tempfile.mkdtemp(prefix="sofake_work_")
        try:
            t0 = time.time()
            recs = run_batch(S, todo, work)
            tm = recs.pop("_timing")
            per = (time.time() - t0) / len(todo)
            for it in todo:
                ans = to_answer(recs[it["key"]], it["path"])
                ans["sec"] = per
                cache.put(it["key"], ans)
            log(f"[sofake] {len(todo)} new image(s) in {time.time() - t0:.0f}s "
                f"({(time.time() - t0) / len(todo):.1f} s/img; DM on {tm['n_deep']}) "
                + " ".join(f"{k}={v:.0f}s" for k, v in tm.items() if k != "n_deep"))
            if keep_work:
                shutil.copytree(work, keep_work, dirs_exist_ok=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    out, fresh = [], {it["key"] for it in todo}
    for it in items:
        a = dict(cache.get(it["key"]))
        if it["key"] in fresh:
            fresh.discard(it["key"])  # GPU time is counted once per distinct image
        else:
            a["sec"] = 0.0
            a["cached"] = True
        out.append(a)
    return out


def serve(S, cache: Cache):
    send({"ready": True, "detector": "so-fake-r1", "version": "v5.64", "offloaded_modules": S["offloaded"]})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            send({"answers": predict(S, cache, req["items"])})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            send({"error": f"{type(e).__name__}: {e}"})


def selftest(S, image: str):
    import hashlib

    key = hashlib.sha1(open(image, "rb").read()).hexdigest()[:20]
    keep = os.path.join(tempfile.gettempdir(), "sofake_selftest_work")
    t0 = time.time()
    ans = predict(S, Cache(None), [{"key": key, "path": image}], keep_work=keep)[0]
    log(f"[selftest] {time.time() - t0:.1f}s; intermediate files kept in {keep}")
    log(json.dumps({k: ans[k] for k in ("observer_answer", "provider_answer", "route_action", "output_source",
                                         "sofake_scope", "verdict", "box224", "boxes", "p_fake_provider")},
                   ensure_ascii=False, indent=2))
    log("[selftest] observer text: " + ans["observer_text"][:600].replace("\n", " "))
    log("[selftest] decision maker text: " + (ans["raw_text"][:600].replace("\n", " ") or "(lazy gate, no DM call)"))
    if os.path.basename(image).lower() == "example.jpg":
        got = {"observer": ans["observer_answer"], "provider": ans["provider_answer"],
               "route": ans["route_action"], "final": ans["sofake_scope"]}
        bad = {k: (got[k], v) for k, v in EXAMPLE_EXPECT.items() if got[k] != v}
        if bad:
            log("[selftest] MISMATCH with the release's documented example result (got, expected): " + json.dumps(bad))
            return 2
        log("[selftest] PASS: matches the release's documented example result")
    return 0


def selftest_batch(S, folder: str, batch: int, n: int | None = None):
    """Same images, observer + decision maker one by one vs `batch` at a time."""
    import hashlib

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    paths = sorted(os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(exts))[:n or batch]
    if not paths:
        log(f"[selftest-batch] no images in {folder}")
        return 2
    items = [{"key": hashlib.sha1(open(p, "rb").read()).hexdigest()[:20], "path": p} for p in paths]
    res = {}
    for b in (1, batch):
        S["gen_batch"] = b
        t0 = time.time()
        res[b] = predict(S, Cache(None), items)
        log(f"[selftest-batch] batch={b}: {len(items)} images in {time.time() - t0:.0f}s "
            f"({(time.time() - t0) / len(items):.1f} s/img)")
    keys = ("sofake_scope", "box224", "route_action", "observer_answer", "provider_answer")
    same_dec = sum(res[1][i]["sofake_scope"] == res[batch][i]["sofake_scope"] for i in range(len(items)))
    same_all = sum(all(res[1][i][k] == res[batch][i][k] for k in keys) for i in range(len(items)))
    same_txt = sum(res[1][i]["observer_text"] == res[batch][i]["observer_text"] for i in range(len(items)))
    for i, p in enumerate(paths):
        a, b_ = res[1][i], res[batch][i]
        log(f"  {os.path.basename(p)}: one-by-one {a['sofake_scope']} {a['box224']} | batch {b_['sofake_scope']} {b_['box224']}")
    log(f"[selftest-batch] same decision {same_dec}/{len(items)}, same decision+box+route {same_all}/{len(items)}, "
        f"identical observer text {same_txt}/{len(items)}")
    return 0


def pixel_key(path: str) -> str:
    """Same key as WorkerDetector.key (hash of the RGB pixels), so probe answers
    are reused by the pilot through the cache."""
    import hashlib

    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()[:20]


def probe(S, samples_path: str, n_fake: int, n_real: int, cache: Cache):
    """Does So-Fake-R1 flag our edits at all? First n_fake x_f and n_real x_real
    of the samples file, in one batched run; per-image table + rates."""
    rows = read_jsonl(samples_path)
    items = [{"kind": "fake", "path": r["fake"], "sid": r.get("id")} for r in rows[:n_fake]] + \
            [{"kind": "real", "path": r["real"], "sid": r.get("id")} for r in rows[:n_real]]
    for it in items:
        it["key"] = pixel_key(it["path"])
    t0 = time.time()
    ans = predict(S, cache, items)
    log(f"[probe] {len(items)} images in {time.time() - t0:.0f}s")
    out_path = os.path.join(os.path.dirname(os.path.abspath(samples_path)), "sofake_probe.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for it, a in zip(items, ans):
            f.write(json.dumps({**it, **{k: a.get(k) for k in ("sofake_scope", "box224", "route_action", "output_source",
                                                              "observer_answer", "provider_answer", "p_fake_provider")}},
                               ensure_ascii=False) + "\n")
            pf = a.get("p_fake_provider")
            log(f"  {it['kind']:4s} {os.path.basename(it['path'])[:42]:42s} obs={str(a.get('observer_answer')):14s} "
                f"prov={str(a.get('provider_answer')):14s} p_prov={'n/a' if pf is None else f'{pf:.2f}'} "
                f"route={str(a.get('route_action')):16s} final={a.get('sofake_scope')} box={a.get('box224')}")
    for kind in ("fake", "real"):
        sub = [a for it, a in zip(items, ans) if it["kind"] == kind]
        if not sub:
            continue
        flag = sum(a.get("verdict") == "fake" for a in sub)
        tamp = sum(a.get("sofake_scope") == "TAMPERED" and bool(a.get("boxes")) for a in sub)
        log(f"[probe] {kind}: flagged fake {flag}/{len(sub)}, TAMPERED with a box {tamp}/{len(sub)}")
    log(f"[probe] saved {out_path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="So-Fake repo folder (contains sofake_r1/runtime)")
    ap.add_argument("--weights", required=True, help="folder with observer_adapter / decision_maker_adapter / provider")
    ap.add_argument("--qwen-base", required=True, help="local Qwen2.5-VL-7B-Instruct folder")
    ap.add_argument("--gpu-mem", type=float, default=11.0, help="GiB of bf16 weights on the GPU (rest streamed from RAM); 0 = all on GPU")
    ap.add_argument("--cpu-mem", type=float, default=40.0)
    ap.add_argument("--cache", default=None, help="jsonl answer cache (reused across runs)")
    ap.add_argument("--vram-cap", type=float, default=0.93, help="fraction of VRAM PyTorch may use (0 = no cap)")
    ap.add_argument("--probe", default=None, help="samples.jsonl: detection probe on fakes and reals, then exit")
    ap.add_argument("--probe-fake", type=int, default=20)
    ap.add_argument("--probe-real", type=int, default=10)
    ap.add_argument("--gen-batch", type=int, default=8, help="images per generate() call (amortizes weight streaming)")
    ap.add_argument("--selftest-batch", default=None,
                    help="folder of images: compare answers generated one by one vs in one batch, then exit")
    ap.add_argument("--selftest", default=None, help="run one image (e.g. sofake_r1/examples/example.jpg) and exit")
    a = ap.parse_args(argv)
    S = load(a)
    S["gen_batch"] = a.gen_batch
    if a.probe:
        return probe(S, a.probe, a.probe_fake, a.probe_real, Cache(a.cache))
    if a.selftest_batch:
        return selftest_batch(S, a.selftest_batch, a.gen_batch)
    if a.selftest:
        return selftest(S, a.selftest)
    serve(S, Cache(a.cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
