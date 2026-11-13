"""FakeShield MFLM (GLaMM + SAM) as a worker process: DTE-FDM's description -> mask.

Runs in its own conda env ('fsmflm': the SIDA env cloned, with
transformers==4.28.0, the version FakeShield's scripts install before MFLM).
Started (and stopped again, to free the GPU for DTE-FDM) by
revert_audit.detectors.fakeshield.FakeShieldDetector.finalize().

Reproduces MFLM/test.py (the paper's evaluation script) with the repo's own code:
  input  = 'This photo has been tampered. The following is a description of the
            tampering area and the basis for judgment: \\n' + DTE-FDM answer + '\\n' + ' ' + question
  prompt = 'The <image> provides an overview of the picture.\\n' + input, llava_v1 template
  model.evaluate(..., max_tokens_new=512, bboxes=None); mask = union of the [SEG] masks > 0
Deviations: the question is picked from test.py's list of three by the image hash
instead of random.choice (reproducible); the region encoder (mmcv/mmdet, used only
when the user supplies boxes, which test.py never does) is replaced by a stand-in
during import, so mmcv need not be compiled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types

PROTO = "@@REVERT@@"
QUESTIONS = [
    "Based on the description of the tampered area, please give the mask of the tampered area.",
    "Please provide a mask of the tampered region based on the description of the tampered region given above.",
    "Please give the mask of the tampered area based on the description of the tampered area given above.",
]
HEAD = "This photo has been tampered. The following is a description of the tampering area and the basis for judgment: \n"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(obj):
    sys.stdout.write(PROTO + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def question_for(key: str) -> str:
    return QUESTIONS[int(key[:8], 16) % len(QUESTIONS)]


def mflm_input(dte_text: str, key: str) -> str:
    return HEAD + dte_text + "\n" + " {}".format(question_for(key))


def find_mflm(repo: str) -> str:
    for c in (repo, os.path.join(repo, "MFLM"), os.path.join(repo, "FakeShield-main", "MFLM")):
        if os.path.isfile(os.path.join(c, "model", "GLaMM.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(f"MFLM/model/GLaMM.py not found under {repo}")


def import_glamm(mflm_dir: str):
    """Import GLaMMForCausalLM with model.layers (mmcv/mmdet) replaced by a stand-in."""
    import torch.nn as nn

    sys.path.insert(0, mflm_dir)

    class NoRegionEncoder(nn.Module):  # only used when boxes are given as input
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, *a, **k):
            raise RuntimeError("region (box) inputs are not supported in this audit")

    stub = types.ModuleType("model.layers")
    stub.MLVLROIQueryModule = NoRegionEncoder
    had = "model.layers" in sys.modules
    sys.modules["model.layers"] = stub
    try:
        from model.GLaMM import GLaMMForCausalLM
    finally:
        if not had:
            sys.modules.pop("model.layers", None)
    return GLaMMForCausalLM


def plan_offload(sizes, layer_names, budget):
    dm = {k: 0 for k in sizes}
    if budget is None:
        return dm
    gpu = sum(sizes.values())
    for name in reversed(layer_names):
        if gpu <= budget:
            break
        dm[name] = "cpu"
        gpu -= sizes[name]
    return dm


def load(args) -> dict:
    mflm = find_mflm(args.repo)
    GLaMMForCausalLM = import_glamm(mflm)
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor

    from model.SAM.utils.transforms import ResizeLongestSide

    t0 = time.time()
    ver = os.path.join(args.weights, "MFLM")
    tok = AutoTokenizer.from_pretrained(ver, model_max_length=1536, padding_side="right", use_fast=False)
    tok.pad_token = tok.unk_token
    ids = {k: tok(t, add_special_tokens=False).input_ids[0] for k, t in
           (("bbox_token_idx", "<bbox>"), ("seg_token_idx", "[SEG]"), ("bop_token_idx", "<p>"), ("eop_token_idx", "</p>"))}
    model = GLaMMForCausalLM.from_pretrained(ver, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **ids)
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    cfg = model.get_model().config
    if args.clip:
        cfg.vision_tower = args.clip
    model.get_model().initialize_vision_modules(cfg)
    model.get_model().get_vision_tower().to(dtype=torch.bfloat16)
    model = model.bfloat16()
    model.eval()

    inner = model.model
    sizes, layers = {}, []
    for name, child in inner.named_children():
        if name == "layers":
            for i, layer in enumerate(child):
                k = f"model.layers.{i}"
                sizes[k] = sum(p.numel() * p.element_size() for p in layer.parameters())
                layers.append(k)
        else:
            sizes[f"model.{name}"] = sum(p.numel() * p.element_size() for p in child.parameters()) + \
                sum(b.numel() * b.element_size() for b in child.buffers())
    for name, child in model.named_children():
        if name != "model":
            sizes[name] = sum(p.numel() * p.element_size() for p in child.parameters())
    dm = plan_offload(sizes, layers, int(args.gpu_mem * 2**30) if args.gpu_mem else None)
    n_cpu = sum(v == "cpu" for v in dm.values())
    if n_cpu == 0:
        model = model.cuda()
    else:
        from accelerate import dispatch_model

        model = dispatch_model(model, device_map=dm, main_device=0)
    log(f"[mflm] loaded in {time.time() - t0:.0f}s; {sum(sizes.values()) / 2**30:.1f} GiB bf16, {n_cpu} of "
        f"{len(layers)} LLaMA layers in CPU RAM; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return {"model": model, "tok": tok, "clip": CLIPImageProcessor.from_pretrained(model.config.vision_tower),
            "transform": ResizeLongestSide(1024), "n_offloaded": n_cpu}


def grounding_enc_processor(x):
    import torch
    import torch.nn.functional as F

    mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - mean) / std
    h, w = x.shape[-2:]
    return F.pad(x, (0, 1024 - w, 0, 1024 - h))


def segment(S, path: str, dte_text: str, key: str, mask_dir: str) -> dict:
    import cv2
    import numpy as np
    import torch

    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

    t0 = time.time()
    prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + \
        mflm_input(dte_text, key).replace("&lt;", "<").replace("&gt;", ">")
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    image_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    orig = [image_np.shape[:2]]
    g = S["clip"].preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().bfloat16()
    im = S["transform"].apply_image(image_np)
    resize = [im.shape[:2]]
    gr = grounding_enc_processor(torch.from_numpy(im).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().bfloat16()
    input_ids = tokenizer_image_token(prompt, S["tok"], return_tensors="pt").unsqueeze(0).cuda()
    with torch.no_grad():
        out_ids, pred_masks = S["model"].evaluate(g, gr, input_ids, resize, orig, max_tokens_new=512, bboxes=None)
    out_ids = out_ids[0][out_ids[0] != IMAGE_TOKEN_INDEX]
    text = S["tok"].decode(out_ids, skip_special_tokens=False).replace("\n", "").replace("  ", " ")
    text = text.split("ASSISTANT: ")[-1]
    n_seg = text.count("[SEG]")
    res = {"mflm_text": text, "n_seg": n_seg, "mask_path": None, "mask_frac": 0.0, "mflm_question": question_for(key)}
    if n_seg and pred_masks and pred_masks[0].numel():
        m = (pred_masks[0].float().cpu().numpy() > 0)
        m = m[-n_seg:].any(axis=0) if m.ndim == 3 else m
        os.makedirs(mask_dir, exist_ok=True)
        mp = os.path.join(mask_dir, key + ".png")
        cv2.imwrite(mp, (m * 255).astype(np.uint8))
        res.update({"mask_path": os.path.abspath(mp), "mask_frac": float(m.mean())})
    res["mflm_sec"] = time.time() - t0
    return res


def serve(S, mask_dir):
    send({"ready": True, "detector": "fakeshield-mflm", "n_offloaded": S["n_offloaded"]})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            out = [segment(S, it["path"], it["text"], it["key"], mask_dir) for it in req["items"]]
            send({"answers": out})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            send({"error": f"{type(e).__name__}: {e}"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FakeShield repository folder (contains MFLM/)")
    ap.add_argument("--weights", required=True, help="fakeshield-v1-22b folder")
    ap.add_argument("--clip", default=None, help="local openai/clip-vit-large-patch14-336")
    ap.add_argument("--gpu-mem", type=float, default=9.5)
    ap.add_argument("--mask-dir", default=os.path.join("runs", "fakeshield_masks"))
    ap.add_argument("--selftest", default=None, help="image path; with --selftest-text (a DTE-FDM answer)")
    ap.add_argument("--selftest-text", default=None)
    a = ap.parse_args(argv)
    S = load(a)
    if a.selftest:
        import hashlib

        key = hashlib.sha1(open(a.selftest, "rb").read()).hexdigest()[:20]
        text = a.selftest_text or ("1. Whether the picture has been tampered with / Description of the tampered area: "
                                   "The picture has been tampered with, specifically in the central part of the image.")
        t0 = time.time()
        r = segment(S, a.selftest, text, key, a.mask_dir)
        log(json.dumps(r, ensure_ascii=False))
        log(f"[selftest] {time.time() - t0:.0f}s")
        return 0
    serve(S, a.mask_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
