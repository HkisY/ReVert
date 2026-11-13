"""SIDA inference worker. Runs INSIDE the SIDA conda environment (old
transformers 4.31), started by revert_audit.detectors.sida.SidaDetector.

It only uses SIDA's own code paths; the model is loaded as in SIDA's chat.py.
Two inference protocols (--protocol):
  paper   (default) SIDA's test.py: model_forward(inference=True) on the training
          conversation; the class is read where the answer's [CLS] is predicted,
          exactly as in training and in the paper's evaluation.
  readme  SIDA's chat.py: SIDAForCausalLM.evaluate() unchanged, with the README
          prompt. The model does not generate [CLS] itself; the class is read at
          the [CLS] written in the prompt, a position it was never trained on.
For 'readme', two things are added, neither of which changes the model:

  * a forward hook on the classification head records its logits, so that we
    get class probabilities (evaluate() only returns the argmax);
  * optional CPU offload of some LLaMA layers (accelerate), because the bf16
    model (~15.3 GB) does not fit in 16 GB together with SAM's activations.
    The weights stay bf16; offloading only changes where they are stored.

Protocol (one JSON object per line):
    parent -> worker  {"cmd": "predict", "items": [{"path": ..., "key": ...}], "explain": true}
    worker -> parent  @@REVERT@@{"answers": [...]}      or  @@REVERT@@{"error": "..."}
Everything SIDA prints goes to stderr, so the protocol channel stays clean.

Self-test on one image (compares our class read-out with SIDA's own answer):
    python sida_worker.py --repo /path/to/SIDA --weights /path/to/SIDA-7B --selftest some.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

PROTO = "@@REVERT@@"
# The prompt the SIDA authors use for inference (README "Inference" example, copied
# verbatim including its typos). SIDA's training question from utils/SID_Set.py
# ("Can you identify if this image is real, full synthetic, or tampered image? ...")
# was tried first: SIDA-7B then answers in LISA's style ("Sure, the segmentation
# result is [SEG].") without the [CLS] token its classifier reads, i.e. no decision.
PROMPT = ("Please answer begin with [CLS] for classification, if the image is  tampered, "
          "ouput mask the tampered region.")
CLASSES = ["real", "full_synthetic", "tampered"]
# SIDA's training/test question and answers (utils/SID_Set.py). The 'paper' protocol
# reproduces test.py: teacher-forced conversation, class read at the position that
# predicts the answer's first token [CLS]; the mask is decoded from the [SEG] of the
# tampered answer (test.py uses the ground-truth label there, we use SIDA's own class).
TRAIN_Q = ("Can you identify if this image is real, full synthetic, or tampered image? "
           "Please mask the tampered regions if it is tampered.")
TRAIN_A = {0: "[CLS] The image is real", 1: "[CLS] The image is full synthetic", 2: "[CLS] The image is tampered [SEG]"}
IMAGE_EXTRA = 255  # one <image> token becomes 256 CLIP patch features (SIDA hard-codes 255)


# --------------------------------------------------------------- pure helpers
def cls_row_position(output_ids: list[int], cls_token_idx: int, image_extra: int = IMAGE_EXTRA) -> int | None:
    """Row of the cls-head output that SIDA.evaluate() uses for its decision:
    evaluate() builds mask = [0]*255 + [ids[1:] == [CLS]] and takes the LAST
    selected row."""
    pos = [image_extra + j for j in range(len(output_ids) - 1) if output_ids[j + 1] == cls_token_idx]
    return pos[-1] if pos else None


def plan_offload(sizes: dict[str, int], layer_names: list[str], budget_bytes: int | None) -> dict[str, object]:
    """device_map: everything on GPU 0 except as many of the LAST LLaMA layers
    on 'cpu' as needed to bring the GPU share under budget_bytes."""
    dm: dict[str, object] = {k: 0 for k in sizes}
    if budget_bytes is None:
        return dm
    gpu = sum(sizes.values())
    for name in reversed(layer_names):
        if gpu <= budget_bytes:
            break
        dm[name] = "cpu"
        gpu -= sizes[name]
    if gpu > budget_bytes:
        raise ValueError(f"even with all LLaMA layers offloaded the GPU part is {gpu / 2**30:.1f} GiB "
                         f"> budget {budget_bytes / 2**30:.1f} GiB")
    return dm


def safe_decode(tok, ids) -> str:
    """Decode ids that may contain the image placeholder (-200) or ids the
    tokenizer does not know; unknown ids are shown as <id:N> instead of crashing."""
    n = len(tok)
    out, run = [], []
    for t in ids:
        if 0 <= t < n:
            run.append(t)
            continue
        if run:
            out.append(tok.decode(run, skip_special_tokens=False))
            run = []
        if t != -200:
            out.append(f"<id:{t}>")
    if run:
        out.append(tok.decode(run, skip_special_tokens=False))
    return "".join(out)


def softmax(xs):
    import math

    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


# ------------------------------------------------------------------- loading
def import_sida_model(repo: str):
    """Import SIDAForCausalLM from the SIDA repo.

    model/SIDA.py imports deepspeed and torchviz at module level but never uses
    them at inference; both are painful on Windows. A stand-in module is put in
    sys.modules ONLY while model/SIDA.py is being imported, and transformers is
    imported fully before that: transformers asks importlib whether deepspeed
    is installed, and a stand-in visible at that moment makes it crash
    ("deepspeed.__spec__ is None") or wrongly believe deepspeed is installed."""
    sys.path.insert(0, os.path.abspath(repo))
    try:  # import everything of transformers that SIDA touches, with no stand-ins around
        import transformers.modeling_utils  # noqa: F401
        from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel  # noqa: F401
    except ImportError:
        pass
    added = []
    for name in ("deepspeed", "torchviz"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            m = types.ModuleType(name)
            m.make_dot = lambda *a, **k: None
            sys.modules[name] = m
            added.append(name)
    try:
        from model.SIDA import SIDAForCausalLM
    finally:
        for name in added:  # nothing else may see the stand-ins
            sys.modules.pop(name, None)
    return SIDAForCausalLM


def load(repo: str, weights: str, clip: str, gpu_gib: float | None):
    SIDAForCausalLM = import_sida_model(repo)
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor

    from model.segment_anything.utils.transforms import ResizeLongestSide

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(weights, model_max_length=512, padding_side="right", use_fast=False)
    tok.pad_token = tok.unk_token
    seg_idx = tok("[SEG]", add_special_tokens=False).input_ids[0]
    cls_idx = tok("[CLS]", add_special_tokens=False).input_ids[0]
    model = SIDAForCausalLM.from_pretrained(weights, low_cpu_mem_usage=True, vision_tower=clip,
                                            seg_token_idx=seg_idx, cls_token_idx=cls_idx, torch_dtype=torch.bfloat16)
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    for k in ("ce_loss_weight", "dice_loss_weight", "bce_loss_weight", "cls_loss_weight", "mask_loss_weight"):
        setattr(model, k, 1.0)  # test.py defaults; only enter a loss value we never use
    model.get_model().initialize_vision_modules(model.get_model().config)  # as chat.py
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
    total = sum(sizes.values())
    budget = int(gpu_gib * 2**30) if gpu_gib else None
    dm = plan_offload(sizes, layers, budget)
    n_cpu = sum(v == "cpu" for v in dm.values())
    if n_cpu == 0:
        model = model.cuda()
    else:
        from accelerate import dispatch_model

        model = dispatch_model(model, device_map=dm, main_device=0)
    print(f"[sida] loaded in {time.time() - t0:.0f}s; weights {total / 2**30:.1f} GiB bf16, "
          f"{n_cpu} of {len(layers)} LLaMA layers kept in CPU RAM; GPU now {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
          file=sys.stderr, flush=True)
    clip_proc = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    return {"model": model, "tok": tok, "cls_idx": cls_idx, "clip": clip_proc,
            "transform": ResizeLongestSide(1024), "n_offloaded": n_cpu, "weights_gib": total / 2**30}


def _prompt_ids(S, prompt: str | None = None):
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN

    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + (prompt or S.get("prompt") or PROMPT)
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    return tokenizer_image_token(conv.get_prompt(), S["tok"], return_tensors="pt").unsqueeze(0).cuda()


def _inputs(S, path):
    import cv2
    import torch
    import torch.nn.functional as F

    image_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = S["clip"].preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().bfloat16()
    image = S["transform"].apply_image(image_np)
    resize_list = [image.shape[:2]]
    x = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # chat.py preprocess()
    mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - mean) / std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, 1024 - w, 0, 1024 - h))
    return image_np, image_clip, x.unsqueeze(0).cuda().bfloat16(), resize_list, original_size_list


def infer(S, path: str, max_new_tokens: int = 512):
    """Returns (answer dict without mask_path, mask bool array or None)."""
    import torch

    from utils.utils import IMAGE_TOKEN_INDEX

    model = S["model"]
    image_np, image_clip, image, resize_list, original_size_list = _inputs(S, path)
    input_ids = _prompt_ids(S)
    cap = {}
    hook = model.model.cls_head[0].register_forward_hook(lambda m, i, o: cap.__setitem__("out", o.detach().float().cpu()))
    t0 = time.time()
    try:
        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                                    max_new_tokens=max_new_tokens, tokenizer=None)
    finally:
        hook.remove()
    ids = output_ids[0].tolist()
    text = safe_decode(S["tok"], [t for t in ids if t != IMAGE_TOKEN_INDEX])
    pos = cls_row_position(ids, S["cls_idx"])
    if pos is None or "out" not in cap:
        return {"verdict": None, "p_fake": None, "boxes": [], "raw_text": text, "parse_mode": "failed",
                "sida_class": None, "sec": time.time() - t0}, None
    logits = cap["out"][0]
    if logits.shape[0] != IMAGE_EXTRA + len(ids) - 1:
        raise RuntimeError(f"cls-head output length {logits.shape[0]} != 255 + {len(ids) - 1}: "
                           "the read-out would not match SIDA.evaluate()")
    probs = softmax(logits[pos].tolist())
    pred = max(range(3), key=lambda k: probs[k])
    if (pred == 2) != (len(pred_masks) > 0):  # evaluate() segments iff its own argmax is 'tampered'
        raise RuntimeError(f"class read-out ({CLASSES[pred]}) disagrees with SIDA.evaluate() "
                           f"({'mask' if pred_masks else 'no mask'})")
    mask, n_seg = None, 0
    if pred == 2 and pred_masks[0].shape[0] > 0:
        n_seg = int(pred_masks[0].shape[0])
        mask = pred_masks[0][0].float().cpu().numpy() > 0  # chat.py keeps the first mask
    ans = {"verdict": "real" if pred == 0 else "fake", "verdict_native": True, "p_fake": 1.0 - probs[0],
           "probs": dict(zip(CLASSES, probs)), "sida_class": CLASSES[pred], "boxes": [], "artifacts": [],
           "raw_text": text, "parse_mode": "native", "n_seg_masks": n_seg,
           "mask_frac": float(mask.mean()) if mask is not None else 0.0, "sec": time.time() - t0}
    return ans, mask


# ---------------------------------------------------------------------- main
def _train_conv_ids(S, answer: str):
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN

    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{TRAIN_Q}")
    conv.append_message(conv.roles[1], answer)
    text = conv.get_prompt().replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    return tokenizer_image_token(text, S["tok"], return_tensors="pt").unsqueeze(0).cuda()


def infer_paper(S, path: str):
    """SIDA as evaluated in its test.py: model_forward(inference=True) on the
    training conversation. Pass 1 reads the class; pass 2 (only if SIDA says
    'tampered') decodes the mask from the tampered answer's [SEG]."""
    import torch

    model = S["model"]
    image_np, image_clip, image, resize_list, original_size_list = _inputs(S, path)
    h, w = original_size_list[0]

    def fwd(answer, label):
        ids = _train_conv_ids(S, answer)
        with torch.no_grad():
            return model.model_forward(
                images=image, images_clip=image_clip, input_ids=ids,
                cls_labels=torch.tensor([label]).cuda(), labels=None,
                attention_masks=torch.ones_like(ids), offset=torch.tensor([0, 1]).long().cuda(),
                masks_list=[torch.zeros(1, h, w)], cls_labels_list=[torch.tensor([label])],
                label_list=[torch.zeros(h, w)], resize_list=resize_list, inference=True)

    t0 = time.time()
    out = fwd(TRAIN_A[0], 0)  # the class position comes before any answer word: the answer text cannot leak
    logits = out["logits"].float().cpu()
    if logits.shape[0] != 1:
        raise RuntimeError(f"expected exactly one [CLS] in the conversation, got {logits.shape[0]}")
    probs = softmax(logits[0].tolist())
    pred = max(range(3), key=lambda k: probs[k])
    mask, n_seg = None, 0
    if pred == 2:
        pm = fwd(TRAIN_A[2], 2)["pred_masks"]
        if pm and pm[0].shape[0] > 0:
            n_seg = int(pm[0].shape[0])
            mask = pm[0][0].float().cpu().numpy() > 0
    ans = {"verdict": "real" if pred == 0 else "fake", "verdict_native": True, "p_fake": 1.0 - probs[0],
           "probs": dict(zip(CLASSES, probs)), "sida_class": CLASSES[pred], "boxes": [], "artifacts": [],
           "raw_text": TRAIN_A[pred], "parse_mode": "native", "n_seg_masks": n_seg, "sida_protocol": "paper",
           "mask_frac": float(mask.mean()) if mask is not None else 0.0, "sec": time.time() - t0}
    return ans, mask


def run_one(S, path, max_new_tokens):
    if S.get("protocol", "paper") == "paper":
        return infer_paper(S, path)
    ans, mask = infer(S, path, max_new_tokens)
    ans["sida_protocol"] = "readme"
    return ans, mask


def serve(S, proto, mask_dir: str, max_new_tokens: int):
    import cv2
    import numpy as np
    import torch

    def send(obj):
        proto.write(PROTO + json.dumps(obj) + "\n")
        proto.flush()

    send({"ready": True, "n_offloaded_layers": S["n_offloaded"], "weights_gib": S["weights_gib"]})
    while True:
        line = sys.stdin.readline()
        if not line:  # parent closed the pipe
            break
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        try:
            out = []
            for it in req["items"]:
                ans, mask = run_one(S, it["path"], max_new_tokens)
                ans["mask_path"] = None
                if mask is not None and req.get("explain", True):
                    os.makedirs(mask_dir, exist_ok=True)
                    mp = os.path.join(mask_dir, it["key"] + ".png")
                    cv2.imwrite(mp, mask.astype(np.uint8) * 255)
                    ans["mask_path"] = os.path.abspath(mp)
                out.append(ans)
            send({"answers": out})
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            send({"error": f"{e.__class__.__name__}: {e}"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="folder of the SIDA git repository")
    ap.add_argument("--weights", required=True, help="folder of saberzl/SIDA-7B")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14", help="CLIP vision tower (HF id or local folder)")
    ap.add_argument("--gpu-mem", type=float, default=12.0, help="GiB of bf16 weights kept on the GPU; rest in CPU RAM")
    ap.add_argument("--mask-dir", default="sida_masks")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--protocol", default="paper", choices=["paper", "readme"],
                    help="paper = SIDA's test.py (training question, class at the answer's [CLS]); "
                         "readme = chat.py with the README prompt")
    ap.add_argument("--prompt", default=None, help="override the inference prompt (default: the SIDA README prompt)")
    ap.add_argument("--quickcheck", default=None, help="samples.jsonl: both protocols on the first --n samples (real/fake/raw)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--selftest", default=None, help="image path: compare our read-out with SIDA's own answer")
    a = ap.parse_args(argv)
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # anything printed by SIDA / transformers goes to stderr
    sys.stdout = sys.stderr
    S = load(a.repo, a.weights, a.clip, a.gpu_mem)
    S["prompt"] = a.prompt or PROMPT
    S["protocol"] = a.protocol
    if a.quickcheck:
        rows = []
        with open(a.quickcheck, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        # alternate editors so both appear among the first n
        by_ed = {}
        for r in rows:
            by_ed.setdefault(r.get("editor"), []).append(r)
        picked, i = [], 0
        while len(picked) < a.n and any(i < len(v) for v in by_ed.values()):
            for v in by_ed.values():
                if i < len(v) and len(picked) < a.n:
                    picked.append(v[i])
            i += 1
        table = {}
        for proto_name in ("paper", "readme"):
            S["protocol"] = proto_name
            for r in picked:
                for kind in ("real", "fake", "raw"):
                    ans, mask = run_one(S, r[kind], a.max_new_tokens)
                    key = (proto_name, r.get("editor"), kind)
                    table.setdefault(key, []).append((ans["sida_class"], ans["p_fake"], ans["sec"]))
        print("\n[quickcheck] protocol  editor      image  n  real/full_syn/tampered/none   mean p_fake  s/img", file=sys.stderr)
        for (pn, ed, kind), v in sorted(table.items(), key=lambda x: (x[0][0], str(x[0][1]), x[0][2])):
            c = [sum(x[0] == k for x in v) for k in CLASSES] + [sum(x[0] is None for x in v)]
            pf = [x[1] for x in v if x[1] is not None]
            print(f"[quickcheck] {pn:8s}  {str(ed):10s}  {kind:5s}  {len(v):2d}  {c[0]:3d}/{c[1]:3d}/{c[2]:3d}/{c[3]:3d}"
                  f"           {sum(pf) / max(1, len(pf)):.3f}       {sum(x[2] for x in v) / len(v):.1f}", file=sys.stderr)
        raise SystemExit(0)
    if a.selftest:
        ans, mask = infer(S, a.selftest, a.max_new_tokens)
        # SIDA's own answer: evaluate() with the tokenizer returns a canned sentence naming its class
        import torch

        image_np, image_clip, image, rl, ol = _inputs(S, a.selftest)
        with torch.no_grad():
            oid, _ = S["model"].evaluate(image_clip, image, _prompt_ids(S), rl, ol, max_new_tokens=a.max_new_tokens,
                                         tokenizer=S["tok"])
        official = safe_decode(S["tok"], oid[0].tolist())
        tok, emb = S["tok"], S["model"].get_input_embeddings().weight.shape[0]
        print(f"[selftest] tokenizer size {len(tok)}, model vocab {emb}, [CLS] id {S['cls_idx']}, "
              f"[CLS] pieces {tok.tokenize('[CLS]')}, [SEG] pieces {tok.tokenize('[SEG]')}", file=sys.stderr)
        print(f"[selftest] our generated text: {ans.get('raw_text', '')[-400:]!r}", file=sys.stderr)
        word = {"real": "real", "full_synthetic": "full synthetic", "tampered": "tampered"}.get(ans["sida_class"], "??")
        ok = ans["sida_class"] is not None and f"classified as {word}" in official
        print(json.dumps({k: v for k, v in ans.items() if k != "raw_text"}, ensure_ascii=False, default=str), file=sys.stderr)
        print(f"[selftest] official answer: {official!r}", file=sys.stderr)
        print(f"[selftest] readme protocol {'PASS' if ok else 'FAIL'}: our class '{ans['sida_class']}', {ans['sec']:.1f} s/image", file=sys.stderr)
        pa, pmask = infer_paper(S, a.selftest)
        print(f"[selftest] paper protocol: class '{pa['sida_class']}', probs "
              f"{ {k: round(v, 4) for k, v in pa['probs'].items()} }, mask "
              f"{'none' if pmask is None else f'{pmask.mean():.1%} of the image'}, {pa['sec']:.1f} s/image", file=sys.stderr)
        raise SystemExit(0 if ok else 1)
    serve(S, proto, a.mask_dir, a.max_new_tokens)


if __name__ == "__main__":
    main()
