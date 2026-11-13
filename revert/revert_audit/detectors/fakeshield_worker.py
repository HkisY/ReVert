"""FakeShield (ICLR 2025) DTE-FDM as a persistent worker process.

Runs in its own conda env ('fsdte': the SIDA env cloned, with
transformers==4.37.2, the version FakeShield's scripts install before running
DTE-FDM). Started by revert_audit.detectors.fakeshield.FakeShieldDetector.

What is reproduced (FakeShield-main/DTE-FDM, unchanged code imported from the repo)
  - llava.model.builder.load_pretrained_model(model_path, None, "llava-v1.5-13b")
  - DomainTagGenerator (DTG.pth, ResNet-50) -> domain sentence prepended to the
    question, exactly as llava/serve/cli.py
  - the CLI question, conv template llava_v1, process_images, tokenizer_image_token
  - verdict rule of FakeShield's own MFLM stage: 'has not been tampered with' in
    the DTE-FDM answer -> real (no mask), otherwise fake
Deviations (stated in configs/gates.json before use)
  - greedy decoding (the release samples with temperature 0.2), so that one
    image always gets one answer; max_new_tokens 1024 as in llava/eval/model_vqa.py
  - p_fake: the answer always starts with
      '1. Whether the picture has been tampered with / Description of the tampered area: The picture has'
    and continues with 'been' (tampered) or 'not'. p_fake = P(been) / (P(been) + P(not))
    from ONE forward pass on prompt + that prefix, i.e. the model's own decision
    at the point where it writes it. Consistency with the greedy text is logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

PROTO = "@@REVERT@@"
QUESTION = ("Was this photo taken directly from the camera without any processing? Has it been tampered with by any "
            "artificial photo modification techniques such as ps? Please zoom in on any details in the image, paying "
            "special attention to the edges of the objects, capturing some unnatural edges and perspective "
            "relationships, some incorrect semantics, unnatural lighting and darkness etc.")
DOMAIN = {0: "This is a picture that is suspected to have been tampered with by AIGC inpainting. ",
          1: "This is a picture that is suspected to have been tampered with by DeepFake. ",
          2: "This is a picture that is suspected to have been tampered with by Photoshop. "}
PREFIX = "1. Whether the picture has been tampered with / Description of the tampered area: The picture has"
NOT_TAMPERED = "has not been tampered with"
MAX_NEW = 1024
CACHE_VERSION = "fakeshield-dte-w1"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(obj):
    sys.stdout.write(PROTO + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def verdict_from_text(text: str) -> str:
    return "real" if NOT_TAMPERED in text else "fake"


def find_dte(repo: str) -> str:
    for c in (repo, os.path.join(repo, "DTE-FDM"), os.path.join(repo, "FakeShield-main", "DTE-FDM")):
        if os.path.isdir(os.path.join(c, "llava")):
            return os.path.abspath(c)
    raise FileNotFoundError(f"DTE-FDM/llava not found under {repo}")


def load(args) -> dict:
    import torch

    dte = find_dte(args.repo)
    sys.path.insert(0, dte)
    import llava.model.llava_arch as arch
    from llava.model.builder import load_pretrained_model
    from llava.serve.cli import DomainTagGenerator

    if args.clip:  # local copy of openai/clip-vit-large-patch14-336 instead of the hub id in the config
        orig = arch.build_vision_tower

        def build_local(cfg, **kw):
            for k in ("mm_vision_tower", "vision_tower"):
                if getattr(cfg, k, None):
                    setattr(cfg, k, args.clip)
            return orig(cfg, **kw)

        arch.build_vision_tower = build_local
    w = args.weights
    t0 = time.time()
    kw = {}
    if args.gpu_mem and not (args.load_8bit or args.load_4bit):
        kw["max_memory"] = {0: f"{args.gpu_mem}GiB", "cpu": f"{args.cpu_mem}GiB"}
    tok, model, improc, _ = load_pretrained_model(os.path.join(w, "DTE-FDM"), None, "llava-v1.5-13b",
                                                  args.load_8bit, args.load_4bit, device_map="auto", **kw)
    model.eval()
    dtg = DomainTagGenerator(model_path=os.path.join(w, "DTG.pth"))
    ids_been = tok("has been", add_special_tokens=False).input_ids[-1]
    ids_not = tok("has not", add_special_tokens=False).input_ids[-1]
    dmap = getattr(model, "hf_device_map", {}) or {}
    n_cpu = sum(1 for d in dmap.values() if str(d) in ("cpu", "disk"))
    log(f"[fakeshield] DTE-FDM loaded in {time.time() - t0:.0f}s; modules on CPU: {n_cpu}; "
        f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB; tokens been={ids_been} not={ids_not}")
    return {"tok": tok, "model": model, "improc": improc, "dtg": dtg, "been": ids_been, "not": ids_not,
            "offloaded": n_cpu, "quant": "8bit" if args.load_8bit else "4bit" if args.load_4bit else "fp16"}


def build_inputs(S, path: str, with_prefix: bool):
    import torch
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from PIL import Image

    model = S["model"]
    label = S["dtg"].predict(path)
    inp = DOMAIN.get(label, "") + QUESTION
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if with_prefix:
        prompt = prompt + " " + PREFIX
    image = Image.open(path).convert("RGB")
    dev = "cuda:0"
    input_ids = tokenizer_image_token(prompt, S["tok"], IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(dev)
    image_tensor = process_images([image], S["improc"], model.config)
    image_tensor = image_tensor.to(dev, dtype=torch.float16)
    return input_ids, image_tensor, image.size, label


def score(S, path: str) -> tuple[float, int]:
    import torch

    input_ids, img, size, label = build_inputs(S, path, with_prefix=True)
    with torch.inference_mode():
        out = S["model"](input_ids=input_ids, images=img, image_sizes=[size])
    lg = out.logits[0, -1].float()
    lb, ln = lg[S["been"]], lg[S["not"]]
    return float(torch.sigmoid(lb - ln)), label


def explain(S, path: str) -> str:
    import torch

    input_ids, img, size, _ = build_inputs(S, path, with_prefix=False)
    with torch.inference_mode():
        out = S["model"].generate(input_ids, images=img, image_sizes=[size], do_sample=False,
                                  max_new_tokens=MAX_NEW, use_cache=True)
    return S["tok"].batch_decode(out, skip_special_tokens=True)[0].strip()


def answer(S, path: str, want_text: bool, force_text: bool = False) -> dict:
    """want_text: the caller needs the explanation. Under greedy decoding the first
    word after '... The picture has' is argmax(been, not), i.e. p_fake >= 0.5, so
    when p_fake < 0.5 the answer is known to read 'has not been tampered with'
    (no mask) and is not generated (force_text generates it anyway)."""
    t0 = time.time()
    p, label = score(S, path)
    ans = {"p_fake": p, "domain_tag": int(label), "boxes": None, "mask_path": None, "artifacts": [],
           "dte_quant": S["quant"], "verdict_native": True}
    if want_text and (p >= 0.5 or force_text):
        text = explain(S, path)
        v = verdict_from_text(text)
        ans.update({"verdict": v, "raw_text": text, "dte_text": text, "parse_mode": "native",
                    "prefix_ok": text.startswith(PREFIX), "score_agrees": (p >= 0.5) == (v == "fake"),
                    "needs_mask": v == "fake"})
    elif want_text:
        ans.update({"verdict": "real", "raw_text": "", "parse_mode": "native", "text_skipped": True,
                    "needs_mask": False})
    else:
        ans.update({"verdict": "fake" if p >= 0.5 else "real", "raw_text": "", "parse_mode": "score_only",
                    "needs_mask": False})
    ans["sec"] = time.time() - t0
    return ans


class Cache:
    def __init__(self, path):
        self.path, self.d = path, {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("v") == CACHE_VERSION:
                            self.d[(r["key"], r["text"])] = r["answer"]

    def get(self, key, text):
        return self.d.get((key, True)) or (None if text else self.d.get((key, False)))

    def put(self, key, text, ans):
        self.d[(key, text)] = ans
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"v": CACHE_VERSION, "key": key, "text": text, "answer": ans}, ensure_ascii=False) + "\n")


def predict(S, cache, items, explain_flag: bool) -> list[dict]:
    out = []
    for it in items:
        a = cache.get(it["key"], explain_flag)
        if a is None:
            a = answer(S, it["path"], explain_flag)
            cache.put(it["key"], explain_flag, a)
        else:
            a = dict(a, sec=0.0, cached=True)
        out.append(a)
    return out


def pixel_key(path: str) -> str:
    import hashlib

    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()[:20]


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def probe(S, samples_path, n, n_text, cache):
    """Score-only detection probe on the first n samples (x_f, x_raw, x_real), plus
    full answers (text) for the first n_text x_f and x_real."""
    rows = [json.loads(x) for x in open(samples_path, encoding="utf-8") if x.strip()][:n]
    res = {"fake": [], "raw": [], "real": []}
    t0 = time.time()
    for i, r in enumerate(rows):
        for kind in ("fake", "raw", "real"):
            if not r.get(kind):
                continue
            want = i < n_text and kind in ("fake", "real")
            a = predict(S, cache, [{"key": pixel_key(r[kind]), "path": r[kind]}], want)[0]
            res[kind].append(a)
            if want:
                log(f"  {kind:4s} {os.path.basename(r[kind])[:40]:40s} p_fake={a['p_fake']:.2f} verdict={a['verdict']} "
                    f"tag={a['domain_tag']} prefix_ok={a.get('prefix_ok')} | {a['raw_text'][:160]!r}")
        if (i + 1) % 10 == 0:
            log(f"[probe] {i + 1}/{len(rows)} samples, {time.time() - t0:.0f}s")
    real = [a["p_fake"] for a in res["real"]]
    for kind in ("fake", "raw", "real"):
        ps = [a["p_fake"] for a in res[kind]]
        if not ps:
            continue
        flag = sum(p >= 0.5 for p in ps)
        extra = f"  AUC vs real {auc(ps, real):.3f}" if kind != "real" else ""
        log(f"[probe] {kind}: flagged {flag}/{len(ps)} (p_fake >= 0.5), mean p_fake {sum(ps) / len(ps):.3f}{extra}")
    txt = [a for k in ("fake", "real") for a in res[k] if "prefix_ok" in a]
    if txt:
        log(f"[probe] text answers: starts with the expected prefix {sum(a['prefix_ok'] for a in txt)}/{len(txt)}, "
            f"greedy verdict agrees with p_fake {sum(a['score_agrees'] for a in txt)}/{len(txt)}")
    tags = {}
    for a in res["fake"] + res["real"]:
        tags[a["domain_tag"]] = tags.get(a["domain_tag"], 0) + 1
    log(f"[probe] domain tags (0 AIGC inpainting, 1 DeepFake, 2 Photoshop): {tags}")
    log(f"[probe] {time.time() - t0:.0f}s total")
    return 0


def serve(S, cache):
    send({"ready": True, "detector": "fakeshield-dte", "quant": S["quant"], "offloaded_modules": S["offloaded"]})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            send({"answers": predict(S, cache, req["items"], bool(req.get("explain", True)))})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            free_cuda()  # an OOM leaves the failed generation's blocks cached: release them before the retry
            send({"error": f"{type(e).__name__}: {e}"})


def free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FakeShield repository folder (contains DTE-FDM/llava)")
    ap.add_argument("--weights", required=True, help="fakeshield-v1-22b folder (DTE-FDM/, MFLM/, DTG.pth)")
    ap.add_argument("--clip", default=None, help="local openai/clip-vit-large-patch14-336 (default: from the hub)")
    ap.add_argument("--gpu-mem", type=float, default=11.0)
    ap.add_argument("--cpu-mem", type=float, default=40.0)
    ap.add_argument("--load-8bit", action="store_true", help="the release's own --load-8bit option")
    ap.add_argument("--load-4bit", action="store_true", help="the release's own --load-4bit option")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--probe", default=None, help="samples.jsonl: detection probe, then exit")
    ap.add_argument("--probe-n", type=int, default=40)
    ap.add_argument("--probe-text", type=int, default=3, help="also generate the explanation for the first N")
    ap.add_argument("--selftest", default=None, help="one image: print score and explanation, then exit")
    a = ap.parse_args(argv)
    S = load(a)
    cache = Cache(a.cache)
    if a.selftest:
        t0 = time.time()
        ans = answer(S, a.selftest, True, force_text=True)
        log(json.dumps({k: v for k, v in ans.items() if k != "raw_text"}, ensure_ascii=False))
        log("[selftest] DTE-FDM answer: " + ans["raw_text"])
        log(f"[selftest] {time.time() - t0:.0f}s; prefix_ok={ans['prefix_ok']} score_agrees={ans['score_agrees']}")
        return 0
    if a.probe:
        return probe(S, a.probe, a.probe_n, a.probe_text, cache)
    serve(S, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
