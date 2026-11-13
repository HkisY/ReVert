"""Zero-shot Qwen-VL family detector (Qwen2.5-VL, Qwen3-VL) via transformers.

Two calls per batch:
  1. generate():  JSON answer with verdict + evidence boxes + explanation
  2. forward():   p(fake) read from the next-token logits after the forced
                  prefix '{"verdict": "' under the SAME prompt, so the score
                  and the explanation come from the same decision context.

Coordinate frames (the most common source of silent bugs):
  - Qwen2.5-VL grounds in ABSOLUTE pixels of the resized model input
    (H', W' = grid_h * patch, grid_w * patch)          -> coord_mode="abs"
  - Qwen3-VL grounds in RELATIVE coordinates on a 0..1000 grid
                                                        -> coord_mode="rel1000"
  coord_mode="auto" picks by model id. VERIFY on 20 images in the pilot by
  drawing the returned boxes (scripts/visualize_boxes.py).
"""

from __future__ import annotations

import numpy as np

from ..parsing import parse_response
from .base import Detector

PROMPT = (
    "You are an expert in image forensics. Decide whether this image is an authentic "
    "camera photograph, or whether it was fully or partially generated or edited by AI "
    "(for example inpainting, object replacement, local style change, or face swap).\n"
    "Respond with ONLY a JSON object, no other text:\n"
    '{"verdict": "real" or "fake", '
    '"evidence": [{"region": [x1, y1, x2, y2], "artifact": "short description"}], '
    '"explanation": "one or two sentences"}\n'
    "Give one evidence item per suspicious region, with a bounding box around it. "
    "If the verdict is real, evidence must be an empty list."
)
SCORE_PREFIX = '{"verdict": "'


class QwenVLDetector(Detector):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        coord_mode: str = "auto",
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str | None = None,
        processor_kwargs: dict | None = None,
        prompt: str = PROMPT,
        max_pixels: int | None = None,
        load_in_4bit: bool = False,
        gpu_mem_gib: float | None = None,
        cpu_mem_gib: float | None = None,
        offload_folder: str | None = None,
        offload_kv: bool = False,
    ):
        """max_pixels: cap on the resized input (Qwen2.5-VL processor
        argument); fewer pixels = fewer visual tokens = less memory.
        load_in_4bit: NF4 weights via bitsandbytes. This CHANGES the audited
        model (it is no longer the released checkpoint), so use it only for
        models that do not fit in bf16 and report them as '-nf4' variants.
        gpu_mem_gib / cpu_mem_gib: bf16 weights that do not fit in gpu_mem_gib
        are kept in CPU RAM and streamed to the GPU layer by layer during each
        forward pass (accelerate offload). The arithmetic still runs on the GPU
        in bf16, so the outputs are those of the released model; only speed
        changes. Larger batches amortize the streaming. offload_folder spills
        to disk if CPU RAM is also too small (very slow).
        offload_kv: keep the generation KV cache in CPU RAM (transformers
        'offloaded' cache, prefetched layer by layer). Exact same result;
        frees GPU memory for larger batches, which amortize weight streaming.
        EXPERIMENTAL: measure with scripts/bench_gpu.py --offload-kv first."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id.split("/")[-1] + ("-nf4" if load_in_4bit else "")
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = {"cache_implementation": "offloaded"} if offload_kv else {}
        pk = dict(processor_kwargs or {})
        if max_pixels:
            pk["max_pixels"] = int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_id, **pk)
        self.processor.tokenizer.padding_side = "left"
        kw = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if gpu_mem_gib is not None:
            mm = {0: f"{gpu_mem_gib}GiB"}
            if cpu_mem_gib is not None:
                mm["cpu"] = f"{cpu_mem_gib}GiB"
            kw["max_memory"] = mm
        if offload_folder:
            kw["offload_folder"] = offload_folder
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        # with offloading, model.device can point at the CPU; inputs go to the GPU
        self.input_device = "cuda:0" if torch.cuda.is_available() else self.model.device
        self.offloaded = any(str(d) in ("cpu", "disk") for d in getattr(self.model, "hf_device_map", {}).values())
        if coord_mode == "auto":
            coord_mode = "rel1000" if "qwen3" in model_id.lower() else "abs"
        self.coord_mode = coord_mode
        ip = self.processor.image_processor
        self.patch = int(getattr(ip, "patch_size", 14))
        self.merge = int(getattr(ip, "merge_size", 2))
        tok = self.processor.tokenizer
        self.fake_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("fake", "Fake")})
        self.real_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("real", "Real")})

    # ------------------------------------------------------------------ utils
    def _chat_text(self) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _to_orig(self, box, h, w, grid_hw) -> list[float]:
        x1, y1, x2, y2 = box
        if self.coord_mode == "rel1000":
            sx, sy = w / 1000.0, h / 1000.0
        else:  # absolute pixels on the resized input
            hh, ww = grid_hw[0] * self.patch, grid_hw[1] * self.patch
            sx, sy = w / float(ww), h / float(hh)
        return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    # ---------------------------------------------------------------- predict
    def predict(self, images: list[np.ndarray], explain: bool = True) -> list[dict]:
        import torch
        from PIL import Image

        pil = [Image.fromarray(im) for im in images]
        text = self._chat_text()
        dev = self.input_device

        # 1) explanation (skipped for score-only control queries)
        texts, grid, ntok = [""] * len(pil), [[1, 1, 1]] * len(pil), [0] * len(pil)
        if explain:
            inputs = self.processor(text=[text] * len(pil), images=pil, return_tensors="pt", padding=True).to(dev)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          **self.gen_kwargs)
            gen = out[:, inputs["input_ids"].shape[1]:]
            texts = self.processor.batch_decode(gen, skip_special_tokens=True)
            pad = self.processor.tokenizer.pad_token_id
            ntok = (gen != pad).sum(dim=1).tolist() if pad is not None else [gen.shape[1]] * len(pil)
            g = inputs.get("image_grid_thw")
            grid = g.tolist() if g is not None else [[1, 1000 // self.patch, 1000 // self.patch]] * len(pil)

        # 2) score: p(fake) after forced prefix
        sinputs = self.processor(text=[text + SCORE_PREFIX] * len(pil), images=pil, return_tensors="pt", padding=True).to(dev)
        with torch.inference_mode():
            logits = self.model(**sinputs).logits[:, -1, :].float()
        lf = torch.logsumexp(logits[:, self.fake_ids], dim=-1)
        lr = torch.logsumexp(logits[:, self.real_ids], dim=-1)
        p_fake = torch.sigmoid(lf - lr).cpu().tolist()

        answers = []
        for im, t, g, p, nt in zip(images, texts, grid, p_fake, ntok):
            if not explain:
                answers.append({"verdict": None, "p_fake": float(p), "boxes": None, "mask_path": None,
                                "artifacts": [], "raw_text": "", "parse_mode": "score_only"})
                continue
            h, w = im.shape[:2]
            parsed = parse_response(t)
            boxes = [self._to_orig(b, h, w, g[1:]) for b in parsed["boxes"]]
            answers.append(
                {
                    "verdict": parsed["verdict"],
                    "p_fake": float(p),
                    "boxes": boxes,
                    "mask_path": None,
                    "artifacts": parsed["artifacts"],
                    "raw_text": t,
                    "parse_mode": parsed["parse_mode"],
                    "n_gen_tokens": int(nt),
                }
            )
        return answers
