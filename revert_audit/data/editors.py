"""Image editors used to make the fakes.

kind = "inpaint"   regenerates only inside a given mask       (sdxl, flux-fill, kandinsky)
kind = "instruct"  follows a text instruction, regenerates the whole image;
                   the builder recovers the mask by differencing
                   (kontext, qwen-edit, api-openai, api-gemini)

Memory options (none of them changes what the model computes):
  offload      whole sub-models on the GPU only while they run
  seq_offload  layer-by-layer streaming: bf16 FLUX / Kontext on 16 GB cards
  nf4          4-bit transformer weights: this DOES change the generator, so
               the editor is renamed '<name>-nf4' in the data
Do not enable VAE tiling: tiled decoding can leave seams in x_raw.
"""

from __future__ import annotations

import io
import os

import numpy as np

REPOS = {
    "sdxl": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "flux-fill": "black-forest-labs/FLUX.1-Fill-dev",
    "kontext": "black-forest-labs/FLUX.1-Kontext-dev",
    "qwen-edit": "Qwen/Qwen-Image-Edit",
    "qwen-image": "Qwen/Qwen-Image",  # text-to-image MMDiT + the InstantX inpainting ControlNet (below)
}
# Qwen-Image has no inpainting checkpoint of its own. Plain latent blending (QwenImageInpaintPipeline) gave
# incoherent fills in the smoke run; this mask-conditioned ControlNet is the Qwen-Image counterpart of FLUX-Fill.
QWEN_INPAINT_CN = "InstantX/Qwen-Image-ControlNet-Inpainting"
# Third editor of the full experiment (tried after the qwen-image smoke runs; not used in the paper): Qwen-Image-Edit through
# diffusers' QwenImageEditInpaintPipeline (mask-based; the edit model sees the whole photo).
REPOS["qwen-edit-inpaint"] = "Qwen/Qwen-Image-Edit"
# Third editor of the full experiment, final (after the Qwen-Image line failed the smoke rule three times):
# Kandinsky 2.2 inpainting, a dedicated inpainting U-Net (Sber AI; unCLIP image prior + MoVQ latents), loaded through
# AutoPipelineForInpainting as the combined prior+decoder pipeline. The prior repo is loaded with it.
REPOS["kandinsky"] = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
KANDINSKY_PRIOR = "kandinsky-community/kandinsky-2-2-prior"
KINDS = {"sdxl": "inpaint", "flux-fill": "inpaint", "kontext": "instruct", "qwen-edit": "instruct",
         "qwen-image": "inpaint", "qwen-edit-inpaint": "inpaint", "kandinsky": "inpaint", "api-openai": "instruct", "api-gemini": "instruct"}


def qwen_work_size(w: int, h: int) -> tuple[int, int]:
    """Qwen-Image (and its inpainting ControlNet) is trained around 1 MP; COCO photos are ~0.3 MP. Edit at ~1 MP
    (QWEN_WORK_MP, default 1.0; 0 = native size), sides multiples of 64, never smaller than the photo."""
    mp = float(os.environ.get("QWEN_WORK_MP", "1.0"))
    if mp <= 0 or w * h >= mp * 2**20:
        return w, h
    s = (mp * 2**20 / (w * h)) ** 0.5
    return max(w, int(round(w * s / 64)) * 64), max(h, int(round(h * s / 64)) * 64)


def offline_kw() -> dict:
    """With HF_HUB_OFFLINE=1 (the 4090 setup) say so explicitly: some diffusers versions ignore the variable for
    sharded weights and ask the Hub for the file list (OfflineModeIsEnabled). Loading from the cache is identical."""
    return {"local_files_only": True} if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on") else {}


def nf4_transformer(cls, repo: str, subfolder: str = "transformer"):
    import torch
    from diffusers import BitsAndBytesConfig

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    return cls.from_pretrained(repo, subfolder=subfolder, quantization_config=qc, torch_dtype=torch.bfloat16,
                               **offline_kw())


def place_pipe(pipe, offload: bool, seq_offload: bool) -> None:
    if seq_offload:
        pipe.enable_sequential_cpu_offload()
    elif offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")


# Sub-folders each local copy must contain (diffusers layout, as on the Hub).
REQUIRED = {
    "sdxl": ["scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"],
    "flux-fill": ["scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae"],
    "kontext": ["scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae"],
    "qwen-edit": ["scheduler", "text_encoder", "tokenizer", "transformer", "vae"],
    "qwen-image": ["scheduler", "text_encoder", "tokenizer", "transformer", "vae"],
    "qwen-edit-inpaint": ["scheduler", "text_encoder", "tokenizer", "transformer", "vae"],
}


def check_local_model(path: str, editor: str) -> list[str]:
    """Return what is missing from a manually downloaded model folder
    (empty list = looks complete)."""
    import glob

    if not os.path.isdir(path):
        return [f"folder does not exist: {path}"]
    missing = []
    if not os.path.exists(os.path.join(path, "model_index.json")):
        missing.append("model_index.json")
    for sub in REQUIRED.get(editor, []):
        d = os.path.join(path, sub)
        if not os.path.isdir(d):
            missing.append(sub + "/")
        elif sub not in ("scheduler", "tokenizer", "tokenizer_2") and not (
                glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin"))):
            missing.append(f"{sub}/*.safetensors (weights)")
    return missing


class Editor:
    name = "base"
    kind = "inpaint"

    def edit(self, image, mask: np.ndarray | None, prompt: str, seed: int):
        raise NotImplementedError


class DiffusersEditor(Editor):
    def __init__(self, name: str, offload: bool = False, seq_offload: bool = False, nf4: bool = False, steps: int | None = None,
                 path: str | None = None):
        import torch

        self.torch = torch
        self.base_name, self.kind = name, KINDS[name]
        self.name = (name + "-cn" if name == "qwen-image" else name) + ("-nf4" if nf4 and name != "sdxl" else "")
        repo = REPOS[name]
        if path:  # manually downloaded copy: use it, never touch the network
            missing = check_local_model(path, name)
            if missing:
                raise FileNotFoundError(f"{path} is not a complete {name} model folder; missing: {missing}. "
                                        f"Keep the folder structure exactly as on https://huggingface.co/{repo}")
            repo = path
        from ..gpu import check_host_memory

        check_host_memory(name, offload or seq_offload, nf4)
        if name == "sdxl":
            from diffusers import AutoPipelineForInpainting

            try:  # half-precision weight files if they were downloaded ...
                pipe = AutoPipelineForInpainting.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16", **offline_kw())
            except (OSError, ValueError):  # ... otherwise the full-precision ones, cast to fp16 (same result)
                pipe = AutoPipelineForInpainting.from_pretrained(repo, torch_dtype=torch.float16, **offline_kw())
        elif name == "flux-fill":
            from diffusers import FluxFillPipeline, FluxTransformer2DModel

            kw = {"transformer": nf4_transformer(FluxTransformer2DModel, repo)} if nf4 else {}
            pipe = FluxFillPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw, **offline_kw())
        elif name == "kontext":
            from diffusers import FluxKontextPipeline, FluxTransformer2DModel

            kw = {"transformer": nf4_transformer(FluxTransformer2DModel, repo)} if nf4 else {}
            pipe = FluxKontextPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw, **offline_kw())
        elif name == "qwen-edit":
            from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel

            kw = {"transformer": nf4_transformer(QwenImageTransformer2DModel, repo)} if nf4 else {}
            pipe = QwenImageEditPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw, **offline_kw())
        elif name == "qwen-image":  # needs diffusers >= 0.36 (QwenImageControlNetInpaintPipeline)
            from diffusers import (QwenImageControlNetInpaintPipeline, QwenImageControlNetModel,
                                   QwenImageTransformer2DModel)

            cn = QwenImageControlNetModel.from_pretrained(QWEN_INPAINT_CN, torch_dtype=torch.bfloat16, **offline_kw())
            kw = {"transformer": nf4_transformer(QwenImageTransformer2DModel, repo)} if nf4 else {}
            pipe = QwenImageControlNetInpaintPipeline.from_pretrained(repo, controlnet=cn, torch_dtype=torch.bfloat16,
                                                                      **kw, **offline_kw())
        elif name == "kandinsky":  # fp16 fits easily; same descriptive prompts and mask convention as SDXL
            from diffusers import AutoPipelineForInpainting

            pipe = AutoPipelineForInpainting.from_pretrained(repo, torch_dtype=torch.float16, **offline_kw())
        elif name == "qwen-edit-inpaint":  # diffusers >= 0.36 (QwenImageEditInpaintPipeline)
            from diffusers import QwenImageEditInpaintPipeline, QwenImageTransformer2DModel

            kw = {}
            if nf4:  # 20B transformer AND the 7B VL text encoder in 4 bit: with model offload each fits a 24 GB card
                # next to the ~1 MP VAE pass (the bf16 text encoder alone is ~16.6 GB)
                from transformers import BitsAndBytesConfig as TfBnb
                from transformers import Qwen2_5_VLForConditionalGeneration

                kw["transformer"] = nf4_transformer(QwenImageTransformer2DModel, repo)
                kw["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    repo, subfolder="text_encoder", torch_dtype=torch.bfloat16, **offline_kw(),
                    quantization_config=TfBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.bfloat16))
            pipe = QwenImageEditInpaintPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw, **offline_kw())
        else:
            raise ValueError(name)
        self.pipe = pipe
        self.steps = steps
        self.instruct_prompts = name == "qwen-edit-inpaint"  # the builder phrases the same edit as an instruction
        if name == "qwen-image" and not seq_offload:
            self._qwen_place()
        else:
            place_pipe(pipe, offload, seq_offload)
        pipe.set_progress_bar_config(disable=True)

    # ---- Qwen-Image: explicit placement instead of model offload (offload hooks left the 7B text encoder or the
    # 4-bit transformer on the GPU at the wrong time -> out of memory at ~1 MP, and 4-bit weights cannot be swapped
    # out reliably). Denoiser, ControlNet and VAE stay on the GPU (~15 GiB). The text encoder stays in RAM and
    # encodes prompts on the CPU (bf16, same weights and code as the pipeline's own encode_prompt); it is detached
    # from the pipeline so the pipeline runs on the GPU. Prompts come from a small vocabulary and are cached.
    def _qwen_place(self):
        torch, p = self.torch, self.pipe
        self._te, self._qcache = p.text_encoder, {}
        p.text_encoder = None
        self._te.to("cpu")
        for m in (p.transformer, p.controlnet, p.vae):
            m.to("cuda")
        torch.cuda.empty_cache()
        print(f"[qwen] on GPU: transformer + ControlNet + VAE = {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
              f"text encoder in RAM (prompts encoded on the CPU, cached)", flush=True)

    def _qwen_embed(self, prompt: str):
        import time

        torch, p = self.torch, self.pipe
        if prompt not in self._qcache:
            t0 = time.time()
            p.text_encoder = self._te
            try:
                with torch.no_grad():
                    e, m = p.encode_prompt(prompt=prompt, device=torch.device("cpu"), max_sequence_length=512)
            finally:
                p.text_encoder = None
            self._qcache[prompt] = (e, m)
            print(f"[qwen] encoded prompt {len(self._qcache)} on the CPU in {time.time() - t0:.0f}s: {prompt!r}",
                  flush=True)
        return tuple(None if x is None else x.to("cuda") for x in self._qcache[prompt])

    def edit(self, image, mask, prompt, seed):
        from PIL import Image

        g = self.torch.Generator("cpu").manual_seed(int(seed))  # cpu generator: identical noise with any offload mode
        W, H = image.size
        n = self.base_name
        if n == "sdxl":
            out = self.pipe(prompt=prompt, image=image, mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
                            height=H, width=W, strength=0.99, num_inference_steps=self.steps or 30,
                            guidance_scale=8.0, generator=g).images[0]
        elif n == "flux-fill":
            out = self.pipe(prompt=prompt, image=image, mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
                            height=H, width=W, guidance_scale=30.0, num_inference_steps=self.steps or 50,
                            max_sequence_length=512, generator=g).images[0]
        elif n == "kontext":
            out = self.pipe(image=image, prompt=prompt, guidance_scale=2.5, num_inference_steps=self.steps or 28,
                            generator=g).images[0]
        elif n == "qwen-image":  # whole image generated from noise, the ControlNet sees the masked image + mask
            # (the builder keeps only the pixels inside the mask and scales the output back to the photo's size);
            # the masked image is VAE-encoded with the global RNG, so seed it too for reproducibility
            self.torch.manual_seed(int(seed))
            Wq, Hq = qwen_work_size(W, H)
            img_q = image.resize((Wq, Hq), Image.LANCZOS) if (Wq, Hq) != (W, H) else image
            m_q = Image.fromarray(mask.astype(np.uint8) * 255)
            if (Wq, Hq) != (W, H):
                m_q = m_q.resize((Wq, Hq), Image.NEAREST)
            pe, pm = self._qwen_embed(prompt)
            ne, nm = self._qwen_embed(" ")
            out = self.pipe(prompt_embeds=pe, prompt_embeds_mask=pm, negative_prompt_embeds=ne,
                            negative_prompt_embeds_mask=nm, control_image=img_q, control_mask=m_q,
                            controlnet_conditioning_scale=1.0, height=Hq, width=Wq, true_cfg_scale=4.0,
                            num_inference_steps=self.steps or 50, generator=g).images[0]
            if out.size != (W, H):
                out = out.resize((W, H), Image.LANCZOS)
        elif n == "kandinsky":  # white = repaint (diffusers >= 0.19 convention, as for SDXL); native size
            out = self.pipe(prompt=prompt, image=image, mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
                            height=H, width=W, num_inference_steps=self.steps or 50, prior_num_inference_steps=25,
                            guidance_scale=4.0, prior_guidance_scale=4.0, generator=g).images[0]
        elif n == "qwen-edit-inpaint":  # same edit as the other editors, phrased as an instruction (build.py); the pipeline
            # works at ~1 MP (its own calculate_dimensions), the result is scaled back; only pixels in the mask are used
            self.torch.manual_seed(int(seed))
            # the model sees the photo with the masked area filled gray (it would otherwise copy the object it sees)
            ref = np.asarray(image.convert("RGB")).copy()
            ref[mask.astype(bool)] = 128
            out = self.pipe(prompt=prompt, negative_prompt=" ", image=Image.fromarray(ref),
                            mask_image=Image.fromarray(mask.astype(np.uint8) * 255), strength=1.0, true_cfg_scale=4.0,
                            num_inference_steps=self.steps or 50, generator=g).images[0]
            if out.size != (W, H):
                out = out.resize((W, H), Image.LANCZOS)
        else:  # qwen-edit
            out = self.pipe(image=image, prompt=prompt, true_cfg_scale=4.0, num_inference_steps=self.steps or 50,
                            generator=g).images[0]
        return out


class OpenAIEditor(Editor):
    """Commercial instruction editor via the OpenAI Images API (costs money).
    Needs `pip install openai` and OPENAI_API_KEY. The model name changes
    over time: check the provider's docs and pass --api-model."""

    kind = "instruct"

    def __init__(self, model: str = "gpt-image-1"):
        from openai import OpenAI

        self.client, self.model, self.name = OpenAI(), model, f"api-openai:{model}"

    def edit(self, image, mask, prompt, seed):
        import base64

        from PIL import Image

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.name = "image.png"
        buf.seek(0)
        r = self.client.images.edit(model=self.model, image=buf, prompt=prompt)
        return Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")


class GeminiEditor(Editor):
    """Commercial instruction editor via the Google GenAI SDK (costs money).
    Needs `pip install google-genai` and GEMINI_API_KEY (or GOOGLE_API_KEY).
    Check the provider's docs for the current image-editing model name."""

    kind = "instruct"

    def __init__(self, model: str = "gemini-2.5-flash-image"):
        from google import genai

        self.client, self.model, self.name = genai.Client(), model, f"api-gemini:{model}"

    def edit(self, image, mask, prompt, seed):
        from PIL import Image

        r = self.client.models.generate_content(model=self.model, contents=[prompt, image])
        for part in r.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("the API returned no image (refused or text-only answer)")


def build_editor(name: str, offload=False, seq_offload=False, nf4=False, api_model: str | None = None, steps=None,
                 path: str | None = None) -> Editor:
    if name == "api-openai":
        return OpenAIEditor(api_model or "gpt-image-1")
    if name == "api-gemini":
        return GeminiEditor(api_model or "gemini-2.5-flash-image")
    if name in REPOS:
        return DiffusersEditor(name, offload, seq_offload, nf4, steps, path)
    raise ValueError(f"unknown editor {name}; choose from {sorted(KINDS)}")


def api_key_present(name: str) -> bool:
    if name == "api-openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if name == "api-gemini":
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return True


def main(argv=None):
    """python -m revert_audit.data.editors --check /path/to/FLUX.1-Kontext-dev --editor kontext"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="local model folder to verify")
    ap.add_argument("--editor", required=True, choices=sorted(REQUIRED))
    a = ap.parse_args(argv)
    missing = check_local_model(a.check, a.editor)
    if missing:
        print(f"[editors] INCOMPLETE {a.check}: missing {missing}")
        raise SystemExit(1)
    print(f"[editors] OK: {a.check} looks like a complete {a.editor} model")


if __name__ == "__main__":
    main()
