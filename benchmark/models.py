"""Pluggable model adapters for evaluation.

The base class ``EditModel`` declares a single method, ``edit(image, prompt)``,
that maps a PIL Image and an instruction string to a PIL Image. Concrete
implementations wrap diffusion pipelines or hosted multimodal APIs.

Add new adapters by subclassing ``EditModel`` and registering them in
``_REGISTRY`` at the bottom of the file. We deliberately keep these
implementations thin; production users typically extend them with logging,
caching, batching, retries, etc.
"""
from __future__ import annotations

import abc
import base64
import io
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image


# ---------------------------------------------------------------------------
class EditModel(abc.ABC):
    """Image-editing model adapter."""

    name: str = "abstract"

    @abc.abstractmethod
    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        ...


# ---------------------------------------------------------------------------
# Diffusion: works for InstructPix2Pix, MagicBrush, HQ-Edit, ...
# ---------------------------------------------------------------------------
@dataclass
class _SDInstructPix2PixConfig:
    model_id: str
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    use_euler_ancestral: bool = False  # some checkpoints recommend this
    resolution: int = 512


class SDInstructPix2Pix(EditModel):
    """Wrapper around ``StableDiffusionInstructPix2PixPipeline``.

    Configure with the HF model id of any compatible checkpoint. The
    inference parameters are exposed because different checkpoints publish
    different recommended values; defaults match the original instruct-pix2pix.
    """

    def __init__(self, cfg: _SDInstructPix2PixConfig, device: str = "cuda"):
        import torch
        from diffusers import (StableDiffusionInstructPix2PixPipeline,
                                EulerAncestralDiscreteScheduler)
        self.cfg = cfg
        self.name = cfg.model_id
        self.device = device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            cfg.model_id, torch_dtype=torch.float16, safety_checker=None,
        ).to(device)
        if cfg.use_euler_ancestral:
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        import torch
        img = image.convert("RGB").resize((self.cfg.resolution, self.cfg.resolution))
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.pipe(
                prompt,
                image=img,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                image_guidance_scale=self.cfg.image_guidance_scale,
            ).images[0]
        return out


# ---------------------------------------------------------------------------
# OpenAI image-edit (gpt-image-1)
# ---------------------------------------------------------------------------
class OpenAIImageEdit(EditModel):
    """Adapter for the OpenAI Images Edit endpoint.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable.
    """

    name = "openai/gpt-image-1"

    def __init__(self, quality: str = "medium", size: str = "1024x1024"):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("set OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
        self.quality = quality
        self.size = size

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        result = self.client.images.edit(
            model="gpt-image-1",
            image=buf,
            prompt=prompt,
            size=self.size,
            quality=self.quality,
            n=1,
        )
        b64 = result.data[0].b64_json
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------
# Google Gemini 2.5 Flash Image
# ---------------------------------------------------------------------------
class GeminiImageEdit(EditModel):
    """Adapter for Gemini 2.5 Flash Image.

    Reads the API key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
    Gemini may occasionally return only text for some prompts; we surface
    that as a ``RuntimeError`` so the caller can decide what to do
    (skip the row, retry, fall back to identity).
    """

    name = "google/gemini-2.5-flash-image"

    def __init__(self):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("set GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        resp = self.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, image.convert("RGB")],
        )
        if not resp.candidates:
            raise RuntimeError("no candidates in response")
        for part in resp.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGB")
        # No image in the response (e.g., model declined with text only).
        raise RuntimeError("response carried no inline image")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY = {
    # diffusion checkpoints — pick whichever you like
    "ip2p":       lambda **kw: SDInstructPix2Pix(_SDInstructPix2PixConfig(
                      model_id="timbrooks/instruct-pix2pix"), **kw),
    "magicbrush": lambda **kw: SDInstructPix2Pix(_SDInstructPix2PixConfig(
                      model_id="vinesmsuic/magicbrush-jul7"), **kw),
    "hq_edit":    lambda **kw: SDInstructPix2Pix(_SDInstructPix2PixConfig(
                      model_id="UCSC-VLAA/HQ-Edit-ckpt",
                      num_inference_steps=30,
                      guidance_scale=7.0,
                      use_euler_ancestral=True), **kw),
    # API backends
    "gpt_image":  lambda **kw: OpenAIImageEdit(**kw),
    "gemini":     lambda **kw: GeminiImageEdit(**kw),
}


def get_model(name: str, **kwargs) -> EditModel:
    """Factory: instantiate a registered model by short name."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown model '{name}'. options: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY)
