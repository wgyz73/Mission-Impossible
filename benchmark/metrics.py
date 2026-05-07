"""Metric implementations.

The functions here compute *pair distances* (or text-image cosines).
Aggregation into rates such as "over-edit rate at threshold tau" is left
to the caller; we deliberately do not bake any reference threshold into
the code.

Backbone weights are downloaded lazily and cached. CLIP and DINOv2 use a
single GPU when one is available; CPU fallback is also supported.
"""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from typing import Iterable

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy-loaded backbones (single instance per process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _lpips_net():
    import lpips
    return lpips.LPIPS(net="alex").to(_device()).eval()


@lru_cache(maxsize=1)
def _clip(name: str):
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(name, pretrained="openai")
    return model.to(_device()).eval(), pre, open_clip.get_tokenizer(name)


@lru_cache(maxsize=1)
def _dinov2():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(_device()).eval()
    return proc, model


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------
def _resize(img: Image.Image, side: int) -> Image.Image:
    return img.convert("RGB").resize((side, side))


def _to_lpips_tensor(img: Image.Image):
    import torch, torchvision.transforms as T
    return T.ToTensor()(_resize(img, 256)).unsqueeze(0).to(_device()) * 2 - 1


# ---------------------------------------------------------------------------
# Pixel- / structure-level
# ---------------------------------------------------------------------------
def lpips_distance(a: Image.Image, b: Image.Image) -> float:
    """Perceptual LPIPS distance using the AlexNet backbone."""
    import torch
    with torch.no_grad():
        return float(_lpips_net()(_to_lpips_tensor(a), _to_lpips_tensor(b)).item())


def ssim_score(a: Image.Image, b: Image.Image) -> float:
    """Multi-channel SSIM in [-1, 1]; 1 means identical."""
    from skimage.metrics import structural_similarity as ssim
    a512 = np.asarray(_resize(a, 512), dtype=np.float32) / 255.0
    b512 = np.asarray(_resize(b, 512), dtype=np.float32) / 255.0
    return float(ssim(a512, b512, data_range=1.0, channel_axis=-1))


def mse(a: Image.Image, b: Image.Image) -> float:
    """Mean-squared-error in [0, 1]."""
    a512 = np.asarray(_resize(a, 512), dtype=np.float32) / 255.0
    b512 = np.asarray(_resize(b, 512), dtype=np.float32) / 255.0
    return float(np.mean((a512 - b512) ** 2))


# ---------------------------------------------------------------------------
# Feature-space metrics
# ---------------------------------------------------------------------------
def _clip_image_embed(model_name: str, img: Image.Image):
    import torch
    model, pre, _ = _clip(model_name)
    with torch.no_grad():
        x = pre(img.convert("RGB")).unsqueeze(0).to(_device())
        f = model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)


def _clip_text_embed(model_name: str, text: str):
    import torch
    model, _, tok = _clip(model_name)
    with torch.no_grad():
        ids = tok([text]).to(_device())
        f = model.encode_text(ids)
        return f / f.norm(dim=-1, keepdim=True)


def clip_image_image_cosine(a: Image.Image, b: Image.Image,
                             model_name: str = "ViT-L-14") -> float:
    """Cosine similarity between two images in CLIP feature space."""
    e_a = _clip_image_embed(model_name, a)
    e_b = _clip_image_embed(model_name, b)
    return float((e_a * e_b).sum().item())


def dinov2_distance(a: Image.Image, b: Image.Image) -> float:
    """``1 - cos(DINOv2-L CLS)``: a self-supervised semantic distance."""
    import torch
    proc, model = _dinov2()
    with torch.no_grad():
        def emb(img):
            inp = proc(images=img.convert("RGB"), return_tensors="pt").to(_device())
            v = model(**inp).last_hidden_state[:, 0]
            return v / v.norm(dim=-1, keepdim=True)
        return float(1.0 - (emb(a) * emb(b)).sum().item())


def clipt_delta(prompt: str, source: Image.Image, output: Image.Image,
                model_name: str = "ViT-L-14") -> float:
    """``cos(prompt, output) - cos(prompt, source)``.

    Used as an over-edit signal on no-op prompts: a model that holds the
    image still keeps this near zero; a model that follows the prompt's
    surface vocabulary pulls the embedding in the prompt's direction
    (positive value).
    """
    e_txt = _clip_text_embed(model_name, prompt)
    e_src = _clip_image_embed(model_name, source)
    e_out = _clip_image_embed(model_name, output)
    return float(((e_txt * e_out).sum() - (e_txt * e_src).sum()).item())


# ---------------------------------------------------------------------------
# VLM-OE — yes/no judgement by a multimodal LLM
# ---------------------------------------------------------------------------
_VLM_FULL_NOOP_PROMPT = (
    "You are evaluating an image-editing model on a no-op (no visible change "
    "expected) instruction.\n"
    "Edit instruction: \"{prompt}\"\n"
    "Image 1 is the source. Image 2 is the model's output.\n"
    "Did the model make any visible semantic change to the image (objects, "
    "colors, composition, texture) beyond what the instruction strictly "
    "requires?\n"
    "Reply with exactly one word: Yes or No."
)


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def vlm_oe_judge(prompt: str, source: Image.Image, output: Image.Image,
                 model: str = "gpt-4o", detail: str = "low") -> str:
    """Ask a multimodal LLM whether the output exhibits any semantic change
    beyond what the prompt requires. Returns one of ``"yes"`` / ``"no"``.

    Reads the API key from ``OPENAI_API_KEY``. The prompt template above
    is the one we use for the no-op subset; the partial-no-op variant
    needs three images and a slightly different question — write a sister
    function for it.
    """
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY for VLM-OE")
    client = openai.OpenAI(api_key=api_key)
    msg = [
        {"type": "text", "text": _VLM_FULL_NOOP_PROMPT.format(prompt=prompt)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{_b64(source)}", "detail": detail}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{_b64(output)}", "detail": detail}},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        max_tokens=8, temperature=0.0,
    )
    txt = (resp.choices[0].message.content or "").strip().lower()
    first = txt.split()[0].rstrip(".,!?:;").lower() if txt else ""
    if first not in ("yes", "no"):
        raise ValueError(f"VLM judge returned non-binary reply: {txt!r}")
    return first


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def over_edit_rate(values: Iterable[float], threshold: float) -> float:
    """Fraction of ``values`` that are >= ``threshold``.

    Used to summarize a per-row metric vector (LPIPS, DINOv2-distance,
    etc.) into a single "over-edit" rate at a chosen tau. The choice of
    threshold is intentionally a parameter — sweep it and consult rank
    stability before settling on a reporting value.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    return float(sum(1 for v in vals if v >= threshold) / len(vals))
