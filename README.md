# MISSION IMPOSSIBLE — Reference Evaluation Code

> Anonymous submission to NeurIPS 2026 — code released for double-blind review.

This repository contains a minimal reference implementation for evaluating instruction-based image-editing models on the **MISSION IMPOSSIBLE** no-op benchmark.

It does **not** ship any benchmark numbers or model checkpoints; the dataset itself is hosted separately. This codebase only provides the loader, model adapters, and metric implementations needed to reproduce evaluation runs.

## What's here

```
mission_impossible_eval/
├── benchmark/
│   ├── dataset.py        # load full_noop + partial_noop (from HF Hub or local)
│   ├── models.py         # pluggable model adapters
│   └── metrics.py        # LPIPS / SSIM / DINOv2 / CLIP-L / ΔCLIP-T / VLM-OE
├── scripts/
│   ├── run_inference.py  # CLI: run a model on the benchmark, save outputs
│   └── compute_metrics.py# CLI: score saved outputs with the metric suite
└── examples/quickstart.py
```

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

1. Obtain the benchmark dataset (anonymous, hosted as a dataset repository on the HuggingFace Hub — refer to your reviewer link). For `full_noop`, follow `data/full_noop/IMAGES.md` from the dataset repo to reconstruct image files.
2. Run a model on the benchmark:
   ```bash
   python scripts/run_inference.py \
     --model ip2p \
     --sample data/full_noop/prompts.json \
     --root  data/full_noop \
     --out   outputs/ip2p
   ```
3. Score the outputs:
   ```bash
   python scripts/compute_metrics.py \
     --out_dir outputs/ip2p \
     --sample  data/full_noop/prompts.json \
     --root    data/full_noop \
     --metrics lpips,ssim,dinov2,clip_l,clipt
   ```

`compute_metrics.py` writes a per-row JSON; aggregation (mean, over-edit rate at chosen τ, per-task breakdowns) is left to the user's notebook of choice. We do not bake any reference threshold into the code.

## Notes

- All anchor thresholds, ranking tables, and reported numbers belong to the paper, not to the code. The code exposes metrics as plain pair-distances; combining them into an over-edit rate at a chosen τ is a one-liner downstream.
- Model adapters are intentionally thin. The provided ones are `ip2p`-style diffusion editors (works for any HF model that loads via `StableDiffusionInstructPix2PixPipeline`), an OpenAI image-edit adapter, and a Google Gemini image-edit adapter. New backbones can be added by subclassing `EditModel` in `benchmark/models.py`.
- VLM-OE relies on a multimodal API; you must supply your own credentials (env var, see `metrics.py`).

## License

This evaluation code is released under the MIT License (see `LICENSE`). The benchmark data is released separately under CC-BY-4.0; image content carries its upstream license (see the dataset card).
