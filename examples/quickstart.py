"""Minimal end-to-end example.

Edit the constants below to point at your local dataset checkout, pick a
model, then run this file. It loads a handful of full_noop rows, edits
each one, scores the result, and prints a summary.

For the partial_noop subset, swap ``load_full_noop`` for
``load_partial_noop`` and replace ``r.prompt`` with ``r.combined_instruction``.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from benchmark.dataset import load_full_noop
from benchmark.models import get_model
from benchmark import metrics as M

DATA_ROOT  = Path("data/full_noop")              # edit me
SAMPLE     = DATA_ROOT / "prompts.json"
MODEL_NAME = "ip2p"                              # edit me
LIMIT      = 5                                   # how many rows to demo


def main():
    rows = load_full_noop(SAMPLE)[:LIMIT]
    model = get_model(MODEL_NAME)
    print(f"loaded {len(rows)} rows; using model {model.name}")

    for r in rows:
        src = Image.open(DATA_ROOT / r.image_path)
        out = model.edit(src, r.prompt)
        # Pair distances:
        d_lpips = M.lpips_distance(src, out)
        d_dino  = M.dinov2_distance(src, out)
        d_clip  = M.clip_image_image_cosine(src, out)
        d_clipt = M.clipt_delta(r.prompt, src, out)
        print(f"{r.key:60s}  LPIPS={d_lpips:.3f}  DINO_d={d_dino:.3f}  "
              f"CLIP-L={d_clip:.3f}  ΔCLIP-T={d_clipt:+.3f}")


if __name__ == "__main__":
    main()
