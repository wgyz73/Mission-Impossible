"""Run an editing model over a benchmark sample file.

Usage:
    python scripts/run_inference.py \
        --model    ip2p \
        --sample   data/full_noop/prompts.json \
        --root     data/full_noop \
        --out      outputs/ip2p

For ``full_noop`` the input is ``prompts.json`` (one entry per image, each
with multiple categories of prompts). For ``partial_noop`` pass the
``partial_test.jsonl`` file instead — the script auto-detects the format
by suffix.

Outputs are written as JPEGs at ``<out>/outputs/<key>.jpg`` where
``<key>`` is the per-row identifier defined by the dataset module
(``image_id__task__prompt_idx`` for full_noop, ``sample_id`` for
partial_noop). Already-present files are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.dataset import (load_full_noop, load_partial_noop,    # noqa: E402
                                FullNoOpSample, PartialNoOpSample)
from benchmark.models import get_model, list_models                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help=f"adapter name, one of {list_models()}")
    ap.add_argument("--sample", required=True,
                    help="prompts.json or partial_test.jsonl")
    ap.add_argument("--root", required=True,
                    help="directory whose layout matches the sample paths")
    ap.add_argument("--out", required=True,
                    help="output directory (will receive an outputs/ subdir)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max rows to run (0 = all)")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    sample_path = Path(args.sample)
    root = Path(args.root)
    out_dir = Path(args.out) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect format by suffix
    if sample_path.suffix == ".jsonl":
        rows = load_partial_noop(sample_path)
        kind = "partial_noop"
    elif sample_path.suffix == ".json":
        rows = load_full_noop(sample_path)
        kind = "full_noop"
    else:
        raise SystemExit(f"unknown sample format: {sample_path.suffix}")

    if args.limit:
        rows = rows[: args.limit]
    print(f"sample: {kind}  rows: {len(rows)}  out: {out_dir}")

    model = get_model(args.model)
    print(f"model: {model.name}")

    t0 = time.time()
    done = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    for i, r in enumerate(rows):
        out_path = out_dir / f"{r.key}.jpg"
        if out_path.exists():
            skipped += 1
            continue
        prompt = r.combined_instruction if isinstance(r, PartialNoOpSample) else r.prompt
        src = root / r.image_path
        try:
            img = Image.open(src)
            edited = model.edit(img, prompt)
            edited.convert("RGB").resize((512, 512)).save(out_path, quality=92)
            done += 1
        except Exception as e:
            errors.append((r.key, f"{type(e).__name__}: {e}"))
            print(f"  ! {r.key}: {errors[-1][1]}")
            continue
        if (done + skipped) % args.log_every == 0:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(rows) - i - 1) / max(rate, 1e-9) / 60
            print(f"  [{i+1}/{len(rows)}]  done={done}  skipped={skipped}  "
                  f"err={len(errors)}  {rate:.2f}/s  ETA {eta:.1f} min")

    print(f"\nfinished in {(time.time()-t0)/60:.1f} min  "
          f"done={done}  skipped={skipped}  errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
