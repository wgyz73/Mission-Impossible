"""Score model outputs against the source images using the metric suite.

Usage:
    python scripts/compute_metrics.py \
        --out_dir outputs/ip2p \
        --sample  data/full_noop/prompts.json \
        --root    data/full_noop \
        --metrics lpips,ssim,dinov2,clip_l,clipt

The script expects model outputs at ``<out_dir>/outputs/<key>.jpg`` (i.e.
the layout produced by ``run_inference.py``). Per-row scores are written
to ``<out_dir>/metrics.json``; aggregating to means or to over-edit rates
at a chosen threshold is the user's responsibility.

The optional ``vlm_oe`` metric uses a multimodal LLM judge and reads
``OPENAI_API_KEY``. It is the slowest and most expensive metric — turn
it off in development by omitting it from ``--metrics``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.dataset import (load_full_noop, load_partial_noop,    # noqa: E402
                                FullNoOpSample, PartialNoOpSample)
from benchmark import metrics as M                                    # noqa: E402


def _row_iter(sample_path: Path):
    if sample_path.suffix == ".jsonl":
        return load_partial_noop(sample_path)
    if sample_path.suffix == ".json":
        return load_full_noop(sample_path)
    raise SystemExit(f"unknown sample format: {sample_path.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="directory containing outputs/<key>.jpg from run_inference.py")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--root", required=True,
                    help="directory whose layout matches sample paths")
    ap.add_argument("--metrics", default="lpips,ssim,dinov2,clip_l,clipt",
                    help="comma-separated subset of "
                         "{lpips,ssim,mse,clip_l,dinov2,clipt,vlm_oe}")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    outputs_dir = out_dir / "outputs"
    root = Path(args.root)
    rows = _row_iter(Path(args.sample))
    metric_set = set(m.strip() for m in args.metrics.split(",") if m.strip())
    print(f"out: {out_dir}  rows: {len(rows)}  metrics: {sorted(metric_set)}")

    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        out_path = outputs_dir / f"{r.key}.jpg"
        if not out_path.exists():
            continue
        src_path = root / r.image_path
        prompt = r.combined_instruction if isinstance(r, PartialNoOpSample) else r.prompt
        rec = {"key": r.key,
               **({"task": r.task, "prompt_idx": r.prompt_idx, "image_id": r.image_id}
                  if isinstance(r, FullNoOpSample) else
                  {"sample_id": r.sample_id, "no_op_category": r.no_op_category})}
        with Image.open(src_path) as src, Image.open(out_path) as out:
            src = src.convert("RGB")
            out = out.convert("RGB")
            if "lpips"  in metric_set:  rec["lpips"]      = M.lpips_distance(src, out)
            if "ssim"   in metric_set:  rec["ssim"]       = M.ssim_score(src, out)
            if "mse"    in metric_set:  rec["mse"]        = M.mse(src, out)
            if "clip_l" in metric_set:  rec["clip_l"]     = M.clip_image_image_cosine(src, out)
            if "dinov2" in metric_set:  rec["dino_dist"]  = M.dinov2_distance(src, out)
            if "clipt"  in metric_set:  rec["clipt_delta"]= M.clipt_delta(prompt, src, out)
            if "vlm_oe" in metric_set:
                try:
                    rec["vlm_oe"] = M.vlm_oe_judge(prompt, src, out)
                except Exception as e:
                    rec["vlm_oe"] = None
                    rec["vlm_err"] = f"{type(e).__name__}: {e}"
        results.append(rec)
        if (i + 1) % 50 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{i+1}/{len(rows)}]  {rate:.1f}/s")

    path = out_dir / "metrics.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {path}  rows={len(results)}")


if __name__ == "__main__":
    main()
