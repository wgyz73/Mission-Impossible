"""Loaders for the MISSION IMPOSSIBLE benchmark.

Two subsets are supported:

  * ``full_noop`` — single-instruction no-ops. Each *image* has prompts in
    seven categories with up to five prompt variants per category. We emit
    one row per ``(image, task, prompt_idx)`` tuple to make iteration
    straightforward.
  * ``partial_noop`` — composite instructions of the form
    ``"<feasible>. Then, <no-op>."``. Each line of the JSONL file is one
    sample.

Both loaders work either against a local checkout of the dataset
directory or against a HuggingFace Hub copy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


# ---------------------------------------------------------------------------
# Sample types
# ---------------------------------------------------------------------------
@dataclass
class FullNoOpSample:
    image_id: str
    pool_id: str
    image_path: str   # relative path under <data_root>; reconstruct images first
    task: str         # one of 7 no-op categories
    prompt_idx: int   # within the category
    prompt: str

    @property
    def key(self) -> str:
        return f"{self.image_id}__{self.task}__{self.prompt_idx}"


@dataclass
class PartialNoOpSample:
    sample_id: str
    image_path: str
    target_path: str
    feasible_clause: str
    no_op_clause: str
    no_op_category: str
    combined_instruction: str
    source_dataset: str

    @property
    def key(self) -> str:
        return self.sample_id


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _resolve_local(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_full_noop(prompts_path: str | Path) -> list[FullNoOpSample]:
    """Read ``prompts.json`` and expand into ``(image, task, prompt_idx)`` rows."""
    p = _resolve_local(prompts_path)
    with open(p) as f:
        entries = json.load(f)
    rows: list[FullNoOpSample] = []
    for e in entries:
        for task, prompts in (e.get("tasks") or {}).items():
            for i, prompt in enumerate(prompts):
                rows.append(FullNoOpSample(
                    image_id=e["image_id"],
                    pool_id=e["pool_id"],
                    image_path=e["image_path"],
                    task=task,
                    prompt_idx=i,
                    prompt=prompt,
                ))
    return rows


def load_partial_noop(jsonl_path: str | Path) -> list[PartialNoOpSample]:
    """Read ``partial_test.jsonl`` (one record per line)."""
    p = _resolve_local(jsonl_path)
    rows: list[PartialNoOpSample] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(PartialNoOpSample(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                target_path=r["target_path"],
                feasible_clause=r["feasible_clause"],
                no_op_clause=r["no_op_clause"],
                no_op_category=r["no_op_category"],
                combined_instruction=r["combined_instruction"],
                source_dataset=r.get("source_dataset", ""),
            ))
    return rows


def iter_samples(rows: Iterable, batch_size: int = 1) -> Iterator[list]:
    """Generic batcher; works for either sample type."""
    batch: list = []
    for r in rows:
        batch.append(r)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
