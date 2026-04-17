"""Shared primitives for latency benchmarks under `benchmark_impl/`.

Used by both:
- `benchmark_impl/latency_bench.py` (CPU, local `models/<subdir>/` weights)
- `benchmark_impl/latency_cuda/src/bench.py` (CUDA, HF-id weights)

Kept dep-light on purpose (torch + stdlib) so downstream venvs can import
this module via `sys.path` without inheriting either tool's dep set.
"""
from __future__ import annotations

from pathlib import Path


FILL_TEXT = (
    "The capital of France is Paris, a city known for its art, fashion, "
    "gastronomy, and culture. Photosynthesis converts light energy to chemical "
    "energy in plant cells. Modern distributed systems must balance "
    "consistency, availability, and partition tolerance under network failure. "
)


def percentile(xs, p: float):
    """Linear-interpolation percentile. Returns None on empty input."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def make_input(tokenizer, target_length: int) -> dict:
    """Tokenize FILL_TEXT (looped if needed) to exactly `target_length` tokens.

    Returns a dict of cpu tensors; caller moves to device if needed.

    Adds explicit `position_ids = arange(L)` — some custom-modeling-code
    models (Alibaba-NLP/gte-base-en-v1.5 NewModel, nomic_bert) fail
    non-deterministically when the caller relies on the model to build
    position_ids internally; the arange buffer is safe to pass and is
    accepted-or-ignored by every standard HF encoder.
    """
    import torch

    text = FILL_TEXT
    while len(tokenizer(text, add_special_tokens=True)["input_ids"]) < target_length:
        text = text + FILL_TEXT
    enc = tokenizer(
        text, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=target_length, padding=False,
    )
    enc["position_ids"] = torch.arange(enc["input_ids"].shape[1]).unsqueeze(0)
    return enc


def get_max_pos(model) -> int:
    """Return the model's max input length, with a 512 default for models
    that don't declare one."""
    cfg = model.config
    for attr in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 512


def get_disk_size_mb(model_dir: Path) -> float:
    return sum(p.stat().st_size for p in Path(model_dir).rglob("*") if p.is_file()) / (1024 * 1024)
