#!/usr/bin/env bash
# Lightweight-classifier latency bench on Linux + CUDA.
#
# Outputs: one JSON under results/ per invocation (latency_cuda.json, latency_cpu.json).

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Step 0: sync deps. Cheap no-op if the venv is already in sync; required
# after any pyproject.toml change. Skipping this was the root cause of the
# bert-tiny/bert-mini tokenizer-convert failure in the 2026-04-16 run.
uv sync

# Step 1: pre-fetch HF weights so mid-bench downloads don't pollute load_time_s.
# Idempotent — skips files already in the HF cache.
uv run python scripts/download.py

LENGTHS="32 64 128 256 512 1024 2048 4096"
# LENGTHS="32 64 128 256 512"       # shorter — fast smoke
# LENGTHS="32 64 128 256 512 1024 2048 4096 8192"  # push past BERT's 512 cap

# --- CUDA pass (default on Ubuntu/CUDA) ---
uv run python src/bench.py --device cuda --lengths $LENGTHS \
                            --out results/latency_cuda.json

# --- CPU pass (same machine; PyTorch uses threading-controlled CPU kernels) ---
uv run python src/bench.py --device cpu  --lengths $LENGTHS \
                            --out results/latency_cpu.json

# --- Or split by group (separate output files) ---
# uv run python src/bench.py --device cuda --models pi-trained    --lengths $LENGTHS \
#                             --out results/latency_cuda_pi-trained.json
# uv run python src/bench.py --device cuda --models arch-baseline --lengths $LENGTHS \
#                             --out results/latency_cuda_arch-baseline.json

# --- Ad-hoc subsets — edit the --models list to pick specific candidates ---
# uv run python src/bench.py --device cuda \
#     --models bert-tiny,minilm-L6-H384,fmops-distilbert,protectai-deberta-v2 \
#     --lengths $LENGTHS

echo "[run] done. Results in $(pwd)/results/"
