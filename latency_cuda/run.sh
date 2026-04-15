#!/usr/bin/env bash
# Lightweight-classifier latency bench on Linux + CUDA.
#
# Usage:
#   1. uv sync                                  # first time only
#   2. uv run python scripts/download.py        # pre-fetch all weights
#   3. bash run.sh                              # uncommented combos run
#
# Outputs: one JSON under results/ per invocation.

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LENGTHS="32 64 128 256 512 1024 2048 4096"
# LENGTHS="32 64 128 256 512"       # shorter — fast smoke
# LENGTHS="32 64 128 256 512 1024 2048 4096 8192"  # push past BERT's 512 cap

# --- All candidates (PI-trained + arch-baselines) in one sweep ---
uv run python src/bench.py --lengths $LENGTHS

# --- Or split by group (separate output files) ---
# uv run python src/bench.py --models pi-trained    --lengths $LENGTHS \
#                             --out results/latency_cuda_pi-trained.json
# uv run python src/bench.py --models arch-baseline --lengths $LENGTHS \
#                             --out results/latency_cuda_arch-baseline.json

# --- Ad-hoc subsets — edit this list to pick specific models ---
# uv run python src/bench.py \
#     --models bert-tiny,minilm-L6-H384,fmops-distilbert,protectai-deberta-v2 \
#     --lengths $LENGTHS

# --- CPU comparison pass (same machine) ---
# uv run python src/bench.py --device cpu --lengths $LENGTHS \
#                             --out results/latency_cpu.json

echo "[run] done. Results in $(pwd)/results/"
