#!/usr/bin/env bash
# Train-and-evaluate sweep for the top GPU-latency winners.
#
# Runs sequentially: train each model with the v2 recipe (full fine-tune +
# differential LR), then evaluate on four held-out test sets, then aggregate
# everything into one summary JSON + a markdown table.
#
# Expected wall-clock on a single RTX 3090 (bf16, bs=32, 5 epochs,
# ~24k training samples): ~60–90 min total for 5 candidates.
#
# --- Quick sanity (smoke) run first ------------------------------------
# Before the real sweep, ALWAYS run smoke mode to catch tokenizer / forward /
# save-load / eval errors in ~2 min instead of ~90 min:
#
#     SMOKE=1 bash scripts/lightweight_sweep.sh
#
# Defaults to a single model (bge-micro-v2), 200 train samples × 1 epoch,
# 100 eval samples per test set. End-to-end pipeline validation with no
# meaningful accuracy signal — pass/fail is "did it run to completion?".
#
# If smoke passes, run the full sweep:
#
#     bash scripts/lightweight_sweep.sh
#
# --- Other knobs -------------------------------------------------------
#   MODELS="bge-micro-v2,distilbert-base"  bash scripts/lightweight_sweep.sh
#   EPOCHS=8  bash scripts/lightweight_sweep.sh
#   SMOKE=1 MODELS="bert-L4-H256,bge-micro-v2"  bash scripts/lightweight_sweep.sh

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

# ── Preflight 1: BIPIA is local-only; bail early with a clear instruction. ─
BIPIA_MARKER="datasets/bipia_repo/benchmark/text_attack_train.json"
if [[ ! -f "${BIPIA_MARKER}" ]]; then
  cat >&2 <<EOF
ERROR: BIPIA training data not found at ${BIPIA_MARKER}

The v2 training recipe reads BIPIA from local disk (datasets/ is gitignored
per repo policy). Run the one-shot bootstrap script first:

    bash scripts/setup_data.sh

It clones microsoft/BIPIA (shallow, ~few MB) and warms the HuggingFace cache
for the five HF datasets used in training + eval. After that completes,
rerun this script.
EOF
  exit 2
fi

# ── Preflight 2: GPU required. Training on CPU is ~30× slower and not a ─
# test of anything this sweep cares about (we're comparing bf16 GPU
# trainability, not x86 CPU throughput). Fail loudly, not silently.
if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
  if ! uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    cat >&2 <<EOF
ERROR: CUDA is not available in the current venv.

Training on CPU would take ~30× longer (days, not hours) and the whole
point of this sweep is batch=1 GPU trainability. Fix torch first:

    uv sync   # re-resolve against the torch==2.6.0 pin in pyproject.toml

If torch still won't see the GPU after that, run:

    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

Expected: 2.6.0+cu124  True  12.4   (on the 3090 with driver CUDA ≥ 12.4)

If you really do want to run on CPU anyway (you don't), set ALLOW_CPU=1.
EOF
    exit 3
  fi
fi

DATE_TAG="$(date +%Y-%m-%d)"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RESULTS_DIR="result/lightweight_sweep_${DATE_TAG}_smoke"
else
  RESULTS_DIR="result/lightweight_sweep_${DATE_TAG}"
fi
SUMMARY_JSON="${RESULTS_DIR}/summary.json"
mkdir -p "${RESULTS_DIR}"

# ── Candidate matrix. Short name → HF id → extra train flags. ─────────
ALL_CANDIDATES=(
  "bert-L4-H256|google/bert_uncased_L-4_H-256_A-4|"
  "bge-micro-v2|TaylorAI/bge-micro-v2|"
  "minilm-L6-H384|sentence-transformers/all-MiniLM-L6-v2|"
  "bert-L4-H512|google/bert_uncased_L-4_H-512_A-8|"
  "distilbert-base|distilbert-base-uncased|"
)

# Default model filter for smoke: just bge-micro-v2 (smallest → fastest pipeline check).
SMOKE_DEFAULT_MODELS="bge-micro-v2"

# Filter candidates by MODELS env var (or SMOKE default if in smoke mode and MODELS unset).
FILTER="${MODELS:-}"
if [[ -z "${FILTER}" && "${SMOKE:-0}" == "1" ]]; then
  FILTER="${SMOKE_DEFAULT_MODELS}"
fi

if [[ -n "${FILTER}" ]]; then
  CANDIDATES=()
  while IFS= read -r line; do
    name="${line%%|*}"
    if [[ ",${FILTER}," == *",${name},"* ]]; then
      CANDIDATES+=("${line}")
    fi
  done < <(printf '%s\n' "${ALL_CANDIDATES[@]}")
else
  CANDIDATES=("${ALL_CANDIDATES[@]}")
fi

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "ERROR: MODELS filter matched 0 candidates. Check spelling against:"
  printf '  %s\n' "${ALL_CANDIDATES[@]%%|*}"
  exit 1
fi

# Training hyperparameters. Kept constant across models — we're measuring
# architectural transferability, not tuning per-model.
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR_BERT="${LR_BERT:-2e-5}"
LR_HEAD="${LR_HEAD:-1e-3}"
MAX_LENGTH="${MAX_LENGTH:-256}"

EXTRA_TRAIN_FLAGS=""
EVAL_SMOKE_FLAG=""
MODE_BANNER="[FULL RUN]"
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_TRAIN_FLAGS="--max-samples 200 --epochs 1"
  EVAL_SMOKE_FLAG="--smoke"
  EPOCHS=1
  MODE_BANNER="[SMOKE MODE — pipeline validation only, accuracy numbers are meaningless]"
fi

echo "=== lightweight sweep (${DATE_TAG}) ${MODE_BANNER} ==="
echo "results dir: ${RESULTS_DIR}"
echo "candidates:  $(printf '%s ' "${CANDIDATES[@]%%|*}")"
echo "epochs=${EPOCHS}  bs=${BATCH_SIZE}  lr_bert=${LR_BERT}  lr_head=${LR_HEAD}  max_length=${MAX_LENGTH}"
echo

# ── Per-model loop ──────────────────────────────────────────────────────
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r name hf_id extra <<< "${entry}"

  if [[ "${SMOKE:-0}" == "1" ]]; then
    save_dir="models/${name}-indirect-sweep-smoke"
  else
    save_dir="models/${name}-indirect-sweep"
  fi
  eval_json="${RESULTS_DIR}/${name}.json"

  echo "──────────────────────────────────────────────────────────────────"
  echo "▶  ${name}  (${hf_id})  ${MODE_BANNER}"
  echo "   save_dir: ${save_dir}"
  echo "   eval:     ${eval_json}"
  echo "──────────────────────────────────────────────────────────────────"

  # Device flag: cuda by default (preflight already enforced availability),
  # cpu only if user explicitly opted in.
  device_flag="cuda"
  if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
    device_flag="auto"
  fi

  # Idempotent training: skip ONLY if we have a complete checkpoint
  # (best_model.pt + config.json) that records GPU training. Missing
  # config.json means partial / interrupted training — always retrain.
  skip_train=0
  if [[ -f "${save_dir}/best_model.pt" && "${SMOKE:-0}" != "1" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    if [[ ! -f "${save_dir}/config.json" ]]; then
      echo "[force retrain] ${save_dir} has best_model.pt but no config.json — partial / interrupted ckpt, retraining"
    else
      trained_device=$(uv run python -c "
import json,sys
cfg=json.load(open('${save_dir}/config.json'))
print(cfg.get('device','unknown'))" 2>/dev/null || echo "unknown")
      if [[ "${trained_device}" == cuda* ]]; then
        skip_train=1
      else
        echo "[force retrain] ${save_dir} was trained on ${trained_device}; GPU retrain required"
      fi
    fi
  fi

  if [[ ${skip_train} -eq 1 ]]; then
    echo "[skip train] ${save_dir}/best_model.pt already present (GPU-trained)"
  else
    t0=$(date +%s)
    uv run python train_v2.py \
      --base-model "${hf_id}" \
      --save-dir   "${save_dir}" \
      --device     "${device_flag}" \
      --epochs     "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --lr-bert    "${LR_BERT}" \
      --lr-head    "${LR_HEAD}" \
      --max-length "${MAX_LENGTH}" \
      ${extra} \
      ${EXTRA_TRAIN_FLAGS}
    t1=$(date +%s)
    echo "[train] ${name} took $((t1 - t0))s"
  fi

  # Eval (always runs; cheap compared to training).
  if [[ -f "${save_dir}/best_model.pt" ]]; then
    uv run python scripts/eval_trained.py \
      --save-dir "${save_dir}" \
      --output   "${eval_json}" \
      --device   "${device_flag}" \
      ${EVAL_SMOKE_FLAG}
  else
    echo "[skip eval] no checkpoint at ${save_dir}/best_model.pt"
  fi
  echo
done

# ── Aggregate + print the markdown summary ──────────────────────────────
echo "=== aggregating ${MODE_BANNER} ==="
uv run python scripts/aggregate_sweep.py \
  --results-dir "${RESULTS_DIR}" \
  --output      "${SUMMARY_JSON}"

echo
echo "done. summary: ${SUMMARY_JSON}  (+ .md alongside)"
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo
  echo "SMOKE OK. Now rerun without SMOKE=1 for the real sweep:"
  echo "    bash scripts/lightweight_sweep.sh"
fi
