#!/usr/bin/env bash
# One-shot bootstrap for training + eval datasets.
#
# Required locally (not in any HF hub):
#   - microsoft/BIPIA        → cloned into datasets/bipia_repo/
#
# Required via HF `load_dataset` (auto-downloaded on first call, cached under
# ~/.cache/huggingface). This script optionally warms the cache so the real
# training run doesn't block on downloads mid-epoch:
#   - xxz224/prompt-injection-attack-dataset
#   - jayavibhav/prompt-injection
#   - deepset/prompt-injections
#   - xTRam1/safe-guard-prompt-injection
#   - bowen-uchicago/notinject  (eval-only FPR probe)
#
# Usage:
#   bash scripts/setup_data.sh              # clone BIPIA + warm HF cache
#   SKIP_HF=1 bash scripts/setup_data.sh    # only clone BIPIA
#
# Idempotent — safe to rerun.

set -euo pipefail
cd "$(dirname "$0")/.."

BIPIA_DIR="datasets/bipia_repo"
BIPIA_MARKER="${BIPIA_DIR}/benchmark/text_attack_train.json"

# ── 1. BIPIA ────────────────────────────────────────────────────────────
if [[ -f "${BIPIA_MARKER}" ]]; then
  echo "[bipia] already present at ${BIPIA_DIR}"
else
  echo "[bipia] cloning microsoft/BIPIA (shallow)..."
  mkdir -p datasets
  # Use HTTPS so hosts without SSH keys configured still work. Shallow depth
  # keeps the download small (~few MB).
  git clone --depth 1 https://github.com/microsoft/BIPIA.git "${BIPIA_DIR}"
fi

# Sanity check the files train_v2.py + eval_trained.py actually read.
required=(
  "${BIPIA_DIR}/benchmark/text_attack_train.json"
  "${BIPIA_DIR}/benchmark/text_attack_test.json"
  "${BIPIA_DIR}/benchmark/code_attack_train.json"
  "${BIPIA_DIR}/benchmark/code_attack_test.json"
  "${BIPIA_DIR}/benchmark/email/train.jsonl"
  "${BIPIA_DIR}/benchmark/email/test.jsonl"
  "${BIPIA_DIR}/benchmark/code/train.jsonl"
  "${BIPIA_DIR}/benchmark/code/test.jsonl"
  "${BIPIA_DIR}/benchmark/table/train.jsonl"
  "${BIPIA_DIR}/benchmark/table/test.jsonl"
)
missing=0
for f in "${required[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "  [missing] ${f}" >&2
    missing=1
  fi
done
if (( missing )); then
  echo "ERROR: BIPIA clone is missing required files. Delete ${BIPIA_DIR} and rerun." >&2
  exit 1
fi
echo "[bipia] OK — all 10 required files present"

# ── 2. HuggingFace datasets (optional warm) ─────────────────────────────
if [[ "${SKIP_HF:-0}" == "1" ]]; then
  echo "[hf] skipped (SKIP_HF=1)"
  exit 0
fi

echo "[hf] warming dataset cache (one-time; ~2-5 min on first run)..."
uv run python - <<'PY'
from datasets import load_dataset

HF_DATASETS = [
    "xxz224/prompt-injection-attack-dataset",
    "jayavibhav/prompt-injection",
    "deepset/prompt-injections",
    "xTRam1/safe-guard-prompt-injection",
    "bowen-uchicago/notinject",
]
for ds_id in HF_DATASETS:
    print(f"  loading {ds_id} ...", flush=True)
    try:
        load_dataset(ds_id)
        print(f"  [ok] {ds_id}")
    except Exception as exc:
        print(f"  [warn] {ds_id}: {exc}")
print("[hf] done.")
PY

echo
echo "setup complete. Next step:"
echo "    SMOKE=1 bash scripts/lightweight_sweep.sh     # ~2 min pipeline check"
echo "    bash scripts/lightweight_sweep.sh             # ~60-90 min full sweep"
