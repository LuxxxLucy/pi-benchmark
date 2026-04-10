#!/bin/bash
# Run all non-gated HF models on each dataset, organized by dataset size (small first).
# This gives complete cross-model results per dataset before moving to the next.
# The Python runner skips already-completed (model, dataset) pairs automatically.
#
# Skipped (run separately due to size):
#   xTRam1 (~3K test / ~10K full), xxz224 (~6.7K test / ~22K full), jayavibhav (65K)
set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] Will print commands without executing."
    echo
fi

COMMON="--device mps --batch-size 16"

run_dataset() {
    local ds="$1"
    echo "  -> $ds"
    local cmd="PYTHONUNBUFFERED=1 uv run python -m scripts.run_benchmark --datasets $ds $COMMON"
    if $DRY_RUN; then
        echo "     $cmd"
    else
        PYTHONUNBUFFERED=1 uv run python -m scripts.run_benchmark --datasets "$ds" $COMMON
    fi
}

# --- Small datasets (< 1K samples) ---
echo "========================================"
echo "  GROUP 1: Small datasets (< 1K)"
echo "========================================"
for ds in deepset-all jailbreakbench bipia notinject lakera-gandalf do-not-answer; do
    run_dataset "$ds"
done

# --- Medium datasets (1K - 5K samples) ---
echo
echo "========================================"
echo "  GROUP 2: Medium datasets (1K - 5K)"
echo "========================================"
for ds in false-reject in-the-wild-jailbreak wildguardtest neuralchemy semantic-router-jailbreak protectai-validation; do
    run_dataset "$ds"
done

# --- Large datasets (5K, capped from bigger) ---
echo
echo "========================================"
echo "  GROUP 3: Large datasets (5K capped)"
echo "========================================"
for ds in spml-chatbot lakera-mosscap necent-multilingual jailbreakhub aegis-safety toxic-chat or-bench beavertails; do
    run_dataset "$ds"
done

# --- Regenerate report ---
echo
echo "========================================"
echo "  Regenerating report"
echo "========================================"
if $DRY_RUN; then
    echo "  uv run python -m scripts.run_benchmark --report-only"
else
    uv run python -m scripts.run_benchmark --report-only
fi

echo
echo "=== ALL DATASET GROUPS DONE ==="
