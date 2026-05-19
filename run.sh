#!/usr/bin/env bash
# End-to-end PI-defender benchmark runner.
# All output goes to stdout. On the host:
#
#     ./run.sh 2>&1 | tee run.log
#
# Stages:
#   1) cpu_latency: convert artifacts (idempotent; cached) + full sweep
#   2) quantization_study: CPU latency at length 256 (fp32 / bf16 / pt-int8 / onnx-int8)
#   3) quantization_study: GPU accuracy across PI datasets (needs CUDA)
#   4) quantization_study: figures (fig1 = accuracy degrade, fig2 = speedup)
#
# Skip knobs (env vars set to 1):
#   SKIP_CPU_LATENCY     skip stage 1
#   SKIP_QUANT_LATENCY   skip stage 2
#   SKIP_QUANT_ACC       skip stage 3 (set this if no CUDA available)
#   SKIP_FIGURES         skip stage 4
#
# Passthrough knobs:
#   CPU_LATENCY_ARGS     extra args appended to cpu_latency/bench.py
#   QUANT_LATENCY_ARGS   extra args appended to quantization_study/run_latency.py
#   QUANT_ACC_ARGS       extra args appended to quantization_study/run_accuracy.py
#
# Optional env for stage 1 llama.cpp gguf conversion:
#   LLAMA_CPP_CONVERT    path to convert_hf_to_gguf.py (or set LLAMA_CPP_ROOT)
set -eu -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Auto-point Python TLS at the system CA bundle (curl + uv already trust it,
# but huggingface_hub/requests defaults to certifi's bundle which omits the
# corporate proxy CAs that the system trusts).
if [ -z "${REQUESTS_CA_BUNDLE:-}" ]; then
    for p in /etc/ssl/certs/ca-certificates.crt \
             /etc/pki/tls/certs/ca-bundle.crt \
             /etc/ssl/cert.pem \
             /usr/local/etc/openssl/cert.pem; do
        if [ -f "$p" ]; then
            export REQUESTS_CA_BUNDLE="$p"
            export SSL_CERT_FILE="$p"
            export CURL_CA_BUNDLE="$p"
            echo "# CA bundle: $p"
            break
        fi
    done
fi

banner() {
    printf '\n================================================================\n'
    printf '%s\n' "$*"
    printf '================================================================\n'
}

banner "host info"
uname -a
date -u
echo "ROOT=$ROOT"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "no nvidia-smi on PATH"

if [ "${SKIP_CPU_LATENCY:-0}" != "1" ]; then
    banner "[1/4] cpu_latency: conversions"
    cd "$ROOT/cpu_latency"
    uv sync
    uv run python conversions.py --models all --formats all

    banner "[1/4] cpu_latency: bench sweep"
    uv run python bench.py ${CPU_LATENCY_ARGS:-}
else
    echo "skip stage 1 (SKIP_CPU_LATENCY=1)"
fi

if [ "${SKIP_QUANT_LATENCY:-0}" != "1" ]; then
    banner "[2/4] quantization_study: CPU latency"
    cd "$ROOT/quantization_study"
    uv sync
    uv run python run_latency.py ${QUANT_LATENCY_ARGS:-}
else
    echo "skip stage 2 (SKIP_QUANT_LATENCY=1)"
fi

if [ "${SKIP_QUANT_ACC:-0}" != "1" ]; then
    banner "[3/4] quantization_study: accuracy (CUDA recommended)"
    cd "$ROOT/quantization_study"
    uv sync
    uv run python run_accuracy.py ${QUANT_ACC_ARGS:-}
else
    echo "skip stage 3 (SKIP_QUANT_ACC=1)"
fi

if [ "${SKIP_FIGURES:-0}" != "1" ]; then
    banner "[4/4] quantization_study: figures"
    cd "$ROOT/quantization_study"
    uv run python make_figures.py
else
    echo "skip stage 4 (SKIP_FIGURES=1)"
fi

banner "DONE"
date -u
