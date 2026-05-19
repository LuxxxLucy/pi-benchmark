# cpu_latency

CPU latency benchmark across nine fp32 inference runtimes for three PI defender models.

See `design.md` for the spec.

## Setup

```bash
cd cpu_latency
uv sync
```

Optional runtime extras (each runtime that needs heavy deps is opt-in):

```bash
uv sync --extra llamacpp        # llama-cpp-python
uv sync --extra tflite          # ai-edge-torch + tflite-runtime
uv sync --extra executorch      # ExecuTorch edge runtime
uv sync --extra openvino        # OpenVINO + optimum-intel
uv sync --all-extras            # everything
```

## Dry-run (Apple Silicon)

```bash
uv run bench.py \
    --runtimes pytorch_fp32 \
    --models testsavantai-small \
    --lengths 64 \
    --measure-iters 5
```

## Full sweep

```bash
uv run bench.py
```

## Per-format conversions (one-shot)

```bash
uv run conversions.py --models all --formats all
```

Idempotent. Re-runs only re-convert when the source weights have changed.

## NUMA discipline (multi-socket x86 servers)

On dual-socket Xeon hosts, pin the process to one NUMA node before benchmarking.
Threads spread across nodes hurt latency by ~20 % on transformer encoders.

```bash
numactl --cpunodebind=0 --membind=0 uv run bench.py
```

Single-socket servers and Apple Silicon ignore this concern.

## OpenVINO mode

The OpenVINO adapter accepts `--openvino-mode {latency,throughput}`.
For batch=1 latency-sensitive work, use `latency` (default).
For throughput-oriented batch serving, use `throughput`.

## Output

`result/latency_cpu_<platform>.json` with one row per (runtime, model) and per-length stats.

Unsupported cells (runtime cannot run the model, or platform-restricted) emit an `error` field instead of `lengths`.
