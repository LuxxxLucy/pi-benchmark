# pi-benchmark

Prompt-injection classifier evaluation + latency benchmarking.

## Tools

- **`scripts/run_benchmark.py`** — accuracy pipeline across classifiers × datasets. See `config.yaml` for the matrix. Requires the full environment (`uv sync` at the repo root).
- **`latency_cuda/`** — self-contained **latency** benchmark for Linux + CUDA: 15 lightweight classifier candidates (4 families, 4.4M–86M params) across a token-length sweep. Has its own uv project so the Ubuntu/CUDA setup is just `cd latency_cuda && uv sync && bash run.sh`. See `latency_cuda/README.md`.

## Shared code

`bench_common.py` at the repo root holds the primitives shared between the CPU latency workflow (`latency_bench.py`) and the new CUDA one (`latency_cuda/src/bench.py`) — `FILL_TEXT`, `percentile`, `make_input`, `get_max_pos`, `get_disk_size_mb`.
