# latency_cuda — lightweight classifier latency on Linux + CUDA

Measures batch=1 forward-pass latency across token lengths for a curated list of **lightweight transformer classifiers** — both PI-trained deployed models and pure architecture baselines (random head) — so you can see the latency-per-arch landscape independent of task-specific weights.

## Quickstart

```bash
uv sync                                 # install
uv run python scripts/download.py       # pre-fetch all weights into HF cache
bash run.sh                             # sweep all models × all lengths
```

Output lands in `results/latency_cuda.json` (one entry per model, nested per length).

## Model list

Edited in `src/models.py`. Two groups:

| Group | What | Use for |
|---|---|---|
| `pi-trained` | Real prompt-injection classifiers with trained heads (testsavantai, protectai, fmops, deepset, …) | End-to-end production latency |
| `arch-baseline` | Open backbones with random classifier heads (bert-tiny, MiniLM-L6/L12, DeBERTa-v3 xsmall/small/base, DistilBERT) | Architecture-level latency, independent of task weights |

Params range from **4.4M** (`bert-tiny`) to **86M** (`deberta-v3-base`).

## Length sweep

Default: `32 64 128 256 512 1024 2048 4096`. Models with `max_position_embeddings < L` skip that length automatically (most BERT-class models cap at 512).

## Selecting a subset

```bash
uv run python src/bench.py --models bert-tiny,minilm-L6-H384,fmops-distilbert \
                            --lengths 64 128 256 512
uv run python src/bench.py --models pi-trained
uv run python src/bench.py --models arch-baseline
```

## Output schema

```json
{
  "env": {"device": "cuda", "gpu_name": "...", "torch": "...", "cuda": "...",
          "lengths": [...], "warmup_iters": 10, "measure_iters": 50},
  "results": [
    {
      "name": "minilm-L6-H384", "hf_id": "...", "family": "minilm",
      "group": "arch-baseline", "params_M": 22.7,
      "max_position_embeddings": 512, "device": "cuda", "dtype": "bfloat16",
      "lengths": {
        "32":  {"p50_ms": ..., "p95_ms": ..., "mean_ms": ..., "throughput_rps": ...},
        "64":  {...},
        ...
      }
    }
  ]
}
```

## Notes

- CUDA `bfloat16` by default; falls back to `float32` on CPU.
- `torch.cuda.synchronize()` around every measured forward — timings do not under-report.
- `torch.no_grad()`, gc disabled in measure loop.
- Arch-baseline accuracy is meaningless (random classifier head). Only latency is real.

## Code layout & sharing

Primitives (`percentile`, `make_input`, `get_max_pos`, `FILL_TEXT`) live in `../bench_common.py` and are shared with the sibling `benchmark_impl/latency_bench.py` (the pre-existing CPU bench). `src/bench.py` imports them via `sys.path`, since the two tools live in separate uv venvs. New benchmark scripts in this repo should do the same: one leaf script per measurement kind, `bench_common.py` for shared helpers.

**Extending**:

- **New model** → append a `ModelSpec(...)` line to `src/models.py`.
- **New input source** (e.g. real prompts from a HF dataset) → add a sibling of `make_input(...)` in `../bench_common.py`; give `bench.py` an `--input-source` flag.
- **New benchmark** (accuracy, memory, throughput-batched) → new leaf script alongside `bench.py`, reuse `bench_common` + `models`.
