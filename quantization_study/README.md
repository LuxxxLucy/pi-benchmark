# quantization-study

PI defender accuracy and latency under int8 quantization.

See `design.md` for the spec.

## Setup

```bash
cd quantization_study
uv sync
```

## Dry-run (Apple Silicon laptop)

Latency, one model + two precisions, ten iters:

```bash
uv run run_latency.py \
    --models testsavantai-small \
    --precisions fp32,pt_int8_dynamic \
    --measure-iters 10
```

Accuracy, one model + fp32, sample-capped:

```bash
uv run run_accuracy.py \
    --models testsavantai-small \
    --precisions fp32 \
    --max-samples 100
```

Figures from whatever JSON exists in `results/`:

```bash
uv run make_figures.py --input results/
```

## Real run

Accuracy on CUDA, full eval set:

```bash
uv run run_accuracy.py
```

Latency on CPU (any host), all models and precisions:

```bash
uv run run_latency.py
```

Then:

```bash
uv run make_figures.py --input results/
```

## Outputs

- `results/accuracy.json` — per (model, precision) F1 + per-category + FPR.
- `results/latency.json` — per (model, precision) CPU latency + speedup.
- `results/fig1_accuracy_degrade.png` — composite F1 by precision.
- `results/fig2_speedup.png` — CPU latency speedup by precision.
