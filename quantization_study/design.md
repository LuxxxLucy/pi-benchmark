# DESIGN — Quantization-accuracy side study

## Purpose

Measure two effects of int8 quantization on three PI defender models:

1. Composite F1 degradation on the existing PI eval set.
2. CPU forward-pass latency speedup at length 256, batch 1.

The output is two figures, suitable for inclusion in a future `REPORT.md`.

## Why this study exists

The cross-runtime CPU latency bench in `../cpu_latency/bench.py` is float32 only.
That choice exists because we lack baseline numbers on how int8 quantization affects PI defender accuracy.

Published numbers (Microsoft Azure blog, HF Optimum docs) cite under 1 % GLUE drop for BERT-base dynamic int8.
GLUE is not prompt injection.
Adversarial hard-negatives in PI eval sets (jailbreaks, indirect injections, encoded attacks) may degrade disproportionately under quantization.

This study fills the gap with measurements on our actual models and our actual eval set.

## Non-goals

Static int8 quantization is out of scope.
Int4, GPTQ, AWQ, sparse-quantized inference are out of scope.
Quantization across more than three models is out of scope.
Quantization across runtimes other than PyTorch and ONNX is out of scope.

## Scope

Three PI defender models, identical to the latency bench:

| Model | Params | Architecture |
|---|---|---|
| testsavantai-small | 28.8 M | BERT-mini |
| fmops-distilbert | 66 M | DistilBERT |
| deepset-deberta-injection | 184 M | DeBERTa-v3-base |

Four precisions:

1. **fp32** baseline (PyTorch eager).
2. **bf16** via `torch.cpu.amp.autocast(dtype=torch.bfloat16)`. No calibration; lossless for the F1 metric. Only speeds up on x86 CPUs with AVX-512 BF16 or AMX. On Apple Silicon and older x86 it may run *slower* than fp32 (oneDNN may emulate or up-convert).
3. **PyTorch int8 dynamic** via `torch.quantization.quantize_dynamic` on `nn.Linear`.
4. **ONNX int8 dynamic** via `onnxruntime.quantization.quantize_dynamic`, applied after the ORT transformer optimizer pass.

Accuracy eval set: the existing PI composite from `../../REPORT.md` §3.
Three injection-relevant categories: Direct PI (7 datasets), Indirect PI (2 datasets), Jailbreak (4 datasets).
Plus three benign-prompt datasets for FPR (notinject, false-reject, or-bench).

Latency benchmark: length 256, batch 1, ten warmup iterations, one hundred measured iterations.
Higher iteration count than the latency bench to tighten the ratio estimate.

Accuracy hardware: CUDA (GPU box).
Latency hardware: CPU.

## File layout

```
benchmark_impl/quantization_study/
├── design.md                this file
├── README.md                how to run + interpret
├── run_accuracy.py          CUDA eval — produces results/accuracy.json
├── run_latency.py           CPU eval — produces results/latency.json
├── make_figures.py          reads both JSONs; emits two figures
└── results/
    ├── accuracy.json
    ├── latency.json
    ├── fig1_accuracy_degrade.png
    └── fig2_speedup.png
```

## Accuracy methodology (`run_accuracy.py`)

For each (model, precision) pair:

1. Load the model in the target precision:
   - fp32: standard HF load on CUDA.
   - `pt_int8_dynamic`: load fp32, then `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`.
     Set `torch.backends.quantized.engine = "x86"` on Linux, `"qnnpack"` on Apple Silicon.
     Quantized inference runs on CPU since PyTorch dynamic int8 is CPU-only.
   - `onnx_int8_dynamic`: load the pre-converted artifact from `../cpu_latency/conversions.py` and run via ORT `CPUExecutionProvider`.
2. Run the full PI eval set, batched (batch 32 on GPU; batch 1 on CPU for the int8 precisions).
3. Compute per-category F1 at the vendor-chosen threshold (matches the leaderboard methodology in `../../REPORT.md` §3.1, footnote 3).
4. Composite F1 = equal-weighted mean of {Direct F1, Indirect F1, Jailbreak F1}.
5. Maximum FPR across the three benign-prompt datasets.
6. Emit one row per (model, precision):

```json
{
  "model": "fmops-distilbert",
  "precision": "pt_int8_dynamic",
  "composite_f1": 0.685,
  "direct_f1": 0.860,
  "indirect_f1": 0.624,
  "jailbreak_f1": 0.591,
  "max_fpr_3": 0.717,
  "delta_composite_f1_vs_fp32": -0.006,
  "delta_direct_f1_vs_fp32": -0.004,
  "delta_indirect_f1_vs_fp32": -0.006,
  "delta_jailbreak_f1_vs_fp32": -0.007
}
```

Dataset loaders are re-imported from `../scripts/datasets.py` (the existing PI bench's loader module).
No duplicate dataset definitions.

## Latency methodology (`run_latency.py`)

Identical timing harness to `../cpu_latency/bench.py`: ten warmup + one hundred measured forwards, batch 1, single-thread plus best-thread, gc disabled inside the measurement loop.

Three precisions, three models, one length (256), measured on CPU.

Output JSON shape:

```json
{
  "platform": "...",
  "results": [
    {
      "model": "fmops-distilbert",
      "precision": "fp32",
      "num_threads": 4,
      "p50_ms": 42.6,
      "p95_ms": 44.1,
      "mean_ms": 42.9,
      "samples": [...]
    },
    {
      "model": "fmops-distilbert",
      "precision": "pt_int8_dynamic",
      "p50_ms": 22.1,
      "speedup_vs_fp32": 1.93,
      ...
    }
  ]
}
```

`speedup_vs_fp32` is `p50_fp32 / p50_precision` and is computed only when both rows exist.

## Quantization recipes (verified APIs)

**PyTorch int8 dynamic**:

```python
import torch
import torch.quantization

torch.backends.quantized.engine = "x86"  # or "qnnpack" on Apple Silicon
model_q = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
```

Only `nn.Linear` modules quantized.
Activations are quantized dynamically per-forward.
No calibration set needed.

**ONNX int8 dynamic**:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model_opt.onnx",      # post-ORT-optimizer model
    model_output="model.int8.onnx",
    weight_type=QuantType.QInt8,
)
```

Applied after the ORT transformer optimizer pass (`onnxruntime.transformers.optimizer`).
Operator-level scales chosen by ORT at runtime per batch.

## Figure contracts (`make_figures.py`)

Both figures plot the three models on the x-axis, with one bar group per model and three bars per group (one per precision).

**`fig1_accuracy_degrade.png`** — grouped bar chart of composite F1.

- y-axis: composite F1, range 0.0 to 1.0.
- bars: fp32 (light grey), pt_int8_dynamic (mid blue), onnx_int8_dynamic (dark blue).
- annotation: ΔF1 vs fp32 written above each non-fp32 bar with sign.
- secondary panel (right): per-category F1 grouped bars (Direct, Indirect, Jailbreak), one panel per model, showing where the drop concentrates.
- title: "PI-defender composite F1 by precision".
- caption: "Composite F1 is the equal-weighted mean of Direct, Indirect, and Jailbreak F1, evaluated at the vendor-chosen threshold."

**`fig2_speedup.png`** — grouped bar chart of CPU latency speedup.

- y-axis: speedup factor = `p50_fp32 / p50_precision`.
- fp32 bar height is 1.0 by definition (drawn as the reference).
- pt_int8_dynamic and onnx_int8_dynamic bars show the multiplicative speedup.
- annotation: the raw P50 (in milliseconds) written above each bar.
- title: "PI-defender CPU latency speedup at length 256 by precision".

Both figures use matplotlib only (no seaborn), 150 DPI, 5 × 4 inch default size.
Side-by-side layout produces a 10 × 4 figure suitable for a single-row inclusion in `REPORT.md`.

## Dry-run protocol

Three smoke tests, in order.

Latency, single model, single precision, on Mac:

```bash
uv run run_latency.py \
    --models testsavantai-small \
    --precisions fp32,pt_int8_dynamic \
    --measure-iters 10
```

Accuracy, single model, fp32 only, sample-capped, on Mac CPU:

```bash
uv run run_accuracy.py \
    --models testsavantai-small \
    --precisions fp32 \
    --max-samples 100
```

(Note: `--max-samples` only valid during dry-run.
Per the [No sample capping] memory rule, the real run uses the full eval set.)

Figure pipeline, on either:

```bash
uv run make_figures.py --input results/
```

After all three smoke tests pass on Mac, the full run is delegated to your CUDA box for accuracy and to whichever CPU host is canonical for the latency numbers.

## Decision rule

After the study:

- If max ΔF1 across (model, precision) is ≤ 0.01 absolute (one F1 point):
  quantization is "near-free" for our PI defenders.
  The cross-runtime latency bench can be re-extended with quantized cells, annotated with each ΔF1.

- If max ΔF1 > 0.01:
  quantization is a real accuracy cost for our PI defenders.
  The cross-runtime latency bench stays fp32-only.
  The two figures from this study are the canonical quantization datapoint for any future report.

- If max ΔF1 > 0.05:
  quantization is unacceptable.
  This becomes a research note: "dynamic int8 cannot be used on these PI defenders without a custom calibration recipe or QAT."

## Open implementation questions

These will be resolved during dry-run by inspection:

1. Does dynamic int8 on DeBERTa-v3 work end-to-end through `quantize_dynamic`, given the disentangled attention sub-modules contain `nn.Linear` instances buried under custom relative-position layers?
2. Does ONNX int8 dynamic survive the ORT transformer optimizer pass for DeBERTa-v3 (the optimizer's BERT-attention fusion does not match DeBERTa)?
3. What batch size on the CUDA box maximizes throughput for the accuracy eval without OOM on the 184 M DeBERTa model?
