# DESIGN — CPU latency benchmark across nine fp32 runtimes

## Purpose

Measure forward-pass latency of three prompt-injection defenders across nine CPU inference runtimes, all at float32 precision, on x86 Linux and Apple Silicon.

The benchmark answers one question: at fixed precision, how much wall-time variation comes from the runtime choice alone?

## Non-goals

Quantization is out of scope.
It lives in `quantization_study/`.

GPU is out of scope.
The CUDA bench lives in `latency_cuda/`.

Accuracy is out of scope.
It lives in the parent project's `REPORT.md` and in `quantization_study/`.

## Scope

Three PI defender models, picked as the best leak-adjusted composite F1 in each size band from `../../REPORT.md` §3.

| Model | Params | Architecture |
|---|---|---|
| testsavantai-small | 28.8 M | BERT-mini (4-layer) |
| fmops-distilbert | 66 M | DistilBERT |
| deepset-deberta-injection | 184 M | DeBERTa-v3-base |

Four lengths: 64, 128, 256, 512.

Length 1024 is dropped because every picked model has `max_position_embeddings = 512`.
Extending beyond 512 requires a long-context model (e.g. ModernBERT-150M) and lives in a future scope.

Float32 precision across all runtimes.
Batch size 1.
Ten warmup iterations, fifty measured iterations.
Statistics: min, P50, P95, mean, max, plus raw samples.

Threading: report both `intra=1` (single-thread) and `intra=physical_cores` (best-thread).
Inter-op threads always 1.

Cooldown ten seconds between models.

Total cells: 9 runtimes × 3 models × 4 lengths = 108.
Some cells will report "unsupported" per the runtime-support matrix in §Caveats.

## Runtimes

| # | Runtime | Platform | Conversion |
|---|---|---|---|
| 1 | PyTorch eager fp32 | any | none |
| 2 | PyTorch + `torch.compile` (Inductor) | any | none |
| 3 | ONNX Runtime fp32 + transformer optimizer | any | `optimum-cli export onnx` → `onnxruntime.transformers.optimizer` |
| 4 | CTranslate2 fp32 | any | `ct2-transformers-converter` |
| 5 | llama.cpp BERT fp32 (`llama-cpp-python` embedding API + Python classifier head) | any | `convert_hf_to_gguf.py --outtype f32` |
| 6 | TensorFlow Lite + XNNPACK fp32 | any | `ai-edge-torch` direct PT→TFLite |
| 7 | ExecuTorch + XNNPACK fp32 | any | `torch.export.export` then `EdgeProgramManager.to_executorch()` then XNNPACK partitioner |
| 8 | Intel Extension for PyTorch (IPEX) fp32 + oneDNN fusion | Linux x86 | `ipex.optimize(model)` in process |
| 9 | OpenVINO fp32 | Linux x86 | `optimum-cli export openvino` |

## File layout

The entire CPU latency bench lives under `benchmark_impl/cpu_latency/`.
Source HF weights stay one level up at `benchmark_impl/models/`, shared with the rest of the bench project.

```
benchmark_impl/
├── models/                       existing — source HF weights (shared)
├── bench_common.py               existing — shared primitives (percentile, make_input)
└── cpu_latency/
    ├── design.md                 this file
    ├── pyproject.toml            own uv project (per the one-uv-per-folder rule)
    ├── bench.py                  entry point, sweep runner
    ├── conversions.py            fetch + convert + cache
    ├── adapter_pytorch.py        runtime adapter (one file per runtime)
    ├── adapter_compile.py
    ├── adapter_onnx.py
    ├── adapter_ctranslate2.py
    ├── adapter_llamacpp.py
    ├── adapter_tflite.py
    ├── adapter_executorch.py     in-process ExecuTorch runtime + XNNPACK backend
    ├── adapter_ipex.py           Linux x86 only
    ├── adapter_openvino.py       Linux x86 only; --mode latency|throughput
    ├── models_converted/         per-runtime cached weights
    │   ├── <model>/{onnx_opt,ct2,gguf_f32,tflite,pte,ov}/
    │   └── _conversion.log.json
    └── result/                   JSON outputs
```

`bench_common.py` is imported via `sys.path` from the parent directory, matching the pattern in `latency_cuda/src/bench.py`.

## Conversion pipeline (`conversions.py`)

One-shot, idempotent.
Cache layout is keyed by `(model_name, target_format, source_sha)`.

Python API:

```python
def ensure_converted(model_name: str, target_format: str) -> Path:
    """Returns the path to the cached converted artifact.

    target_format ∈ {"hf", "onnx_opt", "ct2", "gguf_f32", "tflite", "ov"}.
    Returns the cached path if it exists and the source HF weights hash
    matches the logged entry. Otherwise runs the conversion and updates
    _conversion.log.json.
    """
```

CLI for batch:

```bash
uv run conversions.py --models all --formats all
uv run conversions.py --models fmops-distilbert --formats onnx_opt,gguf_f32
```

Per-format conversion commands:

| Target | Command |
|---|---|
| `onnx_opt` | `optimum-cli export onnx --task text-classification --model <src> <dst>` then `python -m onnxruntime.transformers.optimizer --input <dst>/model.onnx --output <dst>/model_opt.onnx --model_type bert --num_heads <h> --hidden_size <H>` |
| `ct2` | `ct2-transformers-converter --model <src> --output_dir <dst>` |
| `gguf_f32` | `python llama.cpp/convert_hf_to_gguf.py <src> --outfile <dst>/model.gguf --outtype f32` |
| `tflite` | `python -c "import ai_edge_torch; ..."` (Python script; see Caveats for DeBERTa) |
| `pte` | `torch.export.export(model, args)` → `to_edge()` → `to_backend(XnnpackPartitioner)` → `to_executorch()` |
| `ov` | `optimum-cli export openvino --task text-classification --model <src> <dst>` |

The conversion log captures the command, the source `config.json` SHA, the runtime versions of the converter tools, and the wall-time spent.

## Adapter interface

Every `adapter_*.py` exposes one class:

```python
class Adapter:
    name: str                           # e.g. "pytorch_fp32", "onnx_fp32_opt"

    @classmethod
    def is_supported(cls) -> str | None: ...   # None if supported; else short reason

    def __init__(self, artifact: Path, model_short: str, num_threads: int): ...
    def forward(self, input_ids, attention_mask) -> np.ndarray: ...   # (1, num_labels)
    def cleanup(self) -> None: ...      # optional
```

`is_supported()` returns `None` on platforms / installs that can run this runtime, otherwise a short string explaining the reason.
The harness uses the string verbatim as the `error` field of the unsupported cell.

All adapters are in-process; none requires a subprocess.
`llama.cpp` is wrapped via `llama-cpp-python` bindings, `ExecuTorch` via its Python runtime.
The harness times the wall-clock around `forward()` directly and inlines its own warmup loop.

## JSON output schema

Extends the existing benchmark JSON shape with a `runtime` field.

```json
{
  "platform": "Darwin-24.6.0-arm64-arm-64bit",
  "torch": "2.5.1",
  "runtime_versions": {
    "onnxruntime": "1.20.1",
    "ctranslate2": "4.5.0",
    "llama_cpp_python": "0.3.2",
    "tflite_runtime": "2.17.0"
  },
  "results": [
    {
      "runtime": "onnx_fp32_opt",
      "model": "fmops-distilbert",
      "params_total": 66956290,
      "disk_mb": 256.0,
      "max_position_embeddings": 512,
      "device": "cpu",
      "num_threads_intra": 4,
      "num_threads_inter": 1,
      "batch_size": 1,
      "warmup_iters": 10,
      "measure_iters": 50,
      "load_time_s": 0.42,
      "lengths": {
        "256": {
          "target_length": 256,
          "actual_length": 256,
          "min": 12.34,
          "p50": 13.10,
          "p95": 14.20,
          "mean": 13.22,
          "max": 17.55,
          "samples": [12.34, 12.41, ...]
        }
      }
    }
  ]
}
```

Cells where the runtime cannot run the (model, length) pair emit:

```json
{"runtime": "tflite_fp32", "model": "deepset-deberta-injection",
 "error": "unsupported: ai-edge-torch converter rejects disentangled attention"}
```

This keeps `analyze.py` compatible — missing length keys are treated as no-data, error strings are surfaced in the markdown table as a dash.

## Threading discipline

Single-thread (`intra=1, inter=1`) and best-thread (`intra=physical_cores, inter=1`) are both reported per cell.

Runtime-specific thread setters, all called before model load:

| Runtime | Setter |
|---|---|
| PyTorch | `torch.set_num_threads(n)` + `torch.set_num_interop_threads(1)` |
| `torch.compile` | same as PyTorch |
| ORT | `SessionOptions.intra_op_num_threads = n`, `inter_op_num_threads = 1` |
| CT2 | `Translator(intra_threads=n, inter_threads=1)` |
| llama.cpp | `Llama(..., n_threads=n)` |
| TFLite | `Interpreter(num_threads=n)` |
| OpenVINO | `compile_model(config={"INFERENCE_NUM_THREADS": n, "PERFORMANCE_HINT": <LATENCY|THROUGHPUT>})` |
| IPEX | `torch.set_num_threads(n)` |
| ExecuTorch | `Module(model_path, num_threads=n)` (XNNPACK respects this) |

Garbage collection disabled inside the measurement loop.
`gc.collect()` once before warmup, `gc.disable()` for the inner loop, `gc.enable()` after.

## CLI entry point

All commands run from inside `cpu_latency/`.

```bash
# Full sweep
uv run bench.py

# Subset
uv run bench.py \
    --runtimes pytorch_fp32,onnx_fp32_opt \
    --models fmops-distilbert \
    --lengths 256

# Single-thread only
uv run bench.py --threading single

# OpenVINO throughput mode (other runtimes ignore the flag)
uv run bench.py --runtimes openvino_fp32 --openvino-mode throughput

# NUMA discipline (multi-socket Xeon server)
numactl --cpunodebind=0 --membind=0 uv run bench.py

# Dry-run / smoke test
uv run bench.py \
    --runtimes pytorch_fp32 --models testsavantai-small \
    --lengths 64 --measure-iters 5
```

Default output path: `result/latency_cpu_<platform>.json` (relative to `cpu_latency/`).

## Caveats

| Cell | Expected outcome |
|---|---|
| deepset-deberta-injection × llama.cpp | unsupported — gguf converter does not handle disentangled attention |
| deepset-deberta-injection × TFLite | likely unsupported — `ai-edge-torch` converter may reject disentangled attention |
| deepset-deberta-injection × ExecuTorch | likely unsupported — `torch.export` may reject dynamic shapes in disentangled attention |
| any × IPEX | unsupported on Apple Silicon — platform check |
| any × OpenVINO | unsupported on Apple Silicon — platform check |

Unsupported cells emit the error JSON above; the run does not abort.

## NUMA discipline

On multi-socket x86 servers (e.g. dual-socket Xeon), threads spread across NUMA nodes hurt latency.
Pin to one node before running:

```bash
numactl --cpunodebind=0 --membind=0 uv run bench.py
```

ONNX Runtime CPU EP NUMA-pinning thread placement is automatic when `--cpunodebind` constrains the process.
Reported gains on transformer encoder workloads: ~20 % on a dual-socket Xeon (ONNX Runtime official docs).
Single-socket hosts and Apple Silicon ignore this concern.

## Dry-run protocol

Before any full sweep, validate the harness end-to-end on the smallest cross-section:

```bash
uv run bench.py \
    --runtimes pytorch_fp32 \
    --models testsavantai-small \
    --lengths 64 \
    --measure-iters 5
```

Confirms: model loads, JSON schema is well-formed, `analyze.py` ingests it without error.
Then expand to one runtime per family, then full sweep.

## Open implementation questions

These will be resolved during P0 (conversion pipeline) by inspection:

1. Does `ai-edge-torch` reject DeBERTa-v3 at conversion time or fail silently at inference?
2. Does `torch.export` survive DeBERTa-v3's disentangled attention for the ExecuTorch path?
3. How does OpenVINO's `LATENCY` hint compare to setting `INFERENCE_NUM_THREADS` manually for a batch=1 workload?
