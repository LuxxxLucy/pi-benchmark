"""CPU latency benchmark across nine fp32 runtimes.

Per (runtime, model, length, threading): warmup + measured forwards at batch=1, fp32.
Records min/p50/p95/mean/max + raw samples. Skips unsupported cells gracefully.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from bench_common import FILL_TEXT, get_disk_size_mb, percentile  # noqa: E402

import adapter_openvino  # noqa: E402
from conversions import ensure_converted, ensure_hf_local, model_short_names  # noqa: E402
from runtimes import RUNTIMES, runtime_names  # noqa: E402


DEFAULT_LENGTHS = [64, 128, 256, 512]
DEFAULT_WARMUP = 10
DEFAULT_MEASURE = 50
DEFAULT_COOLDOWN = 10


def _physical_cores() -> int:
    n = psutil.cpu_count(logical=False)
    return max(1, n or 1)


_META_CACHE: dict[str, dict] = {}


def _model_meta(model_short: str) -> dict:
    if model_short in _META_CACHE:
        return _META_CACHE[model_short]
    hf = ensure_hf_local(model_short)
    cfg = json.loads((hf / "config.json").read_text())
    from transformers import AutoModelForSequenceClassification
    pt = AutoModelForSequenceClassification.from_pretrained(str(hf))
    meta = {
        "params_total": sum(p.numel() for p in pt.parameters()),
        "disk_mb": get_disk_size_mb(hf),
        "max_position_embeddings": int(cfg.get("max_position_embeddings", 512)),
    }
    del pt
    gc.collect()
    _META_CACHE[model_short] = meta
    return meta


def _make_input(tokenizer, target_length: int) -> dict:
    enc = tokenizer(
        FILL_TEXT, return_tensors="np", truncation=True,
        padding="max_length", max_length=target_length,
    )
    return {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }


def _stats(samples_ms: list[float]) -> dict:
    return {
        "min": min(samples_ms),
        "p50": percentile(samples_ms, 50),
        "p95": percentile(samples_ms, 95),
        "mean": mean(samples_ms),
        "stdev": stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "max": max(samples_ms),
        "samples": samples_ms,
    }


def _measure(adapter, input_ids, attention_mask, warmup: int, measure: int) -> dict:
    for _ in range(warmup):
        adapter.forward(input_ids, attention_mask)
    gc.collect()
    gc.disable()
    samples: list[float] = []
    try:
        for _ in range(measure):
            t = time.perf_counter_ns()
            adapter.forward(input_ids, attention_mask)
            samples.append((time.perf_counter_ns() - t) / 1e6)
    finally:
        gc.enable()
    return _stats(samples)


def bench_cell(runtime: str, model: str, lengths: list[int],
               num_threads: int, warmup: int, measure: int) -> dict:
    cls, fmt = RUNTIMES[runtime]
    unsupported = cls.is_supported()
    if unsupported:
        return {"runtime": runtime, "model": model, "num_threads_intra": num_threads,
                "error": f"unsupported: {unsupported}"}

    meta = _model_meta(model)
    try:
        artifact = ensure_converted(model, fmt)
    except Exception as e:
        return {"runtime": runtime, "model": model, "num_threads_intra": num_threads,
                **meta, "error": f"conversion failed ({fmt}): {e!r}"}

    t0 = time.perf_counter()
    try:
        adapter = cls(artifact, model, num_threads)
    except Exception as e:
        return {"runtime": runtime, "model": model, "num_threads_intra": num_threads,
                **meta, "error": f"load failed: {e!r}"}
    load_time_s = time.perf_counter() - t0

    valid = [L for L in lengths if L <= meta["max_position_embeddings"]]
    row = {
        "runtime": runtime, "model": model,
        "device": "cpu", "num_threads_intra": num_threads, "num_threads_inter": 1,
        "batch_size": 1, "warmup_iters": warmup, "measure_iters": measure,
        "load_time_s": load_time_s,
        **meta,
        "lengths": {},
    }
    print(f"\n=== {runtime} / {model}  (threads={num_threads}) ===")
    for L in valid:
        enc = _make_input(adapter.tokenizer, L)
        try:
            stats = _measure(adapter, enc["input_ids"], enc["attention_mask"],
                             warmup, measure)
        except Exception as e:
            row["lengths"][str(L)] = {"target_length": L, "actual_length": L,
                                      "error": repr(e)}
            print(f"  len={L:>4}: FAILED  {e!r}")
            continue
        row["lengths"][str(L)] = {"target_length": L, "actual_length": L, **stats}
        print(f"  len={L:>4}: p50={stats['p50']:7.2f}ms  p95={stats['p95']:7.2f}ms  "
              f"mean={stats['mean']:7.2f}ms")
    if hasattr(adapter, "cleanup"):
        adapter.cleanup()
    return row


def runtime_versions() -> dict:
    versions = {"torch": torch.__version__}
    for mod, key in [("onnxruntime", "onnxruntime"),
                     ("ctranslate2", "ctranslate2"),
                     ("llama_cpp", "llama_cpp_python"),
                     ("tflite_runtime", "tflite_runtime"),
                     ("executorch", "executorch"),
                     ("openvino", "openvino"),
                     ("intel_extension_for_pytorch", "ipex")]:
        try:
            m = __import__(mod)
            versions[key] = getattr(m, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtimes", default="all")
    ap.add_argument("--models", default="all")
    ap.add_argument("--lengths", default=",".join(str(L) for L in DEFAULT_LENGTHS))
    ap.add_argument("--threads", type=int, default=_physical_cores())
    ap.add_argument("--threading", choices=["single", "sweep"], default="sweep",
                    help="single: --threads only. sweep: both 1 and --threads.")
    ap.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--measure-iters", type=int, default=DEFAULT_MEASURE)
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    ap.add_argument("--openvino-mode", choices=["latency", "throughput"],
                    default="latency")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    rts = runtime_names() if args.runtimes == "all" else args.runtimes.split(",")
    mdls = model_short_names() if args.models == "all" else args.models.split(",")
    lengths = [int(x) for x in args.lengths.split(",")]
    thread_set = [args.threads] if args.threading == "single" else sorted({1, args.threads})

    adapter_openvino.set_mode(args.openvino_mode)

    out_path = Path(args.out) if args.out else (
        ROOT / "result" / f"latency_cpu_{platform.machine()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    versions = runtime_versions()
    header = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": versions["torch"],
        "runtime_versions": versions,
        "physical_cores": _physical_cores(),
        "logical_cores": psutil.cpu_count(logical=True),
        "thread_set": thread_set,
        "openvino_mode": args.openvino_mode,
        "lengths": lengths,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
    }
    print(json.dumps(header, indent=2))

    rows: list[dict] = []
    for ri, runtime in enumerate(rts):
        if ri > 0 and args.cooldown > 0:
            print(f"\ncooldown {args.cooldown}s...")
            time.sleep(args.cooldown)
        for model in mdls:
            for nt in thread_set:
                row = bench_cell(runtime, model, lengths, nt,
                                 args.warmup_iters, args.measure_iters)
                rows.append(row)
                out_path.write_text(json.dumps({**header, "results": rows}, indent=2))

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
