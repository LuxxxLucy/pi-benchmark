"""CPU latency benchmark for the three precisions at length 256, batch 1.

Per (model, precision): 10 warmup + 100 measured forwards. fp32 row writes
first so non-fp32 rows can compute speedup_vs_fp32.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import platform
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from bench_common import FILL_TEXT, percentile  # noqa: E402

from _adapters import (  # noqa: E402
    make_adapter, model_dir_for, model_short_names, precision_names,
)

LENGTH = 256
WARMUP = 10
MEASURE = 100


def _make_input(tokenizer, target_length: int) -> dict:
    text = FILL_TEXT
    enc = tokenizer(
        text, return_tensors="np", truncation=True, padding="max_length",
        max_length=target_length,
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _percentile_stats(samples_ms: list[float]) -> dict:
    return {
        "min_ms": min(samples_ms),
        "p50_ms": percentile(samples_ms, 50),
        "p95_ms": percentile(samples_ms, 95),
        "mean_ms": mean(samples_ms),
        "stdev_ms": stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "max_ms": max(samples_ms),
    }


def bench_one(model: str, precision: str, num_threads: int,
              warmup: int, measure: int) -> dict:
    print(f"\n=== {model} / {precision} (threads={num_threads}) ===")
    try:
        adapter = make_adapter(precision, model_dir_for(model), num_threads)
    except Exception as e:
        print(f"  load failed: {e}")
        return {"model": model, "precision": precision, "error": repr(e)}

    enc = _make_input(adapter.tokenizer, LENGTH)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

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

    stats = _percentile_stats(samples)
    row = {
        "model": model,
        "precision": precision,
        "num_threads": num_threads,
        "length": LENGTH,
        "warmup_iters": warmup,
        "measure_iters": measure,
        **stats,
        "samples_ms": samples,
    }
    print(f"  p50={stats['p50_ms']:7.2f}ms  p95={stats['p95_ms']:7.2f}ms  "
          f"mean={stats['mean_ms']:7.2f}ms")
    del adapter
    gc.collect()
    return row


def attach_speedup(rows: list[dict]) -> None:
    fp32_by_model = {r["model"]: r.get("p50_ms")
                     for r in rows if r["precision"] == "fp32" and "p50_ms" in r}
    for r in rows:
        if "p50_ms" not in r or r["precision"] == "fp32":
            continue
        base = fp32_by_model.get(r["model"])
        if base and r["p50_ms"] > 0:
            r["speedup_vs_fp32"] = base / r["p50_ms"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(model_short_names()))
    ap.add_argument("--precisions", default=",".join(precision_names()))
    ap.add_argument("--threads", type=int,
                    default=max(1, multiprocessing.cpu_count() // 2))
    ap.add_argument("--warmup-iters", type=int, default=WARMUP)
    ap.add_argument("--measure-iters", type=int, default=MEASURE)
    ap.add_argument("--out", default="results/latency.json")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "num_threads": args.threads,
        "length": LENGTH,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
    }
    print(f"env: {json.dumps(env, indent=2)}")

    rows: list[dict] = []
    for model in models:
        for precision in precisions:
            rows.append(bench_one(
                model, precision, args.threads,
                args.warmup_iters, args.measure_iters,
            ))
            out_path.write_text(json.dumps({"env": env, "results": rows}, indent=2))

    attach_speedup(rows)
    out_path.write_text(json.dumps({"env": env, "results": rows}, indent=2))

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
