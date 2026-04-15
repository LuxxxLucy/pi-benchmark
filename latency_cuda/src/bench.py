"""Latency benchmark for lightweight classifiers on Linux + CUDA.

Per model, per length:
  - 10 warmup forwards (no timing)
  - 50 measured forwards with torch.cuda.synchronize() + perf_counter_ns
  - batch_size=1, dtype=bfloat16 on CUDA / float32 on CPU
  - torch.no_grad()

Auto-downloads weights from HF on first run. Filters lengths > model's
max_position_embeddings automatically.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Share primitives with benchmark_impl/latency_bench.py (CPU sibling).
# bench_common.py has no heavy deps beyond tokenizers (pulled in via
# transformers anyway) — safe to import across venvs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from bench_common import FILL_TEXT, get_max_pos, make_input, percentile  # noqa: E402

from models import CANDIDATES, ModelSpec, by_name  # noqa: E402


DEFAULT_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
WARMUP_ITERS = 10
MEASURE_ITERS = 50
COOLDOWN_S = 2


def pick_device(override: str | None) -> str:
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def make_input_on_device(tokenizer, target_length: int, device: str) -> dict:
    enc = make_input(tokenizer, target_length)
    return {k: v.to(device) for k, v in enc.items()}


def bench_one(spec: ModelSpec, device: str, lengths: list[int],
              warmup: int, measure: int) -> dict:
    print(f"\n=== {spec.name}  ({spec.hf_id})  [{spec.family} / {spec.group}] ===")
    dtype = pick_dtype(device)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id, dtype=dtype,
    ).to(device).eval()
    load_s = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    max_pos = get_max_pos(model)
    print(f"  loaded {load_s:.1f}s — {n_params/1e6:.1f}M params, max_pos={max_pos}, "
          f"dtype={dtype}")

    valid = [L for L in lengths if L <= max_pos]
    if len(valid) != len(lengths):
        skip = [L for L in lengths if L > max_pos]
        print(f"  skipping lengths > max_pos={max_pos}: {skip}")

    per_length: dict[str, dict] = {}
    for L in valid:
        enc = make_input_on_device(tokenizer, L, device)
        actual = int(enc["input_ids"].shape[1])

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(**enc)
            sync(device)

            gc.collect()
            gc.disable()
            lats_ns: list[int] = []
            try:
                for _ in range(measure):
                    sync(device)
                    t = time.perf_counter_ns()
                    _ = model(**enc)
                    sync(device)
                    lats_ns.append(time.perf_counter_ns() - t)
            finally:
                gc.enable()

        lats_ms = [x / 1e6 for x in lats_ns]
        stat = {
            "target_length": L,
            "actual_length": actual,
            "min_ms": min(lats_ms),
            "p50_ms": percentile(lats_ms, 50),
            "p95_ms": percentile(lats_ms, 95),
            "p99_ms": percentile(lats_ms, 99),
            "mean_ms": mean(lats_ms),
            "stdev_ms": stdev(lats_ms) if len(lats_ms) > 1 else 0.0,
            "max_ms": max(lats_ms),
            "throughput_rps": 1000.0 / mean(lats_ms) if mean(lats_ms) > 0 else 0.0,
        }
        per_length[str(L)] = stat
        print(f"  len={L:>5} (actual {actual:>5}): "
              f"p50={stat['p50_ms']:7.2f}ms  p95={stat['p95_ms']:7.2f}ms  "
              f"mean={stat['mean_ms']:7.2f}ms  thr={stat['throughput_rps']:6.1f} rps")

    out = {
        "name": spec.name,
        "hf_id": spec.hf_id,
        "family": spec.family,
        "group": spec.group,
        "note": spec.note,
        "params_M": n_params / 1e6,
        "max_position_embeddings": max_pos,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "load_time_s": load_s,
        "batch_size": 1,
        "warmup_iters": warmup,
        "measure_iters": measure,
        "lengths": per_length,
    }

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="cuda | cpu (default: cuda if available, else cpu)")
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                    help=f"Input token lengths (default: {DEFAULT_LENGTHS})")
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--measure", type=int, default=MEASURE_ITERS)
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_S,
                    help="Seconds between models (lets GPU settle).")
    ap.add_argument("--models", default="all",
                    help="Comma-separated short names (e.g. "
                         "'bert-tiny,minilm-L6-H384,fmops-distilbert'), "
                         "'all', 'pi-trained', or 'arch-baseline'.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: results/latency_<device>.json).")
    args = ap.parse_args()

    device = pick_device(args.device)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.models == "all":
        picks = list(CANDIDATES)
    elif args.models in ("pi-trained", "arch-baseline"):
        picks = [m for m in CANDIDATES if m.group == args.models]
    else:
        picks = [by_name(n.strip()) for n in args.models.split(",") if n.strip()]

    out_path = Path(args.out) if args.out else Path("results") / f"latency_{device}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device == "cuda" else None,
        "device": device,
        "gpu_name": (torch.cuda.get_device_name(0) if device == "cuda" else None),
        "dtype": str(pick_dtype(device)).replace("torch.", ""),
        "lengths": args.lengths,
        "warmup_iters": args.warmup,
        "measure_iters": args.measure,
    }
    print(f"env: {json.dumps(env, indent=2)}")

    all_out = {"env": env, "results": []}
    t0 = time.perf_counter()
    for i, spec in enumerate(picks):
        if i > 0 and args.cooldown > 0:
            time.sleep(args.cooldown)
        try:
            all_out["results"].append(
                bench_one(spec, device, args.lengths, args.warmup, args.measure)
            )
        except Exception as e:
            print(f"FAIL {spec.name}: {e}")
            import traceback; traceback.print_exc()
            all_out["results"].append({"name": spec.name, "hf_id": spec.hf_id,
                                       "error": repr(e)})
        out_path.write_text(json.dumps(all_out, indent=2))  # incremental save

    all_out["total_time_s"] = time.perf_counter() - t0
    out_path.write_text(json.dumps(all_out, indent=2))
    print(f"\nWrote {out_path} (total {all_out['total_time_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
