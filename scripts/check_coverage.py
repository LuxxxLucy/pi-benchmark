"""Check benchmark coverage: done vs total model x dataset pairs."""
import json
import sys
from collections import defaultdict
from pathlib import Path

from .config import load_config

RESULTS_PATH = Path(__file__).parent.parent / "result" / "benchmark_results.json"
NEW_FIELDS = ("device", "batch_size", "timestamp")


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        print(f"No results file at {RESULTS_PATH}")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def main():
    cfg = load_config()
    models = cfg["models"]
    datasets = cfg["datasets"]

    # Separate gated from available
    gated_models = {k for k, v in models.items() if v.get("gated")}
    gated_datasets = {k for k, v in datasets.items() if v.get("gated")}
    avail_models = sorted(set(models) - gated_models)
    avail_datasets = sorted(set(datasets) - gated_datasets)

    results = load_results()

    # Index: (model, dataset) -> result
    done = {}
    for r in results:
        done[(r["model"], r["dataset"])] = r

    total = len(avail_models) * len(avail_datasets)
    done_pairs = {k for k in done if k[0] in avail_models and k[1] in avail_datasets}

    # --- Matrix coverage ---
    print("=== Matrix Coverage ===")
    print(f"Models: {len(avail_models)} available, {len(gated_models)} gated")
    print(f"Datasets: {len(avail_datasets)} available, {len(gated_datasets)} gated")
    print(f"Pairs done: {len(done_pairs)} / {total} ({100*len(done_pairs)/total:.1f}%)")
    print()

    # --- Per-model status ---
    print("=== Per-Model Status ===")
    for m in avail_models:
        ds_done = [d for d in avail_datasets if (m, d) in done]
        ds_missing = [d for d in avail_datasets if (m, d) not in done]
        status = f"{len(ds_done)}/{len(avail_datasets)}"
        line = f"  {m:40s} {status}"
        if ds_missing:
            line += f"  missing: {', '.join(ds_missing)}"
        print(line)
    print()

    # --- Per-dataset status ---
    print("=== Per-Dataset Status ===")
    for d in avail_datasets:
        ms_done = [m for m in avail_models if (m, d) in done]
        print(f"  {d:30s} {len(ms_done)}/{len(avail_models)} models")
    print()

    # --- Inconsistent sample counts ---
    print("=== Sample Count Consistency ===")
    ds_samples = defaultdict(set)
    for (m, d), r in done.items():
        if "samples" in r:
            ds_samples[d].add(r["samples"])
    inconsistent = {d: counts for d, counts in ds_samples.items() if len(counts) > 1}
    if inconsistent:
        for d, counts in sorted(inconsistent.items()):
            print(f"  WARNING {d}: sample counts vary: {sorted(counts)}")
    else:
        print("  All consistent.")
    print()

    # --- Missing metadata ---
    print("=== Missing Metadata ===")
    missing_meta = []
    for r in results:
        lacks = [f for f in NEW_FIELDS if f not in r or r[f] is None]
        if lacks:
            missing_meta.append((r["model"], r["dataset"], lacks))
    if missing_meta:
        print(f"  {len(missing_meta)} result(s) missing new fields:")
        for m, d, lacks in missing_meta[:20]:
            print(f"    {m:30s} x {d:20s}  missing: {', '.join(lacks)}")
        if len(missing_meta) > 20:
            print(f"    ... and {len(missing_meta) - 20} more")
    else:
        print("  All results have device, batch_size, timestamp.")


if __name__ == "__main__":
    main()
