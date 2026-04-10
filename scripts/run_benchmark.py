"""
Prompt injection detection benchmark runner.
Config-driven, with composable pipelines and structured report.

Key design: model-outer loop — each model loads once, runs all datasets, then frees memory.
Results save incrementally after each model×dataset combo.

Accuracy only — latency profiling is a separate pass (see latency_profile.py).
"""
import json
import argparse
import time
from pathlib import Path

import torch
import pandas as pd

from .config import load_config
from .classifiers import build_classifier
from .datasets import load_dataset_samples
from .evaluation import run_classifier
from .report import save_results_json, generate_summary, generate_report

RESULTS_DIR = Path(__file__).parent.parent / "result"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"


def load_existing_results() -> list[dict]:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return []


def _save_incremental(all_results: list[dict]):
    """Save results after each combo for crash recovery."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results_json(all_results, RESULTS_JSON)


def main():
    p = argparse.ArgumentParser(description="PI detection benchmark (accuracy)")
    p.add_argument("--models", nargs="+", help="Models to run (from config.yaml)")
    p.add_argument("--datasets", nargs="+", help="Datasets to run (from config.yaml)")
    p.add_argument("--force", action="store_true", help="Re-run existing results")
    p.add_argument("--report-only", action="store_true", help="Regenerate report from existing results")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force re-download and re-cache all datasets")
    p.add_argument("--device", default="auto", choices=["cpu", "mps", "auto"],
                   help="Device for inference (default: auto = MPS if available)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for inference (default: 16)")
    args = p.parse_args()

    cfg = load_config()
    all_models = cfg["models"]
    all_datasets = cfg["datasets"]

    ds_categories = {name: ds_cfg.get("category", "other") for name, ds_cfg in all_datasets.items()}

    # Resolve device
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Report-only mode
    if args.report_only:
        existing = load_existing_results()
        if not existing:
            print("No existing results to report on.")
            return
        df = pd.DataFrame(existing)
        generate_summary(df)
        generate_report(df, ds_categories)
        return

    models_to_run = args.models or [k for k, v in all_models.items() if not v.get("gated")]
    datasets_to_run = args.datasets or [k for k, v in all_datasets.items() if not v.get("gated")]

    existing = load_existing_results()
    existing_keys = {(r["model"], r["dataset"]) for r in existing}
    all_results = list(existing)

    # Pre-load and cache datasets
    dataset_cache: dict[str, list[dict]] = {}

    # Progress tracking
    total_combos = len(models_to_run) * len(datasets_to_run)
    done = 0
    skipped = 0
    new_count = 0
    t_start = time.time()

    # Model-outer loop: load each model once, run all datasets
    for mi, model_name in enumerate(models_to_run):
        if model_name not in all_models:
            print(f"Unknown model: {model_name}")
            continue

        model_cfg = all_models[model_name]

        # Check if all datasets already done for this model
        pending_ds = [ds for ds in datasets_to_run
                      if ds in all_datasets and (args.force or (model_name, ds) not in existing_keys)]
        if not pending_ds:
            skipped += len(datasets_to_run)
            done += len(datasets_to_run)
            print(f"\n[{mi+1}/{len(models_to_run)}] {model_name}: all datasets done, skipping")
            continue

        # Load model once
        print(f"\n{'='*60}")
        print(f"[{mi+1}/{len(models_to_run)}] Loading model: {model_name} ({model_cfg.get('params', '?')})")
        print(f"  Device: {device}, Batch size: {args.batch_size}")
        try:
            clf = build_classifier(model_name, model_cfg, all_models,
                                   device=device, batch_size=args.batch_size)
        except Exception as e:
            print(f"  [FAIL] Cannot load {model_name}: {e}")
            done += len(datasets_to_run)
            continue

        # Run all datasets with this model
        for ds_name in datasets_to_run:
            done += 1
            if ds_name not in all_datasets:
                continue

            if not args.force and (model_name, ds_name) in existing_keys:
                continue

            # Load dataset (cached in memory across models, on disk across runs)
            if ds_name not in dataset_cache:
                ds_cfg = all_datasets[ds_name]
                try:
                    samples = load_dataset_samples(ds_name, ds_cfg,
                                                   rebuild_cache=args.rebuild_cache)
                    dataset_cache[ds_name] = samples
                except Exception as e:
                    print(f"  [FAIL] Cannot load dataset {ds_name}: {e}")
                    continue

            samples = dataset_cache[ds_name]
            n_inj = sum(1 for s in samples if s["is_injection"])

            # Progress + ETA
            elapsed = time.time() - t_start
            rate = (new_count / elapsed) if elapsed > 0 and new_count > 0 else 0
            remaining_combos = total_combos - done
            eta = f", ETA: {remaining_combos/rate:.0f}s" if rate > 0 else ""
            print(f"\n  [{done}/{total_combos}] {model_name} x {ds_name} "
                  f"({len(samples)} samples, {n_inj} inj){eta}")

            # Run
            result = run_classifier(model_name, clf, samples, ds_name,
                                    model_cfg.get("params", "?"),
                                    device=device, batch_size=args.batch_size)
            result_dict = result.to_dict()

            # Update results: replace if exists, append if new
            new_key = (result_dict["model"], result_dict["dataset"])
            all_results = [r for r in all_results if (r["model"], r["dataset"]) != new_key]
            all_results.append(result_dict)
            existing_keys.add(new_key)
            new_count += 1

            # Incremental save
            _save_incremental(all_results)

        # Free model memory
        del clf
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()

    # Final report
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. {new_count} new results in {elapsed_total:.0f}s ({skipped} skipped)")

    if new_count > 0:
        df = pd.DataFrame(all_results)
        generate_summary(df)
        generate_report(df, ds_categories)
    else:
        print("No new results to report.")


if __name__ == "__main__":
    main()
