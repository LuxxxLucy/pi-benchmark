"""
Prompt injection detection benchmark runner.
Config-driven, with composable pipelines and structured report.
"""
import json
import argparse
from pathlib import Path

import torch
import pandas as pd

from .config import load_config
from .classifiers import build_classifier
from .datasets import load_dataset_samples
from .evaluation import run_classifier
from .report import save_results_json, generate_summary, generate_report

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"


def load_existing_results() -> list[dict]:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return []


def main():
    p = argparse.ArgumentParser(description="PI detection benchmark")
    p.add_argument("--models", nargs="+", help="Models to run (from config.yaml)")
    p.add_argument("--datasets", nargs="+", help="Datasets to run (from config.yaml)")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--force", action="store_true", help="Re-run existing results")
    p.add_argument("--only-new", action="store_true", help="Only run new combos")
    p.add_argument("--report-only", action="store_true", help="Regenerate report from existing results")
    args = p.parse_args()

    cfg = load_config()
    all_models = cfg["models"]
    all_datasets = cfg["datasets"]

    # Build dataset category map for report grouping
    ds_categories = {name: ds_cfg.get("category", "other") for name, ds_cfg in all_datasets.items()}

    # Report-only mode: regenerate from existing JSON
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
    new_results = []

    for ds_name in datasets_to_run:
        if ds_name not in all_datasets:
            print(f"Unknown dataset: {ds_name}")
            continue

        ds_cfg = all_datasets[ds_name]
        samples = None  # lazy load

        for model_name in models_to_run:
            if model_name not in all_models:
                print(f"Unknown model: {model_name}")
                continue

            if not args.force and (model_name, ds_name) in existing_keys:
                print(f"  [skip] {model_name} × {ds_name} (already in results)")
                continue

            # Lazy load dataset
            if samples is None:
                print(f"\nLoading {ds_cfg['desc']}...")
                samples = load_dataset_samples(ds_name, ds_cfg)
                if args.max_samples:
                    import random
                    rng = random.Random(42)
                    rng.shuffle(samples)
                    samples = samples[:args.max_samples]
                n_inj = sum(1 for s in samples if s["is_injection"])
                print(f"  {len(samples)} samples ({n_inj} inj, {len(samples) - n_inj} safe)")

            # Build and run classifier
            model_cfg = all_models[model_name]
            try:
                clf = build_classifier(model_name, model_cfg, all_models)
            except Exception as e:
                print(f"  [fail] {model_name}: {e}")
                continue

            result = run_classifier(model_name, clf, samples, ds_name, model_cfg["params"])
            new_results.append(result.to_dict())

            # Free HF model memory
            if model_cfg.get("type") == "hf":
                del clf
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    # Merge and save
    if new_results:
        new_keys = {(r["model"], r["dataset"]) for r in new_results}
        merged = [r for r in existing if (r["model"], r["dataset"]) not in new_keys]
        merged.extend(new_results)
        save_results_json(merged, RESULTS_JSON)
        df = pd.DataFrame(merged)
        generate_summary(df)

        generate_report(df, ds_categories)
    else:
        print("\nNo new results to save.")


if __name__ == "__main__":
    main()
