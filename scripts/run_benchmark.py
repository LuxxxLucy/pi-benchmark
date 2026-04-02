"""
Prompt injection detection benchmark runner.
Evaluates classifier models on standard datasets.
Outputs results as JSON + markdown summary.
"""
import json
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset


MODELS = {
    # Smallest / fastest first
    "fmops-distilbert": {
        "hf_id": "Fmops/distilbert-prompt-injection",
        "params": "66M",
        "injection_labels": [1],
    },
    "protectai-base-v1": {
        "hf_id": "protectai/deberta-v3-base-prompt-injection",
        "params": "86M",
        "injection_labels": [1],
    },
    "protectai-base-v2": {
        "hf_id": "protectai/deberta-v3-base-prompt-injection-v2",
        "params": "86M",
        "injection_labels": [1],
    },
    "vsr-jailbreak-modernbert": {
        "hf_id": "LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model",
        "params": "149M",
        "injection_labels": [1],
    },
    "vsr-jailbreak-mmbert32k-merged": {
        "hf_id": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
        "params": "307M",
        "injection_labels": [1],
    },
    # Gated models (require HF access approval)
    "prompt-guard-22m": {
        "hf_id": "meta-llama/Llama-Prompt-Guard-2-22M",
        "params": "22M",
        "injection_labels": [1, 2],
        "gated": True,
    },
    "prompt-guard-86m": {
        "hf_id": "meta-llama/Llama-Prompt-Guard-2-86M",
        "params": "86M",
        "injection_labels": [1, 2],
        "gated": True,
    },
    "protectai-small-v2": {
        "hf_id": "protectai/deberta-v3-small-prompt-injection-v2",
        "params": "44M",
        "injection_labels": [1],
        "gated": True,
    },
}

RESULTS_DIR = Path(__file__).parent.parent / "results"


@dataclass
class ModelResult:
    model_name: str
    model_params: str
    dataset_name: str
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    latencies_ms: list = field(default_factory=list)
    errors: int = 0

    @property
    def precision(self):
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self):
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self):
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def accuracy(self):
        correct = self.true_positives + self.true_negatives
        return correct / self.total_samples if self.total_samples > 0 else 0.0

    def latency_stats(self):
        if not self.latencies_ms:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}
        s = sorted(self.latencies_ms)
        n = len(s)
        return {
            "p50": s[n // 2],
            "p95": s[int(n * 0.95)],
            "p99": s[int(n * 0.99)],
            "mean": sum(s) / n,
        }

    def summary_dict(self):
        lat = self.latency_stats()
        return {
            "model": self.model_name,
            "params": self.model_params,
            "dataset": self.dataset_name,
            "samples": self.total_samples,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "latency_p50_ms": round(lat["p50"], 2),
            "latency_p95_ms": round(lat["p95"], 2),
            "latency_mean_ms": round(lat["mean"], 2),
            "errors": self.errors,
        }


def _load_binary_dataset(hf_id, text_col, label_col, injection_label, split=None):
    """Load a dataset with binary labels."""
    if split:
        ds = load_dataset(hf_id, split=split)
        all_rows = list(ds)
    else:
        ds = load_dataset(hf_id)
        all_rows = []
        for s in ds:
            all_rows.extend(ds[s])
    samples = []
    for row in all_rows:
        text = row[text_col]
        label = row[label_col]
        samples.append({"text": text, "is_injection": label == injection_label})
    return samples


def _load_jailbreakbench():
    """Load JailbreakBench harmful/benign behaviors."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    samples = []
    for row in ds["harmful"]:
        samples.append({"text": row["Goal"], "is_injection": True})
    for row in ds["benign"]:
        samples.append({"text": row["Goal"], "is_injection": False})
    return samples


DATASETS = {
    "deepset-all": {
        "desc": "deepset/prompt-injections (all splits)",
        "loader": lambda: _load_binary_dataset("deepset/prompt-injections", text_col="text", label_col="label", injection_label=1),
    },
    "xTRam1": {
        "desc": "xTRam1/safe-guard-prompt-injection (10K+)",
        "loader": lambda: _load_binary_dataset("xTRam1/safe-guard-prompt-injection", text_col="text", label_col="label", injection_label=1),
    },
    "jailbreakbench": {
        "desc": "JailbreakBench/JBB-Behaviors (harmful vs benign goals)",
        "loader": _load_jailbreakbench,
    },
}


INJECTION_LABEL_NAMES = {"injection", "jailbreak", "unsafe", "malicious", "harmful"}
SAFE_LABEL_NAMES = {"safe", "benign", "normal", "legitimate"}


def classify_single(pipe, text, injection_labels, max_length=512):
    """Classify a single text. Returns (predicted_injection: bool, latency_ms: float)."""
    truncated = text[:2000]  # rough truncation before tokenizer
    t0 = time.perf_counter()
    try:
        result = pipe(truncated, truncation=True, max_length=max_length)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        predicted_label = result[0]["label"]

        if isinstance(predicted_label, str):
            lower = predicted_label.lower()
            # Check text labels first
            if lower in INJECTION_LABEL_NAMES:
                return True, elapsed_ms
            if lower in SAFE_LABEL_NAMES:
                return False, elapsed_ms
            # Try LABEL_N format
            if predicted_label.startswith("LABEL_"):
                label_idx = int(predicted_label.split("_")[1])
                return label_idx in injection_labels, elapsed_ms
            # Unknown label — fallback to safe
            return False, elapsed_ms
        else:
            return int(predicted_label) in injection_labels, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Error classifying: {e}", file=sys.stderr)
        return False, elapsed_ms


def _load_model(hf_id):
    """Load model, handling _orig_mod. prefix from torch.compile()."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_id)

    # Download raw weights and check for _orig_mod. prefix
    try:
        weight_file = hf_hub_download(hf_id, "model.safetensors")
        raw_sd = load_file(weight_file)
    except Exception:
        weight_file = hf_hub_download(hf_id, "pytorch_model.bin")
        raw_sd = torch.load(weight_file, map_location="cpu", weights_only=True)

    needs_strip = any(k.startswith("_orig_mod.") for k in raw_sd)
    if needs_strip:
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        print(f"  (stripping _orig_mod. prefix from {len(raw_sd)} keys)")
    else:
        clean_sd = raw_sd

    # Create model and load cleaned weights
    model = AutoModelForSequenceClassification.from_config(config)
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    model.eval()
    return model


def run_model_on_dataset(model_name, model_config, samples, dataset_name):
    """Run a single model on a dataset and return ModelResult."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_config['params']})")
    print(f"Dataset: {dataset_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    hf_id = model_config["hf_id"]
    injection_labels = model_config["injection_labels"]

    print(f"Loading {hf_id}...")
    try:
        model = _load_model(hf_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            dtype=torch.float32,
        )
    except Exception as e:
        print(f"  FAILED to load model: {e}", file=sys.stderr)
        result = ModelResult(
            model_name=model_name,
            model_params=model_config["params"],
            dataset_name=dataset_name,
            errors=len(samples),
        )
        return result

    result = ModelResult(
        model_name=model_name,
        model_params=model_config["params"],
        dataset_name=dataset_name,
    )

    for i, sample in enumerate(samples):
        text = sample["text"]
        is_injection_gt = sample["is_injection"]

        predicted_injection, latency_ms = classify_single(pipe, text, injection_labels)
        result.latencies_ms.append(latency_ms)
        result.total_samples += 1

        if is_injection_gt and predicted_injection:
            result.true_positives += 1
        elif is_injection_gt and not predicted_injection:
            result.false_negatives += 1
        elif not is_injection_gt and predicted_injection:
            result.false_positives += 1
        else:
            result.true_negatives += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} — F1: {result.f1:.3f}, latency_mean: {result.latency_stats()['mean']:.1f}ms")

    summary = result.summary_dict()
    print(f"\nResults: F1={summary['f1']:.4f} Acc={summary['accuracy']:.4f} "
          f"P={summary['precision']:.4f} R={summary['recall']:.4f} "
          f"FPR={summary['fpr']:.4f} Lat_p50={summary['latency_p50_ms']:.1f}ms")

    # Free memory
    del pipe
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def write_results(all_results):
    """Write results as JSON and markdown summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    summaries = [r.summary_dict() for r in all_results]
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nJSON results → {json_path}")

    # Markdown summary
    df = pd.DataFrame(summaries)
    md_path = RESULTS_DIR / "SUMMARY.md"
    with open(md_path, "w") as f:
        f.write("# Prompt Injection Detection Benchmark Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Hardware:** CPU (Apple Silicon M3)\n\n")
        f.write("## Results Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")

        # Auto-generate findings
        best_f1 = df.loc[df["f1"].idxmax()]
        fastest = df.loc[df["latency_p50_ms"].idxmin()]
        f.write(f"- **Best F1:** {best_f1['model']} ({best_f1['f1']:.4f})\n")
        f.write(f"- **Fastest:** {fastest['model']} (p50: {fastest['latency_p50_ms']:.1f}ms)\n")
        f.write(f"- **Best accuracy:** {df.loc[df['accuracy'].idxmax()]['model']} ({df['accuracy'].max():.4f})\n")
        f.write(f"- **Lowest FPR:** {df.loc[df['fpr'].idxmin()]['model']} ({df['fpr'].min():.4f})\n")
    print(f"Markdown summary → {md_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to run (default: non-gated). Options: {list(MODELS.keys())}")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to run (default: all). Options: {list(DATASETS.keys())}")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset (for quick test)")
    args = parser.parse_args()

    # Default: skip gated models
    if args.models:
        models_to_run = args.models
    else:
        models_to_run = [k for k, v in MODELS.items() if not v.get("gated")]

    datasets_to_run = args.datasets or list(DATASETS.keys())

    all_results = []
    for ds_name in datasets_to_run:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}, skipping")
            continue
        ds_info = DATASETS[ds_name]
        print(f"\n{'#'*60}")
        print(f"Loading dataset: {ds_info['desc']}")
        print(f"{'#'*60}")
        samples = ds_info["loader"]()
        if args.max_samples:
            samples = samples[:args.max_samples]
        n_inj = sum(1 for s in samples if s["is_injection"])
        print(f"Loaded {len(samples)} samples ({n_inj} injection, {len(samples)-n_inj} safe)")

        for model_name in models_to_run:
            if model_name not in MODELS:
                print(f"Unknown model: {model_name}, skipping")
                continue
            result = run_model_on_dataset(model_name, MODELS[model_name], samples, ds_name)
            all_results.append(result)

    write_results(all_results)


if __name__ == "__main__":
    main()
