"""
Prompt injection detection benchmark runner.
Supports HuggingFace models and custom classifiers.
Merges new results with existing ones (no re-running completed combos).
"""
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"

# ---------------------------------------------------------------------------
# Classifier interface: (text: str) -> (is_injection: bool, latency_ms: float)
# ---------------------------------------------------------------------------

ClassifierFn = Callable[[str], tuple[bool, float]]

INJECTION_LABEL_NAMES = {"injection", "jailbreak", "unsafe", "malicious", "harmful"}
SAFE_LABEL_NAMES = {"safe", "benign", "normal", "legitimate"}


def _make_hf_classifier(hf_id, injection_labels) -> ClassifierFn:
    """Create a classifier function from a HuggingFace model."""
    model = _load_hf_model(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu", dtype=torch.float32)

    def classify(text: str) -> tuple[bool, float]:
        t0 = time.perf_counter()
        try:
            result = pipe(text[:2000], truncation=True, max_length=512)
            elapsed = (time.perf_counter() - t0) * 1000
            label = result[0]["label"]
            if isinstance(label, str):
                lower = label.lower()
                if lower in INJECTION_LABEL_NAMES:
                    return True, elapsed
                if lower in SAFE_LABEL_NAMES:
                    return False, elapsed
                if label.startswith("LABEL_"):
                    return int(label.split("_")[1]) in injection_labels, elapsed
                return False, elapsed
            return int(label) in injection_labels, elapsed
        except Exception:
            return False, (time.perf_counter() - t0) * 1000

    return classify


def _load_hf_model(hf_id):
    """Load HF model, handling _orig_mod. prefix from torch.compile()."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_id)
    try:
        wf = hf_hub_download(hf_id, "model.safetensors")
        raw_sd = load_file(wf)
    except Exception:
        wf = hf_hub_download(hf_id, "pytorch_model.bin")
        raw_sd = torch.load(wf, map_location="cpu", weights_only=True)

    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        print(f"  (stripped _orig_mod. prefix)")

    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(raw_sd, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    # HuggingFace models
    "fmops-distilbert":          {"hf_id": "Fmops/distilbert-prompt-injection", "params": "66M", "injection_labels": [1]},
    "protectai-base-v1":         {"hf_id": "protectai/deberta-v3-base-prompt-injection", "params": "86M", "injection_labels": [1]},
    "protectai-base-v2":         {"hf_id": "protectai/deberta-v3-base-prompt-injection-v2", "params": "86M", "injection_labels": [1]},
    "vsr-jailbreak-modernbert":  {"hf_id": "LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model", "params": "149M", "injection_labels": [1]},
    "vsr-jailbreak-mmbert32k-merged": {"hf_id": "llm-semantic-router/mmbert32k-jailbreak-detector-merged", "params": "307M", "injection_labels": [1]},
    # Gated
    "prompt-guard-22m":  {"hf_id": "meta-llama/Llama-Prompt-Guard-2-22M", "params": "22M", "injection_labels": [1, 2], "gated": True},
    "prompt-guard-86m":  {"hf_id": "meta-llama/Llama-Prompt-Guard-2-86M", "params": "86M", "injection_labels": [1, 2], "gated": True},
    "protectai-small-v2": {"hf_id": "protectai/deberta-v3-small-prompt-injection-v2", "params": "44M", "injection_labels": [1], "gated": True},
    # Custom classifiers (loaded via _load_custom_classifier)
    "rule-based":    {"custom": "rule_based", "params": "0"},
    "hybrid":        {"custom": "hybrid", "params": "~86M"},
    "embedding-knn": {"custom": "embedding_knn", "params": "~23M"},
    # Indirect injection classifiers
    "stackone-repro": {"custom": "stackone_repro", "params": "~23M"},
}


def _load_custom_classifier(name: str) -> ClassifierFn:
    """Load a custom classifier by name."""
    research_dir = Path(__file__).parent.parent.parent / "direct"
    sys.path.insert(0, str(research_dir))

    if name == "rule_based":
        from rule_based_detector import detect
        def classify(text):
            t0 = time.perf_counter()
            r = detect(text)
            return r.is_injection, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "hybrid":
        from hybrid_classifier import classify as hybrid_classify
        def classify(text):
            t0 = time.perf_counter()
            r = hybrid_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "embedding_knn":
        from embedding_classifier import classify as emb_classify
        def classify(text):
            t0 = time.perf_counter()
            r = emb_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "stackone_repro":
        indirect_dir = Path(__file__).parent.parent.parent / "indirect"
        sys.path.insert(0, str(indirect_dir))
        from stackone_python import classify as so_classify
        def classify(text):
            t0 = time.perf_counter()
            r = so_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    raise ValueError(f"Unknown custom classifier: {name}")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _load_binary(hf_id, text_col="text", label_col="label", injection_label=1, split=None):
    if split:
        ds = load_dataset(hf_id, split=split)
        rows = list(ds)
    else:
        ds = load_dataset(hf_id)
        rows = [r for s in ds for r in ds[s]]
    return [{"text": r[text_col], "is_injection": r[label_col] == injection_label} for r in rows]


def _load_jailbreakbench():
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    return ([{"text": r["Goal"], "is_injection": True} for r in ds["harmful"]] +
            [{"text": r["Goal"], "is_injection": False} for r in ds["benign"]])


def _load_bipia():
    """Load BIPIA indirect injection dataset from local repo.

    Generates poisoned samples by inserting attack strings into benign contexts
    (email, code, table). Each poisoned context is labeled as injection; each
    clean context is included once as benign.
    """
    import random
    bipia_dir = Path(__file__).parent.parent.parent / "indirect" / "bipia_repo" / "benchmark"

    # Load attack strings
    with open(bipia_dir / "text_attack_test.json") as f:
        attacks_raw = json.load(f)
    attacks = []
    for variants in attacks_raw.values():
        attacks.extend(variants)

    # Load benign contexts from all available categories
    samples = []
    for category in ["email", "code", "table"]:
        cat_dir = bipia_dir / category
        test_file = cat_dir / "test.jsonl"
        if not test_file.exists():
            continue
        with open(test_file) as f:
            for line in f:
                row = json.loads(line)
                context = row.get("context", "")
                if isinstance(context, list):
                    context = "\n".join(str(c) for c in context)
                if not context:
                    continue
                # Add clean context as benign
                samples.append({"text": context, "is_injection": False})
                # Insert a random attack into the context (injection sample)
                attack = random.choice(attacks)
                poisoned = context + "\n\n" + attack
                samples.append({"text": poisoned, "is_injection": True})

    random.shuffle(samples)
    return samples


def _load_xxz224():
    """Load xxz224 prompt-injection-attack-dataset (~3747 rows).

    Each row has a benign target_text and 5 attack variants. We include the
    target_text as benign and each attack column as an injection sample.
    """
    ds = load_dataset("xxz224/prompt-injection-attack-dataset", split="train")
    attack_cols = ["naive_attack", "escape_attack", "ignore_attack",
                   "fake_comp_attack", "combine_attack"]
    samples = []
    seen_benign = set()
    for r in ds:
        txt = r["target_text"]
        if txt not in seen_benign:
            seen_benign.add(txt)
            samples.append({"text": txt, "is_injection": False})
        for col in attack_cols:
            samples.append({"text": r[col], "is_injection": True})
    return samples


def _load_notinject():
    """Load NotInject dataset — benign prompts with trigger words (FPR test).

    All 339 samples are benign (is_injection=False). Measures false positive rate.
    """
    ds = load_dataset("leolee99/NotInject")
    samples = []
    for split in ds:
        for r in ds[split]:
            samples.append({"text": r["prompt"], "is_injection": False})
    return samples


DATASETS = {
    # Direct injection datasets
    "deepset-all":    {"desc": "deepset/prompt-injections (662)", "loader": lambda: _load_binary("deepset/prompt-injections")},
    "xTRam1":         {"desc": "xTRam1/safe-guard (10K+)", "loader": lambda: _load_binary("xTRam1/safe-guard-prompt-injection")},
    "jailbreakbench": {"desc": "JailbreakBench (200)", "loader": _load_jailbreakbench},
    # Indirect injection datasets
    "bipia":          {"desc": "BIPIA indirect injection (400)", "loader": _load_bipia},
    "notinject":      {"desc": "NotInject FPR-only (339 benign)", "loader": _load_notinject},
    # Multi-attack / large-scale datasets
    "xxz224":         {"desc": "xxz224 task-hijacking attacks (22K)", "loader": _load_xxz224},
    "jayavibhav":     {"desc": "jayavibhav prompt-injection test (65K)", "loader": lambda: _load_binary("jayavibhav/prompt-injection", split="test")},
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class Result:
    model: str
    params: str
    dataset: str
    samples: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies_ms: list = field(default_factory=list)

    @property
    def precision(self):
        d = self.tp + self.fp; return self.tp / d if d else 0
    @property
    def recall(self):
        d = self.tp + self.fn; return self.tp / d if d else 0
    @property
    def f1(self):
        p, r = self.precision, self.recall; return 2*p*r/(p+r) if p+r else 0
    @property
    def fpr(self):
        d = self.fp + self.tn; return self.fp / d if d else 0
    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.samples if self.samples else 0

    def to_dict(self):
        s = sorted(self.latencies_ms) if self.latencies_ms else [0]
        n = len(s)
        return {
            "model": self.model, "params": self.params, "dataset": self.dataset,
            "samples": self.samples,
            "accuracy": round(self.accuracy, 4), "precision": round(self.precision, 4),
            "recall": round(self.recall, 4), "f1": round(self.f1, 4), "fpr": round(self.fpr, 4),
            "latency_p50_ms": round(s[n//2], 2), "latency_p95_ms": round(s[int(n*0.95)], 2),
            "latency_mean_ms": round(sum(s)/n, 2), "errors": 0,
        }


def run_classifier(name: str, classifier_fn: ClassifierFn, samples: list, dataset_name: str, params: str) -> Result:
    """Run a classifier on samples."""
    print(f"\n  {name} on {dataset_name} ({len(samples)} samples)...", end="", flush=True)
    r = Result(model=name, params=params, dataset=dataset_name)

    for sample in samples:
        is_inj, lat = classifier_fn(sample["text"])
        r.latencies_ms.append(lat)
        r.samples += 1
        if sample["is_injection"] and is_inj: r.tp += 1
        elif sample["is_injection"] and not is_inj: r.fn += 1
        elif not sample["is_injection"] and is_inj: r.fp += 1
        else: r.tn += 1

    print(f" F1={r.f1:.3f} P={r.precision:.3f} R={r.recall:.3f} lat={r.to_dict()['latency_p50_ms']:.1f}ms")
    return r


def load_existing_results() -> list[dict]:
    """Load previously saved results."""
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return []


def save_results(all_dicts: list[dict]):
    """Save results and generate markdown summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_dicts, f, indent=2)

    df = pd.DataFrame(all_dicts)
    with open(RESULTS_DIR / "SUMMARY.md", "w") as f:
        f.write("# Prompt Injection Detection Benchmark Results\n\n")
        f.write(f"**Updated:** {time.strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Hardware:** CPU (Apple Silicon M3)\n\n")

        for ds_name in df["dataset"].unique():
            sub = df[df["dataset"] == ds_name].sort_values("f1", ascending=False)
            f.write(f"### {ds_name} ({sub.iloc[0]['samples']} samples)\n\n")
            cols = ["model", "params", "f1", "precision", "recall", "fpr", "latency_p50_ms"]
            f.write(sub[cols].to_markdown(index=False))
            f.write("\n\n")

    print(f"\nResults → {RESULTS_JSON}")
    print(f"Summary → {RESULTS_DIR / 'SUMMARY.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="PI detection benchmark")
    p.add_argument("--models", nargs="+", help=f"Models: {list(MODELS.keys())}")
    p.add_argument("--datasets", nargs="+", help=f"Datasets: {list(DATASETS.keys())}")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--force", action="store_true", help="Re-run even if results exist")
    p.add_argument("--only-new", action="store_true", help="Only run model+dataset combos not in existing results")
    args = p.parse_args()

    models_to_run = args.models or [k for k, v in MODELS.items() if not v.get("gated")]
    datasets_to_run = args.datasets or list(DATASETS.keys())

    # Load existing results
    existing = load_existing_results()
    existing_keys = {(r["model"], r["dataset"]) for r in existing}

    new_results = []

    for ds_name in datasets_to_run:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}"); continue

        # Load dataset once per dataset
        ds_info = DATASETS[ds_name]
        samples = None  # lazy load

        for model_name in models_to_run:
            if model_name not in MODELS:
                print(f"Unknown model: {model_name}"); continue

            # Skip if already run (unless --force)
            if not args.force and (model_name, ds_name) in existing_keys:
                print(f"  [skip] {model_name} × {ds_name} (already in results)")
                continue

            # Lazy load dataset
            if samples is None:
                print(f"\nLoading {ds_info['desc']}...")
                samples = ds_info["loader"]()
                if args.max_samples:
                    samples = samples[:args.max_samples]
                n_inj = sum(1 for s in samples if s["is_injection"])
                print(f"  {len(samples)} samples ({n_inj} inj, {len(samples)-n_inj} safe)")

            # Build classifier
            cfg = MODELS[model_name]
            try:
                if "custom" in cfg:
                    clf = _load_custom_classifier(cfg["custom"])
                else:
                    clf = _make_hf_classifier(cfg["hf_id"], cfg["injection_labels"])
            except Exception as e:
                print(f"  [fail] {model_name}: {e}")
                continue

            result = run_classifier(model_name, clf, samples, ds_name, cfg["params"])
            new_results.append(result.to_dict())

            # Free HF model memory
            if "custom" not in cfg:
                del clf
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    # Merge with existing (replace if same model+dataset)
    if new_results:
        new_keys = {(r["model"], r["dataset"]) for r in new_results}
        merged = [r for r in existing if (r["model"], r["dataset"]) not in new_keys]
        merged.extend(new_results)
        save_results(merged)
    else:
        print("\nNo new results to save.")


if __name__ == "__main__":
    main()
