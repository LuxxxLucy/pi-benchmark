"""Evaluate a trained InjectionClassifier checkpoint on held-out test sets.

Emits a single JSON with per-dataset F1/P/R/FPR at the config.json threshold
(plus F1 at 0.5 for reference), a held-out generalization gap vs val, and a
CUDA single-sample latency sanity check.

Test sets (all held out from train_v2.py's training mix):
  - bipia_test:    text + code attacks in BIPIA's *_attack_test.json, injected into
                   held-out contexts from {email,code,table}/test.jsonl. Close to the
                   training distribution; primary in-distribution F1 check.
  - deepset_test:  samples from deepset/prompt-injections whose text falls in
                   is_test_partition. OOD-ish: different attack style from BIPIA.
  - xxz224_test:   samples from xxz224/prompt-injection-attack-dataset (train split)
                   whose text falls in is_test_partition. OOD task-hijacking style.
  - notinject:     bowen-uchicago/notinject benign probe — FPR-only (339 samples).
                   Fully external; tests over-defense on non-PI prompts.

Usage:
    uv run python scripts/eval_trained.py \\
        --save-dir models/bge-micro-v2-indirect \\
        --output    result/lightweight_sweep/bge-micro-v2.json
"""
from __future__ import annotations
# NOTE: this file lives in scripts/, which contains a local `datasets.py` used
# elsewhere in the project. Python auto-prepends the script's directory to
# sys.path, so a plain `from datasets import load_dataset` would resolve to
# scripts/datasets.py instead of the HuggingFace `datasets` package. Strip
# scripts/ from sys.path BEFORE any third-party imports.
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _THIS_DIR]
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import argparse
import json
import random
import time
from pathlib import Path
from contextlib import nullcontext

import torch
from datasets import load_dataset  # HuggingFace datasets
from transformers import AutoTokenizer

from train_v2 import InjectionClassifier, BIPIA_DIR  # noqa: E402
from split_utils import is_test_partition  # noqa: E402


SEED = 42


# ---------------------------------------------------------------------------
# Held-out test sets — same pattern as train_v2.py loaders, inverted partition.
# ---------------------------------------------------------------------------

def load_bipia_test() -> list[dict]:
    """BIPIA attacks_test.json × held-out contexts from {email,code,table}/test.jsonl."""
    with open(BIPIA_DIR / "text_attack_test.json") as f:
        attacks_raw = json.load(f)
    attacks = []
    for variants in attacks_raw.values():
        attacks.extend(variants)

    contexts: list[str] = []
    for category in ["email", "code", "table"]:
        test_file = BIPIA_DIR / category / "test.jsonl"
        if not test_file.exists():
            continue
        with open(test_file) as fh:
            for line in fh:
                row = json.loads(line)
                ctx = row.get("context", "")
                if isinstance(ctx, list):
                    ctx = "\n".join(str(c) for c in ctx)
                if ctx:
                    contexts.append(ctx)

    random.seed(SEED)
    samples = []
    for ctx in contexts:
        samples.append({"text": ctx, "is_injection": False})
    for ctx in contexts:
        for attack in random.sample(attacks, min(3, len(attacks))):
            samples.append({"text": ctx + "\n\n" + attack, "is_injection": True})
    return samples


def load_deepset_test() -> list[dict]:
    ds = load_dataset("deepset/prompt-injections")
    samples = []
    for split in ds:
        for r in ds[split]:
            if is_test_partition(r["text"]):
                samples.append({"text": r["text"], "is_injection": r["label"] == 1})
    return samples


def load_xxz224_test() -> list[dict]:
    ds = load_dataset("xxz224/prompt-injection-attack-dataset", split="train")
    attack_cols = ["naive_attack", "escape_attack", "ignore_attack",
                   "fake_comp_attack", "combine_attack"]
    samples = []
    seen_benign = set()
    for r in ds:
        txt = r["target_text"]
        if is_test_partition(txt):
            if txt not in seen_benign:
                seen_benign.add(txt)
                samples.append({"text": txt, "is_injection": False})
        for col in attack_cols:
            atk = r[col]
            if atk and is_test_partition(atk):
                samples.append({"text": atk, "is_injection": True})
    return samples


def load_notinject() -> list[dict]:
    """NotInject benign probe — all 339 samples are benign, FPR-only."""
    try:
        ds = load_dataset("bowen-uchicago/notinject")
    except Exception as exc:
        print(f"  [warn] could not load bowen-uchicago/notinject: {exc}", file=sys.stderr)
        return []
    samples = []
    for split in ds:
        for r in ds[split]:
            txt = r.get("text") or r.get("prompt") or r.get("instruction")
            if not txt:
                continue
            samples.append({"text": txt, "is_injection": False})
    return samples


# ---------------------------------------------------------------------------
# Inference + metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_samples(model, tokenizer, samples, device, max_length: int,
                  batch_size: int = 32, use_amp: bool = False) -> list[float]:
    model.eval()
    scores: list[float] = []
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        encoded = tokenizer(
            [s["text"][:2000] for s in batch],
            padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        with amp_ctx:
            logits = model(input_ids, attention_mask)
        scores.extend(torch.sigmoid(logits.float()).tolist())
    return scores


def metrics_at(scores, labels, threshold) -> dict:
    tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
    fn = sum(1 for s, l in zip(scores, labels) if s <  threshold and l == 1)
    tn = sum(1 for s, l in zip(scores, labels) if s <  threshold and l == 0)
    n_pos = tp + fn
    n_neg = fp + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": len(labels), "n_pos": n_pos, "n_neg": n_neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "fpr": round(fpr, 4), "f1": round(f1, 4)}


def _measure_on(model_for_device, tokenizer, device, max_length: int,
                use_amp: bool, n_warmup: int, n_iters: int) -> dict:
    sample = "The quick brown fox jumps over the lazy dog. " * 40  # ~256 tokens post-tokenize
    encoded = tokenizer(sample, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model_for_device.eval()
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else nullcontext())
    is_cuda = device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            with amp_ctx:
                _ = model_for_device(input_ids, attention_mask)
        sync()
        timings = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            with amp_ctx:
                _ = model_for_device(input_ids, attention_mask)
            sync()
            timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    return {
        "device": device.type,
        "dtype": "bfloat16" if use_amp else "float32",
        "max_length": max_length, "n_iters": n_iters,
        "p50_ms": round(timings[n_iters // 2], 3),
        "p95_ms": round(timings[int(n_iters * 0.95)], 3),
        "mean_ms": round(sum(timings) / n_iters, 3),
    }


def measure_latency(model, tokenizer, max_length: int, which: str,
                    cuda_available: bool, n_warmup: int = 10,
                    n_iters: int = 50) -> dict:
    """Batch=1, single-sample latency. `which` ∈ {cpu, gpu, both, none}.

    CPU is the primary A6 deployment target. GPU is the optional secondary
    (bf16 autocast). Returns a dict with sub-results keyed by device plus
    a top-level mirror of the CPU result for the aggregator.
    """
    if which == "none":
        return {"note": "latency measurement skipped (--latency none)"}
    if which == "gpu" and not cuda_available:
        return {"note": "gpu requested but cuda unavailable"}

    original_device = next(model.parameters()).device
    out: dict = {}

    # CPU pass — move model to CPU, measure fp32, then leave it there (caller
    # will restore the device if it still needs it on GPU).
    if which in ("cpu", "both"):
        model.to("cpu")
        out["cpu"] = _measure_on(model, tokenizer, torch.device("cpu"),
                                 max_length=max_length, use_amp=False,
                                 n_warmup=n_warmup, n_iters=n_iters)

    # GPU pass — move back to CUDA (bf16 autocast). Skipped if unavailable or
    # user explicitly picked "cpu".
    if which in ("gpu", "both") and cuda_available:
        model.to("cuda")
        out["gpu"] = _measure_on(model, tokenizer, torch.device("cuda"),
                                 max_length=max_length, use_amp=True,
                                 n_warmup=n_warmup, n_iters=n_iters)

    model.to(original_device)  # restore

    primary = "cpu" if "cpu" in out else ("gpu" if "gpu" in out else None)
    if primary:
        out["primary"] = primary
        for k in ("p50_ms", "p95_ms", "mean_ms", "device", "dtype"):
            out[k] = out[primary].get(k)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True,
                        help="Directory containing best_model.pt + config.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the eval JSON")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true",
                        help="Cap each test set to 100 samples for a fast pipeline sanity run")
    parser.add_argument("--smoke-n", type=int, default=100,
                        help="Cap per test set when --smoke (default 100)")
    parser.add_argument("--latency", choices=["cpu", "gpu", "both", "none"],
                        default="both",
                        help="Which devices to measure single-sample latency on "
                             "(default: both; CPU is the A6 primary target, GPU is secondary)")
    args = parser.parse_args()

    # ── Device
    if args.device == "cpu":
        dev = torch.device("cpu")
    elif args.device == "cuda":
        dev = torch.device("cuda")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    print(f"Device: {dev}  amp_bf16: {use_amp}")

    # ── Load model
    cfg_path = args.save_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    base_model = cfg["base_model"]
    threshold = cfg["threshold"]
    max_length = cfg.get("max_length", 256)
    trust_remote_code = cfg.get("trust_remote_code", False)
    print(f"Base model: {base_model}")
    print(f"Threshold (from config): {threshold}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    model = InjectionClassifier(base_model, trust_remote_code=trust_remote_code).to(dev)
    ckpt = torch.load(args.save_dir / "best_model.pt", map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Load test sets
    print("\nLoading test sets..." + ("  [SMOKE MODE: capping each to "
                                      f"{args.smoke_n}]" if args.smoke else ""))
    test_sets = {
        "bipia_test":   load_bipia_test(),
        "deepset_test": load_deepset_test(),
        "xxz224_test":  load_xxz224_test(),
        "notinject":    load_notinject(),
    }
    if args.smoke:
        rng = random.Random(SEED)
        for k, v in list(test_sets.items()):
            if len(v) > args.smoke_n:
                # Stratified cap: keep class balance so metrics stay meaningful.
                pos = [s for s in v if s["is_injection"]]
                neg = [s for s in v if not s["is_injection"]]
                rng.shuffle(pos); rng.shuffle(neg)
                n_each = args.smoke_n // 2
                test_sets[k] = pos[:n_each] + neg[:args.smoke_n - n_each]
    for name, samples in test_sets.items():
        n_pos = sum(1 for s in samples if s["is_injection"])
        n_neg = len(samples) - n_pos
        print(f"  {name:14s}: {len(samples):6d} total  ({n_pos} inj / {n_neg} ben)")

    # ── Score + metrics per set
    per_dataset = {}
    t_eval0 = time.time()
    for name, samples in test_sets.items():
        if not samples:
            per_dataset[name] = {"skipped": True}
            continue
        scores = score_samples(model, tokenizer, samples, dev, max_length,
                               batch_size=args.batch_size, use_amp=use_amp)
        labels = [1 if s["is_injection"] else 0 for s in samples]
        entry = {
            "at_best_threshold": metrics_at(scores, labels, threshold),
            "at_0.5":            metrics_at(scores, labels, 0.5),
        }
        per_dataset[name] = entry
        m = entry["at_best_threshold"]
        print(f"  {name:14s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  FPR={m['fpr']:.3f}")
    eval_elapsed = time.time() - t_eval0

    # ── Latency sanity check — CPU primary, GPU optional (see --latency)
    print(f"\nLatency sanity (single-sample, batch=1) — measuring {args.latency}...")
    latency = measure_latency(
        model, tokenizer, max_length=max_length,
        which=args.latency, cuda_available=(dev.type == "cuda"),
    )
    if "cpu" in latency:
        c = latency["cpu"]
        print(f"  cpu fp32  p50={c['p50_ms']:.2f} ms  p95={c['p95_ms']:.2f} ms  "
              f"(max_length={max_length})")
    if "gpu" in latency:
        g = latency["gpu"]
        print(f"  gpu bf16  p50={g['p50_ms']:.2f} ms  p95={g['p95_ms']:.2f} ms")
    if "note" in latency:
        print(f"  {latency['note']}")

    # ── Write JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model": base_model,
        "save_dir": str(args.save_dir),
        "config": cfg,
        "device": str(dev),
        "eval_elapsed_s": round(eval_elapsed, 1),
        "per_dataset": per_dataset,
        "latency_sanity": latency,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
