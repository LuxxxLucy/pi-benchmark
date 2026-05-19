"""PI defender accuracy under three precisions.

Loads datasets via ../scripts/datasets.py and emits per (model, precision):
  - per-category F1 (direct / indirect / jailbreak)
  - composite F1 = mean of the three category F1s
  - max FPR across the three benign-prompt datasets
  - delta_* vs fp32 (added once fp32 row exists)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from scripts.datasets import load_dataset_samples  # noqa: E402

from _adapters import (  # noqa: E402
    make_adapter, model_dir_for, model_short_names, precision_names,
)


DIRECT = ["deepset-all", "protectai-validation", "spml-chatbot",
          "lakera-mosscap", "lakera-gandalf", "neuralchemy",
          "necent-multilingual"]
INDIRECT = ["bipia", "wildguardtest"]
JAILBREAK = ["in-the-wild-jailbreak", "jailbreakbench", "jailbreakhub",
             "semantic-router-jailbreak"]
FPR_BENIGN = ["notinject", "false-reject", "or-bench"]

CATEGORY_DATASETS = {
    "direct": DIRECT,
    "indirect": INDIRECT,
    "jailbreak": JAILBREAK,
}


def _load_config() -> dict:
    with open(ROOT.parent / "config.yaml") as f:
        return yaml.safe_load(f)


def _model_max_length(model_dir: Path) -> int:
    cfg = json.loads((model_dir / "config.json").read_text())
    return int(cfg.get("max_position_embeddings", 512))


def _confusion(preds: np.ndarray, labels: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _fpr(fp: int, tn: int) -> float:
    return fp / (fp + tn) if (fp + tn) else 0.0


def _predict(adapter, texts: list[str], max_length: int,
             batch_size: int = 32) -> np.ndarray:
    """Batched tokenize + forward. Pads to the longest in each batch (or max_length)."""
    preds: list[int] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = adapter.tokenizer(
            chunk, return_tensors="np", truncation=True,
            padding=True, max_length=max_length,
        )
        logits = adapter.forward(enc["input_ids"], enc["attention_mask"])
        preds.extend(np.argmax(logits, axis=-1).astype(np.int64).tolist())
    return np.asarray(preds, dtype=np.int64)


def eval_one(model: str, precision: str, datasets_cfg: dict,
             num_threads: int, max_samples: int | None,
             rebuild_cache: bool) -> dict:
    print(f"\n=== {model} / {precision} ===")
    model_dir = model_dir_for(model)
    try:
        adapter = make_adapter(precision, model_dir, num_threads)
    except Exception as e:
        print(f"  load failed: {e}")
        return {"model": model, "precision": precision, "error": repr(e)}

    max_length = _model_max_length(model_dir)
    per_dataset: dict[str, dict] = {}
    cat_f1s: dict[str, list[float]] = {k: [] for k in CATEGORY_DATASETS}
    fprs_benign: list[float] = []

    every_dataset = sum(CATEGORY_DATASETS.values(), []) + FPR_BENIGN
    for ds_name in every_dataset:
        if ds_name not in datasets_cfg:
            print(f"  skip {ds_name}: not in config.yaml")
            continue
        ds_cfg = datasets_cfg[ds_name]
        try:
            samples = load_dataset_samples(ds_name, ds_cfg, rebuild_cache=rebuild_cache)
        except Exception as e:
            print(f"  {ds_name}: load failed: {e}")
            continue
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
        texts = [s["text"] for s in samples]
        labels = np.asarray([int(s["is_injection"]) for s in samples], dtype=np.int64)
        try:
            preds = _predict(adapter, texts, max_length)
        except Exception as e:
            print(f"  {ds_name}: predict failed: {e}")
            continue
        tp, fp, tn, fn = _confusion(preds, labels)
        f1 = _f1(tp, fp, fn)
        fpr = _fpr(fp, tn)
        per_dataset[ds_name] = {
            "samples": int(labels.shape[0]), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "f1": f1, "fpr": fpr,
        }
        print(f"  {ds_name:>28s}: n={len(samples):>5d}  F1={f1:.3f}  FPR={fpr:.3f}")
        for cat, names in CATEGORY_DATASETS.items():
            if ds_name in names:
                cat_f1s[cat].append(f1)
        if ds_name in FPR_BENIGN:
            fprs_benign.append(fpr)

    direct_f1 = float(np.mean(cat_f1s["direct"])) if cat_f1s["direct"] else 0.0
    indirect_f1 = float(np.mean(cat_f1s["indirect"])) if cat_f1s["indirect"] else 0.0
    jailbreak_f1 = float(np.mean(cat_f1s["jailbreak"])) if cat_f1s["jailbreak"] else 0.0
    composite_f1 = float(np.mean([direct_f1, indirect_f1, jailbreak_f1]))
    max_fpr_3 = float(max(fprs_benign)) if fprs_benign else 0.0

    return {
        "model": model,
        "precision": precision,
        "composite_f1": composite_f1,
        "direct_f1": direct_f1,
        "indirect_f1": indirect_f1,
        "jailbreak_f1": jailbreak_f1,
        "max_fpr_3": max_fpr_3,
        "per_dataset": per_dataset,
    }


def attach_deltas(rows: list[dict]) -> None:
    fp32_by_model = {r["model"]: r for r in rows
                     if r["precision"] == "fp32" and "composite_f1" in r}
    keys = ["composite_f1", "direct_f1", "indirect_f1", "jailbreak_f1"]
    for r in rows:
        if r["precision"] == "fp32" or "composite_f1" not in r:
            continue
        base = fp32_by_model.get(r["model"])
        if not base:
            continue
        for k in keys:
            r[f"delta_{k}_vs_fp32"] = r[k] - base[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(model_short_names()))
    ap.add_argument("--precisions", default=",".join(precision_names()))
    ap.add_argument("--threads", type=int,
                    default=max(1, multiprocessing.cpu_count() // 2))
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Dry-run only. Per [No sample capping] the real run uses the full set.")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--out", default="results/accuracy.json")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]

    cfg = _load_config()
    datasets_cfg = cfg["datasets"]

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "num_threads": args.threads,
        "max_samples_dryrun": args.max_samples,
    }
    print(f"env: {json.dumps(env, indent=2)}")

    rows: list[dict] = []
    for model in models:
        for precision in precisions:
            rows.append(eval_one(
                model, precision, datasets_cfg,
                args.threads, args.max_samples, args.rebuild_cache,
            ))
            out_path.write_text(json.dumps({"env": env, "results": rows}, indent=2))

    attach_deltas(rows)
    out_path.write_text(json.dumps({"env": env, "results": rows}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
