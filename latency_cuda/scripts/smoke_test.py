"""Smoke-test every ModelSpec in src/models.py.

Per model: load tokenizer + model on CPU (fp32), run one forward on a
short input, verify the output has a logits tensor. Emits a pass/fail
table so a new candidate can be added to models.py and confirmed to
load before the full CUDA+CPU bench runs on a GPU box.

Usage:
  uv run python scripts/smoke_test.py                 # all models
  uv run python scripts/smoke_test.py --models a,b    # subset
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# TRANSFORMERS_VERBOSITY must be set before import for logger suppression
# (same reason as bench.py).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT.parent))   # for bench_common.py

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from bench_common import make_input  # noqa: E402
from models import CANDIDATES, by_name  # noqa: E402


def smoke_one(spec) -> dict:
    """Load + one forward. Return a result dict for the summary table."""
    t0 = time.perf_counter()
    trc = spec.trust_remote_code
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.hf_id, use_fast=True, trust_remote_code=trc,
            )
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.hf_id, use_fast=False, trust_remote_code=trc,
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_id, dtype=torch.float32, trust_remote_code=trc,
        ).eval()
        load_s = time.perf_counter() - t0

        n_params = sum(p.numel() for p in model.parameters())
        enc = make_input(tokenizer, 64)
        with torch.no_grad():
            out = model(**enc)

        has_logits = hasattr(out, "logits") and isinstance(out.logits, torch.Tensor)
        logits_shape = tuple(out.logits.shape) if has_logits else None

        del model, tokenizer
        return {
            "name": spec.name, "hf_id": spec.hf_id, "ok": has_logits,
            "params_M": round(n_params / 1e6, 1),
            "load_s": round(load_s, 1),
            "logits_shape": logits_shape,
            "trc": trc, "error": None,
        }
    except Exception as e:
        return {
            "name": spec.name, "hf_id": spec.hf_id, "ok": False,
            "params_M": None, "load_s": None, "logits_shape": None,
            "trc": trc, "error": f"{e.__class__.__name__}: {e}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="Comma-separated short names, or 'all'.")
    args = ap.parse_args()

    if args.models == "all":
        picks = list(CANDIDATES)
    else:
        picks = [by_name(n.strip()) for n in args.models.split(",") if n.strip()]

    results = []
    for spec in picks:
        trc_tag = " [TRC]" if spec.trust_remote_code else ""
        print(f"\n=== {spec.name}  ({spec.hf_id}){trc_tag} ===")
        r = smoke_one(spec)
        results.append(r)
        if r["ok"]:
            print(f"  OK   — {r['params_M']}M params, load {r['load_s']}s, "
                  f"logits {r['logits_shape']}")
        else:
            print(f"  FAIL — {r['error']}")
            traceback.print_exc()

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{'─' * 78}")
    print(f"SUMMARY — {n_ok}/{len(results)} passed\n")
    print(f"{'name':<22} {'params':>8}  {'load':>6}  {'logits':<14}  {'trc':<3}  status")
    print(f"{'-' * 78}")
    for r in results:
        status = "ok" if r["ok"] else "FAIL"
        params = f"{r['params_M']}M" if r["params_M"] is not None else "-"
        load = f"{r['load_s']}s" if r["load_s"] is not None else "-"
        shape = str(r["logits_shape"]) if r["logits_shape"] is not None else "-"
        trc = "Y" if r["trc"] else ""
        print(f"{r['name']:<22} {params:>8}  {load:>6}  {shape:<14}  {trc:<3}  {status}")
        if not r["ok"]:
            print(f"    ↳ {r['error']}")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
