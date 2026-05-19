"""Idempotent model fetch + format conversion + cache.

Each (model, target_format) pair maps to a stable directory under
models_converted/. Re-running ensure_converted with the same arguments is a
no-op if the artifact already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HF_IDS = {
    "testsavantai-small": "testsavantai/prompt-injection-defender-small-v0",
    "fmops-distilbert": "Fmops/distilbert-prompt-injection",
    "deepset-deberta-injection": "deepset/deberta-v3-base-injection",
}

FORMATS = ["hf", "onnx_opt", "ct2", "gguf_f32", "tflite", "pte", "ov"]

CPU_LATENCY_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = CPU_LATENCY_ROOT.parent
HF_ROOT = BENCHMARK_ROOT / "models"
CACHE_ROOT = CPU_LATENCY_ROOT / "models_converted"
LOG_PATH = CACHE_ROOT / "_conversion.log.json"


def model_short_names() -> list[str]:
    return list(HF_IDS.keys())


def model_hf_id(short_name: str) -> str:
    if short_name not in HF_IDS:
        raise KeyError(f"unknown model: {short_name}")
    return HF_IDS[short_name]


def hf_local_dir(short_name: str) -> Path:
    return HF_ROOT / HF_IDS[short_name].replace("/", "--")


def converted_dir(short_name: str, fmt: str) -> Path:
    return CACHE_ROOT / short_name / fmt


def _config_sha(short_name: str) -> str:
    """Hash of config.json content; cheap proxy for "source weights changed"."""
    cfg = hf_local_dir(short_name) / "config.json"
    if not cfg.exists():
        return "missing"
    return hashlib.sha256(cfg.read_bytes()).hexdigest()[:16]


def _log(short_name: str, fmt: str, status: str, detail: str, wall_s: float) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    if LOG_PATH.exists():
        entries = json.loads(LOG_PATH.read_text())
    entries.append({
        "model": short_name, "format": fmt, "status": status,
        "source_sha": _config_sha(short_name),
        "detail": detail, "wall_s": round(wall_s, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    LOG_PATH.write_text(json.dumps(entries, indent=2))


_HF_ALLOW_BASE = [
    "config.json",
    "configuration_*.py",
    "modeling_*.py",
    "tokenizer*",
    "tokenization_*.py",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
]


def _hf_weight_pattern(repo_id: str) -> str:
    """Pick one weight format. Prefer safetensors; otherwise bin (sharded or single)."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id)
    if any(f.endswith(".safetensors") for f in files):
        return "*.safetensors"
    return "pytorch_model*.bin"


def ensure_hf_local(short_name: str) -> Path:
    """Download HF weights to ../models/<local_name>/ if missing. Returns the directory.

    Uses an allow_patterns filter so we only pull the tokenizer + one weight
    format — bare snapshot_download would double-pull bin+safetensors and grab
    the README / .h5 / .msgpack / flax weights that nothing here uses.
    """
    target = hf_local_dir(short_name)
    if (target / "config.json").exists():
        return target
    from huggingface_hub import snapshot_download
    t0 = time.perf_counter()
    repo = HF_IDS[short_name]
    allow = _HF_ALLOW_BASE + [_hf_weight_pattern(repo)]
    cache = snapshot_download(repo, allow_patterns=allow)
    target.mkdir(parents=True, exist_ok=True)
    for f in Path(cache).iterdir():
        if f.is_file():
            shutil.copy2(f, target / f.name)
    _log(short_name, "hf", "ok", f"downloaded from {repo}",
         time.perf_counter() - t0)
    return target


def _convert_onnx_opt(short_name: str, out_dir: Path) -> None:
    from optimum.exporters.onnx import main_export
    from onnxruntime.transformers.optimizer import optimize_model
    from transformers import AutoConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    src = ensure_hf_local(short_name)
    main_export(model_name_or_path=str(src), output=out_dir, task="text-classification")

    raw = out_dir / "model.onnx"
    if not raw.exists():
        raise RuntimeError(
            f"optimum-cli export did not produce {raw}; "
            f"contents: {sorted(p.name for p in out_dir.iterdir())}"
        )
    cfg = AutoConfig.from_pretrained(src)
    opt_path = out_dir / "model_opt.onnx"
    opt = optimize_model(
        str(raw),
        model_type="bert",
        num_heads=cfg.num_attention_heads,
        hidden_size=cfg.hidden_size,
    )
    opt.save_model_to_file(str(opt_path))


def _convert_ct2(short_name: str, out_dir: Path) -> None:
    src = ensure_hf_local(short_name)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [sys.executable, "-m", "ctranslate2.converters.transformers",
           "--model", str(src), "--output_dir", str(out_dir),
           "--copy_files", "tokenizer.json"]
    subprocess.run(cmd, check=True)


def _find_gguf_converter() -> Path | None:
    """Locate llama.cpp's convert_hf_to_gguf.py.

    Looks at $LLAMA_CPP_CONVERT, then PATH, then $LLAMA_CPP_ROOT/convert_hf_to_gguf.py.
    """
    env = os.environ.get("LLAMA_CPP_CONVERT")
    if env and Path(env).exists():
        return Path(env)
    on_path = shutil.which("convert_hf_to_gguf.py")
    if on_path:
        return Path(on_path)
    root = os.environ.get("LLAMA_CPP_ROOT")
    if root:
        candidate = Path(root) / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
    return None


def _convert_gguf_f32(short_name: str, out_dir: Path) -> None:
    converter = _find_gguf_converter()
    if converter is None:
        raise RuntimeError(
            "gguf conversion needs llama.cpp's convert_hf_to_gguf.py. "
            "Set $LLAMA_CPP_CONVERT to its path, or $LLAMA_CPP_ROOT to the "
            "llama.cpp checkout."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ensure_hf_local(short_name)
    cmd = [sys.executable, str(converter), str(src),
           "--outfile", str(out_dir / "model.gguf"), "--outtype", "f32"]
    subprocess.run(cmd, check=True)


def _convert_tflite(short_name: str, out_dir: Path) -> None:
    try:
        import ai_edge_torch
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(f"tflite extras not installed: {e}")

    src = ensure_hf_local(short_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(src)).eval()
    tok = AutoTokenizer.from_pretrained(str(src))
    sample = tok("hello world", return_tensors="pt", padding="max_length",
                 truncation=True, max_length=256)
    args = (sample["input_ids"], sample["attention_mask"])
    edge_model = ai_edge_torch.convert(model, args)
    edge_model.export(str(out_dir / "model.tflite"))


def _convert_pte(short_name: str, out_dir: Path) -> None:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from executorch.exir import to_edge
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
    except ImportError as e:
        raise RuntimeError(f"executorch extras not installed: {e}")

    src = ensure_hf_local(short_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(src)).eval()
    tok = AutoTokenizer.from_pretrained(str(src))
    sample = tok("hello world", return_tensors="pt", padding="max_length",
                 truncation=True, max_length=256)
    args = (sample["input_ids"], sample["attention_mask"])
    exported = torch.export.export(model, args)
    edge = to_edge(exported)
    edge = edge.to_backend(XnnpackPartitioner())
    program = edge.to_executorch()
    (out_dir / "model.pte").write_bytes(program.buffer)


def _convert_ov(short_name: str, out_dir: Path) -> None:
    try:
        from optimum.intel import OVModelForSequenceClassification
    except ImportError as e:
        raise RuntimeError(f"openvino extras not installed: {e}")

    src = ensure_hf_local(short_name)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = OVModelForSequenceClassification.from_pretrained(str(src), export=True)
    model.save_pretrained(str(out_dir))


_CONVERTERS = {
    "onnx_opt": _convert_onnx_opt,
    "ct2": _convert_ct2,
    "gguf_f32": _convert_gguf_f32,
    "tflite": _convert_tflite,
    "pte": _convert_pte,
    "ov": _convert_ov,
}


def ensure_converted(short_name: str, fmt: str) -> Path:
    if fmt == "hf":
        return ensure_hf_local(short_name)
    if fmt not in _CONVERTERS:
        raise ValueError(f"unknown format: {fmt}")
    out_dir = converted_dir(short_name, fmt)
    sentinel = {
        "onnx_opt": out_dir / "model_opt.onnx",
        "ct2": out_dir / "model.bin",
        "gguf_f32": out_dir / "model.gguf",
        "tflite": out_dir / "model.tflite",
        "pte": out_dir / "model.pte",
        "ov": out_dir / "openvino_model.xml",
    }[fmt]
    if sentinel.exists():
        return out_dir
    t0 = time.perf_counter()
    try:
        _CONVERTERS[fmt](short_name, out_dir)
        _log(short_name, fmt, "ok", "", time.perf_counter() - t0)
    except Exception as e:
        _log(short_name, fmt, "fail", repr(e), time.perf_counter() - t0)
        raise
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--formats", default="all")
    args = ap.parse_args()

    models = list(HF_IDS) if args.models == "all" else args.models.split(",")
    formats = FORMATS if args.formats == "all" else args.formats.split(",")

    for m in models:
        for f in formats:
            try:
                p = ensure_converted(m, f)
                print(f"  {m:30s} {f:10s} ok  -> {p}")
            except Exception as e:
                print(f"  {m:30s} {f:10s} FAIL  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
