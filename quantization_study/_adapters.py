"""Precision-specific forward-pass adapters used by run_latency and run_accuracy.

Four precisions: fp32 (PyTorch eager), bf16 (PyTorch autocast), pt_int8_dynamic
(quantize_dynamic on Linear), onnx_int8_dynamic (ORT transformer optimizer + dynamic int8 quant).
"""
from __future__ import annotations

import contextlib
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _set_threads(n: int) -> None:
    torch.set_num_threads(n)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


class _PyTorchBase:
    """Shared PyTorch-eager forward path. Subclasses override `_load_model` and `_autocast`."""

    def __init__(self, model_dir: Path, num_threads: int):
        _set_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = self._load_model(model_dir)
        self.model.eval()

    def _load_model(self, model_dir: Path) -> torch.nn.Module:
        return AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    def _autocast(self):
        return contextlib.nullcontext()

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad(), self._autocast():
            out = self.model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            )
        return out.logits.float().cpu().numpy()


class _PyTorchFp32(_PyTorchBase):
    pass


class _PyTorchInt8Dynamic(_PyTorchBase):
    def __init__(self, model_dir: Path, num_threads: int):
        if platform.machine() in ("arm64", "aarch64"):
            torch.backends.quantized.engine = "qnnpack"
        else:
            torch.backends.quantized.engine = "x86"
        super().__init__(model_dir, num_threads)

    def _load_model(self, model_dir: Path) -> torch.nn.Module:
        fp32 = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        fp32.eval()
        return torch.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)


class _PyTorchBf16(_PyTorchBase):
    def _autocast(self):
        return torch.cpu.amp.autocast(dtype=torch.bfloat16)


class _OnnxInt8Dynamic:
    def __init__(self, model_dir: Path, num_threads: int):
        import onnxruntime as ort

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        int8_path = _ensure_onnx_int8_artifact(model_dir)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(int8_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._wanted = {i.name for i in self.session.get_inputs()}
        self._tti_cache: np.ndarray | None = None

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feeds: dict = {}
        if "input_ids" in self._wanted:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._wanted:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self._wanted:
            if self._tti_cache is None or self._tti_cache.shape != input_ids.shape:
                self._tti_cache = np.zeros_like(input_ids)
            feeds["token_type_ids"] = self._tti_cache
        outs = self.session.run(None, feeds)
        return outs[0]


def _ensure_onnx_int8_artifact(model_dir: Path) -> Path:
    """Lazily produce <model>/onnx_int8_dynamic/model.int8.onnx, cached on disk.

    Cache path is shared with ../cpu_latency/ when available; otherwise
    falls back to a sibling 'onnx_cache/' inside this study's directory.
    """
    shared = Path(__file__).resolve().parent.parent / "cpu_latency" / "models_converted" / model_dir.name
    fallback = Path(__file__).resolve().parent / "onnx_cache" / model_dir.name
    cache_root = shared if shared.parent.exists() else fallback
    int8_dir = cache_root / "onnx_int8_dynamic"
    int8_path = int8_dir / "model.int8.onnx"
    if int8_path.exists():
        return int8_path

    int8_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = cache_root / "onnx_fp32"
    onnx_path = onnx_dir / "model.onnx"

    if not onnx_path.exists():
        onnx_dir.mkdir(parents=True, exist_ok=True)
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=str(model_dir),
            output=onnx_dir,
            task="text-classification",
        )

    opt_path = onnx_dir / "model_opt.onnx"
    if not opt_path.exists():
        from onnxruntime.transformers.optimizer import optimize_model
        cfg = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).config
        opt = optimize_model(
            str(onnx_path),
            model_type="bert",
            num_heads=cfg.num_attention_heads,
            hidden_size=cfg.hidden_size,
        )
        opt.save_model_to_file(str(opt_path))

    pre_path = onnx_dir / "model_opt_pre.onnx"
    if not pre_path.exists():
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(opt_path), str(pre_path))

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(pre_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return int8_path


def make_adapter(precision: str, model_dir: Path, num_threads: int):
    if precision == "fp32":
        return _PyTorchFp32(model_dir, num_threads)
    if precision == "pt_int8_dynamic":
        return _PyTorchInt8Dynamic(model_dir, num_threads)
    if precision == "onnx_int8_dynamic":
        return _OnnxInt8Dynamic(model_dir, num_threads)
    if precision == "bf16":
        return _PyTorchBf16(model_dir, num_threads)
    raise ValueError(f"unknown precision: {precision}")


HF_IDS = {
    "testsavantai-small": "testsavantai/prompt-injection-defender-small-v0",
    "fmops-distilbert": "Fmops/distilbert-prompt-injection",
    "deepset-deberta-injection": "deepset/deberta-v3-base-injection",
}


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
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id)
    if any(f.endswith(".safetensors") for f in files):
        return "*.safetensors"
    return "pytorch_model*.bin"


def model_dir_for(short_name: str) -> Path:
    if short_name not in HF_IDS:
        raise KeyError(f"unknown model short name: {short_name}")
    hf_id = HF_IDS[short_name]
    local_name = hf_id.replace("/", "--")
    root = Path(__file__).resolve().parent.parent / "models" / local_name
    if not (root / "config.json").exists():
        from huggingface_hub import snapshot_download
        import shutil
        allow = _HF_ALLOW_BASE + [_hf_weight_pattern(hf_id)]
        print(f"  downloading {hf_id} -> {root}")
        cache = snapshot_download(hf_id, allow_patterns=allow)
        root.mkdir(parents=True, exist_ok=True)
        for f in Path(cache).iterdir():
            if f.is_file():
                shutil.copy2(f, root / f.name)
    return root


def model_short_names() -> list[str]:
    return list(HF_IDS.keys())


def precision_names() -> list[str]:
    return ["fp32", "pt_int8_dynamic", "onnx_int8_dynamic", "bf16"]
