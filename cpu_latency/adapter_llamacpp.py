"""Adapter: llama.cpp BERT (fp32 gguf) + PyTorch classifier head.

Uses llama-cpp-python's embedding API to get pooled CLS state.
Applies the PyTorch model's classification head on top.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import heads
from conversions import hf_local_dir


class Adapter:
    name = "llamacpp_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return "llama-cpp-python not installed (uv sync --extra llamacpp)"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        import llama_cpp
        gguf_path = artifact / "model.gguf"
        if not gguf_path.exists():
            raise FileNotFoundError(f"gguf missing: {gguf_path}")
        hf_dir = hf_local_dir(model_short)
        self.tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        self.llm = llama_cpp.Llama(
            model_path=str(gguf_path), embedding=True, n_ctx=512,
            n_threads=num_threads, verbose=False,
        )
        self._head_model = AutoModelForSequenceClassification.from_pretrained(str(hf_dir))
        self._head_model.eval()

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        emb = np.asarray(self.llm.embed(text), dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None, :]
        with torch.no_grad():
            logits = heads.from_pooled_cls(self._head_model, torch.from_numpy(emb))
        return logits.numpy()

    def cleanup(self) -> None:
        del self.llm
        del self._head_model
