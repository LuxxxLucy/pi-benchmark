"""Adapter: CTranslate2 fp32 encoder + PyTorch classifier head."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import heads
from conversions import hf_local_dir


class Adapter:
    name = "ctranslate2_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        try:
            import ctranslate2  # noqa: F401
        except ImportError:
            return "ctranslate2 not installed"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        import ctranslate2
        hf_dir = hf_local_dir(model_short)
        self.tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        self.encoder = ctranslate2.Encoder(
            str(artifact), device="cpu", intra_threads=num_threads, inter_threads=1,
        )
        self._head_model = AutoModelForSequenceClassification.from_pretrained(str(hf_dir))
        self._head_model.eval()

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        result = self.encoder.forward_batch([tokens])
        last_hidden = np.asarray(result.last_hidden_state)
        with torch.no_grad():
            logits = heads.from_last_hidden(self._head_model, torch.from_numpy(last_hidden))
        return logits.numpy()

    def cleanup(self) -> None:
        del self.encoder
        del self._head_model
