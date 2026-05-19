"""Adapter: Intel Extension for PyTorch (IPEX) fp32 + oneDNN fusion. Linux x86 only."""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Adapter:
    name = "ipex_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            return "Linux x86 only"
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            return "intel-extension-for-pytorch not installed"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        import intel_extension_for_pytorch as ipex
        torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact))
        base = AutoModelForSequenceClassification.from_pretrained(str(artifact))
        base.eval()
        self.model = ipex.optimize(base, dtype=torch.float32)

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            )
        return out.logits.cpu().numpy()

    def cleanup(self) -> None:
        del self.model
