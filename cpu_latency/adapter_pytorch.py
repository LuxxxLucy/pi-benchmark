"""Adapter: PyTorch eager fp32 (baseline)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Adapter:
    name = "pytorch_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(artifact))
        self.model.eval()

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            )
        return out.logits.cpu().numpy()

    def cleanup(self) -> None:
        del self.model
