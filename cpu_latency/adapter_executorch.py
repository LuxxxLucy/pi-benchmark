"""Adapter: ExecuTorch + XNNPACK fp32 (PyTorch edge runtime)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


class Adapter:
    name = "executorch_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        try:
            from executorch.runtime import Runtime  # noqa: F401
        except ImportError:
            return "executorch not installed (uv sync --extra executorch)"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        from executorch.runtime import Runtime

        pte_path = artifact / "model.pte"
        if not pte_path.exists():
            raise FileNotFoundError(f"pte missing: {pte_path}")

        from conversions import hf_local_dir
        hf_dir = hf_local_dir(model_short)
        self.tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        torch.set_num_threads(num_threads)
        self.runtime = Runtime.get()
        self.program = self.runtime.load_program(str(pte_path))
        self.method = self.program.load_method("forward")

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        inputs = (torch.from_numpy(input_ids), torch.from_numpy(attention_mask))
        out = self.method.execute(inputs)
        return out[0].numpy() if isinstance(out, (list, tuple)) else out.numpy()

    def cleanup(self) -> None:
        del self.method
        del self.program
