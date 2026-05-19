"""Adapter: ONNX Runtime fp32 + transformer optimizer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


class Adapter:
    name = "onnx_fp32_opt"

    @classmethod
    def is_supported(cls) -> str | None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "onnxruntime not installed"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        import onnxruntime as ort
        opt_path = artifact / "model_opt.onnx"
        if not opt_path.exists():
            opt_path = artifact / "model.onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(opt_path), sess_options=opts, providers=["CPUExecutionProvider"]
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

    def cleanup(self) -> None:
        del self.session
