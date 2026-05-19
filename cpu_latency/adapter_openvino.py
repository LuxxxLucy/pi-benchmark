"""Adapter: OpenVINO fp32 (CPU plugin) with latency/throughput performance hint."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


_MODE = "latency"


def set_mode(mode: str) -> None:
    """Called from bench.py before adapter instantiation."""
    global _MODE
    assert mode in {"latency", "throughput"}, mode
    _MODE = mode


class Adapter:
    name = "openvino_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        try:
            import openvino  # noqa: F401
        except ImportError:
            return "openvino not installed (uv sync --extra openvino)"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        import openvino as ov

        xml = artifact / "openvino_model.xml"
        if not xml.exists():
            raise FileNotFoundError(f"openvino model missing: {xml}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact))
        core = ov.Core()
        config = {
            "INFERENCE_NUM_THREADS": num_threads,
            "PERFORMANCE_HINT": "LATENCY" if _MODE == "latency" else "THROUGHPUT",
        }
        compiled = core.compile_model(str(xml), "CPU", config)
        self.infer_req = compiled.create_infer_request()
        self._wanted = [i.any_name for i in compiled.inputs]
        self._tti_cache: np.ndarray | None = None

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feeds = {}
        for name in self._wanted:
            if name == "input_ids":
                feeds[name] = input_ids
            elif name == "attention_mask":
                feeds[name] = attention_mask
            elif name == "token_type_ids":
                if self._tti_cache is None or self._tti_cache.shape != input_ids.shape:
                    self._tti_cache = np.zeros_like(input_ids)
                feeds[name] = self._tti_cache
        result = self.infer_req.infer(feeds)
        return list(result.values())[0]

    def cleanup(self) -> None:
        del self.infer_req
