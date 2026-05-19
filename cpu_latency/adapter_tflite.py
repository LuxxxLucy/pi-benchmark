"""Adapter: TensorFlow Lite + XNNPACK fp32."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


class Adapter:
    name = "tflite_fp32"

    @classmethod
    def is_supported(cls) -> str | None:
        try:
            import tflite_runtime  # noqa: F401
        except ImportError:
            try:
                import tensorflow  # noqa: F401
            except ImportError:
                return "tflite_runtime/tensorflow not installed (uv sync --extra tflite)"
        return None

    def __init__(self, artifact: Path, model_short: str, num_threads: int):
        tflite_path = artifact / "model.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"tflite model missing: {tflite_path}")
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter

        from conversions import hf_local_dir
        hf_dir = hf_local_dir(model_short)
        self.tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        self.interpreter = Interpreter(model_path=str(tflite_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self._inputs = self.interpreter.get_input_details()
        self._outputs = self.interpreter.get_output_details()

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        for info in self._inputs:
            if info["name"].endswith("input_ids"):
                self.interpreter.set_tensor(info["index"], input_ids)
            elif info["name"].endswith("attention_mask"):
                self.interpreter.set_tensor(info["index"], attention_mask)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._outputs[0]["index"])

    def cleanup(self) -> None:
        del self.interpreter
