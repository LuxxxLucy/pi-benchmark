"""Runtime registry: short name -> (Adapter class, conversion target format)."""
from __future__ import annotations

import adapter_compile
import adapter_ctranslate2
import adapter_executorch
import adapter_ipex
import adapter_llamacpp
import adapter_onnx
import adapter_openvino
import adapter_pytorch
import adapter_tflite


RUNTIMES = {
    adapter_pytorch.Adapter.name:     (adapter_pytorch.Adapter,     "hf"),
    adapter_compile.Adapter.name:     (adapter_compile.Adapter,     "hf"),
    adapter_onnx.Adapter.name:        (adapter_onnx.Adapter,        "onnx_opt"),
    adapter_ctranslate2.Adapter.name: (adapter_ctranslate2.Adapter, "ct2"),
    adapter_llamacpp.Adapter.name:    (adapter_llamacpp.Adapter,    "gguf_f32"),
    adapter_tflite.Adapter.name:      (adapter_tflite.Adapter,      "tflite"),
    adapter_executorch.Adapter.name:  (adapter_executorch.Adapter,  "pte"),
    adapter_ipex.Adapter.name:        (adapter_ipex.Adapter,        "hf"),
    adapter_openvino.Adapter.name:    (adapter_openvino.Adapter,    "ov"),
}


def runtime_names() -> list[str]:
    return list(RUNTIMES.keys())
