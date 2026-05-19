"""Architecture-aware classifier heads for runtimes that return raw encoder output.

Two variants:
- `from_last_hidden`: input is the full (B, T, H) sequence; the head selects/pools and projects.
- `from_pooled_cls`: input is the already-pooled (B, H) CLS vector; the head only projects.
"""
from __future__ import annotations

import torch


def _arch(model) -> str:
    name = type(model).__name__
    if "DistilBert" in name:
        return "distilbert"
    if "Deberta" in name:
        return "deberta"
    if "Bert" in name:
        return "bert"
    raise NotImplementedError(f"head not implemented for {name}")


def from_last_hidden(model, last_hidden: torch.Tensor) -> torch.Tensor:
    arch = _arch(model)
    if arch == "distilbert":
        pooled = model.pre_classifier(last_hidden[:, 0])
        pooled = torch.nn.functional.relu(pooled)
        if hasattr(model, "dropout"):
            pooled = model.dropout(pooled)
        return model.classifier(pooled)
    if arch == "deberta":
        pooled = model.pooler(last_hidden)
        if hasattr(model, "dropout"):
            pooled = model.dropout(pooled)
        return model.classifier(pooled)
    pooled = model.bert.pooler(last_hidden)
    if hasattr(model, "dropout"):
        pooled = model.dropout(pooled)
    return model.classifier(pooled)


def from_pooled_cls(model, cls_hidden: torch.Tensor) -> torch.Tensor:
    """For runtimes (e.g. llama.cpp embedding API) that emit a pre-pooled CLS vector.

    We replay the same dense+activation that the HF pooler would have applied,
    then project through the classifier.
    """
    arch = _arch(model)
    if arch == "distilbert":
        pooled = model.pre_classifier(cls_hidden)
        pooled = torch.nn.functional.relu(pooled)
        return model.classifier(pooled)
    if arch == "deberta":
        pooled = torch.tanh(model.pooler.dense(cls_hidden))
        return model.classifier(pooled)
    pooled = torch.tanh(model.bert.pooler.dense(cls_hidden))
    return model.classifier(pooled)
