"""Curated lightweight classifier candidates for latency benchmarking.

Two groups:

1. **PI-trained** — real prompt-injection classifiers with task-trained heads.
   Use these when you want end-to-end latency on the deployed model.

2. **Arch-baseline** — open backbones with randomly-initialized classifier
   heads. Use these to isolate *architectural* latency from task-specific
   weight choices. Accuracy on these is meaningless; latency is real.

Each entry: `(short_name, hf_id, family, note)`. `family` groups by backbone
architecture so the result table can be sorted/compared.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str          # short, human-readable
    hf_id: str         # HuggingFace repo id
    family: str        # "minilm" | "distilbert" | "deberta" | "bert-tiny" | …
    group: str         # "pi-trained" | "arch-baseline"
    note: str = ""


CANDIDATES: list[ModelSpec] = [
    # ── PI-trained (deployed models; latency matches production) ───────────
    ModelSpec("testsavantai-tiny", "testsavantai/prompt-injection-defender-tiny-v0",
              "minilm", "pi-trained",
              "MiniLM-L6 backbone, 22M, 512 max_pos"),
    ModelSpec("testsavantai-small", "testsavantai/prompt-injection-defender-small-v0",
              "minilm", "pi-trained",
              "MiniLM-L12, 33M"),
    ModelSpec("testsavantai-medium", "testsavantai/prompt-injection-defender-medium-v0",
              "deberta", "pi-trained",
              "DeBERTa-v3 small, 44M"),
    ModelSpec("testsavantai-base", "testsavantai/prompt-injection-defender-base-v0",
              "deberta", "pi-trained",
              "DeBERTa-v3 base, 86M"),
    ModelSpec("fmops-distilbert", "fmops/distilbert-prompt-injection",
              "distilbert", "pi-trained",
              "DistilBERT, 66M"),
    ModelSpec("protectai-deberta-v2", "protectai/deberta-v3-base-prompt-injection-v2",
              "deberta", "pi-trained",
              "DeBERTa-v3 base, 86M, stronger recall than v1"),
    ModelSpec("deepset-deberta-injection", "deepset/deberta-v3-base-injection",
              "deberta", "pi-trained",
              "DeBERTa-v3 base, 86M"),

    # ── Architecture baselines (random head — for latency-per-arch) ────────
    ModelSpec("bert-tiny", "prajjwal1/bert-tiny",
              "bert-tiny", "arch-baseline",
              "2-layer 128-hidden, 4.4M; extreme low end"),
    ModelSpec("bert-mini", "prajjwal1/bert-mini",
              "bert-tiny", "arch-baseline",
              "4-layer 256-hidden, 11M"),
    ModelSpec("bert-L4-H256", "google/bert_uncased_L-4_H-256_A-4",
              "bert-tiny", "arch-baseline",
              "Google tiny BERT family, 11M"),
    ModelSpec("minilm-L6-H384", "sentence-transformers/all-MiniLM-L6-v2",
              "minilm", "arch-baseline",
              "ST's popular MiniLM-L6, 22M"),
    ModelSpec("minilm-L12-H384", "microsoft/MiniLM-L12-H384-uncased",
              "minilm", "arch-baseline",
              "Microsoft MiniLM-L12, 33M"),
    ModelSpec("deberta-v3-xsmall", "microsoft/deberta-v3-xsmall",
              "deberta", "arch-baseline",
              "DeBERTa-v3 xsmall, 22M but strong-per-param"),
    ModelSpec("deberta-v3-small", "microsoft/deberta-v3-small",
              "deberta", "arch-baseline",
              "DeBERTa-v3 small, 44M"),
    ModelSpec("distilbert-base", "distilbert-base-uncased",
              "distilbert", "arch-baseline",
              "DistilBERT reference, 66M"),
]


def by_name(name: str) -> ModelSpec:
    for m in CANDIDATES:
        if m.name == name:
            return m
    raise KeyError(name)
