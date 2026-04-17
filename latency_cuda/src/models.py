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
    trust_remote_code: bool = False   # needed for custom model_type (nomic_bert, NewModel, …)


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

    # ── Classic lightweight variants (random head — arch-baseline) ─────────
    ModelSpec("bert-L4-H512", "google/bert_uncased_L-4_H-512_A-8",
              "bert-tiny", "arch-baseline",
              "Google tiny BERT family, 29M (4L-512H); ships tokenizer.json "
              "so no sentencepiece/tiktoken convert required"),
    ModelSpec("albert-base-v2", "albert/albert-base-v2",
              "albert", "arch-baseline",
              "ALBERT-base, 12M raw / 12-layer shared weights — param count "
              "≠ compute; isolates parameter-sharing latency"),
    ModelSpec("electra-small", "google/electra-small-discriminator",
              "electra", "arch-baseline",
              "ELECTRA-small discriminator, 14M; factorized embedding"),
    ModelSpec("mobilebert", "google/mobilebert-uncased",
              "mobilebert", "arch-baseline",
              "MobileBERT, 25M; 24L deep-narrow inverted-bottleneck — tests "
              "depth-vs-width at batch=1"),

    # ── Embedding-backbone fillers (random seq_cls head — arch-baseline) ───
    ModelSpec("bge-micro-v2", "TaylorAI/bge-micro-v2",
              "bert-tiny", "arch-baseline",
              "BGE micro, 17M; BERT 3L-384H — fills 3-layer cell "
              "(published by TaylorAI, not BAAI)"),
    ModelSpec("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5",
              "minilm", "arch-baseline",
              "BGE small EN, 33M; BERT 12L-384H"),

    # ── Modern / long-context encoders (2024-25 SOTA) ──────────────────────
    ModelSpec("modernbert-base", "answerdotai/ModernBERT-base",
              "modernbert", "arch-baseline",
              "ModernBERT-base, 149M, max_pos=8192; RoPE + GeGLU + "
              "local/global attention; flagship 2024 modern encoder"),
    ModelSpec("ettin-encoder-150m", "jhu-clsp/ettin-encoder-150m",
              "modernbert", "arch-baseline",
              "Ettin encoder-150m, 150M, max_pos=8192; ModernBERT-compatible, "
              "fully open-recipe (2025)"),
    ModelSpec("gte-base-en-v1.5", "Alibaba-NLP/gte-base-en-v1.5",
              "new", "arch-baseline",
              "GTE-base EN v1.5, 137M, max_pos=8192; custom NewModel type "
              "(RoPE θ=500k); requires trust_remote_code AND the explicit "
              "position_ids arange that bench_common.make_input now always "
              "passes (its modeling.py indexes an uninitialized buffer "
              "otherwise)",
              trust_remote_code=True),
    # nomic-ai/nomic-embed-text-v1.5 deferred: its custom modeling.py assumes
    # config.classifier_dropout is not None AND loads state_dict with
    # strict=True, so a random-head arch-baseline load fails on missing
    # classifier + pooler weights. Revisit if we want 137M long-context
    # nomic_bert numbers; would need a per-model loader branch.
]


def by_name(name: str) -> ModelSpec:
    for m in CANDIDATES:
        if m.name == name:
            return m
    raise KeyError(name)
