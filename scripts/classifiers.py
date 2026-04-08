"""Classifier loading: HuggingFace, custom, pattern-detector, and pipelines."""
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

ClassifierFn = Callable[[str], tuple[bool, float]]

INJECTION_LABEL_NAMES = {"injection", "prompt_injection", "jailbreak", "unsafe", "malicious", "harmful"}
SAFE_LABEL_NAMES = {"safe", "benign", "normal", "legitimate"}


# ---------------------------------------------------------------------------
# HuggingFace models
# ---------------------------------------------------------------------------

def _load_hf_model(hf_id: str):
    """Load HF model, handling _orig_mod. prefix from torch.compile()."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_id)
    try:
        wf = hf_hub_download(hf_id, "model.safetensors")
        raw_sd = load_file(wf)
    except Exception:
        wf = hf_hub_download(hf_id, "pytorch_model.bin")
        raw_sd = torch.load(wf, map_location="cpu", weights_only=True)

    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        print(f"  (stripped _orig_mod. prefix)")

    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(raw_sd, strict=False)
    model.eval()
    return model


def make_hf_classifier(hf_id: str, injection_labels: list[int]) -> ClassifierFn:
    """Create a classifier function from a HuggingFace model."""
    model = _load_hf_model(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu", dtype=torch.float32)

    def classify(text: str) -> tuple[bool, float]:
        t0 = time.perf_counter()
        try:
            result = pipe(text[:2000], truncation=True, max_length=512)
            elapsed = (time.perf_counter() - t0) * 1000
            label = result[0]["label"]
            if isinstance(label, str):
                lower = label.lower()
                if lower in INJECTION_LABEL_NAMES:
                    return True, elapsed
                if lower in SAFE_LABEL_NAMES:
                    return False, elapsed
                if label.startswith("LABEL_"):
                    return int(label.split("_")[1]) in injection_labels, elapsed
                return False, elapsed
            return int(label) in injection_labels, elapsed
        except Exception:
            return False, (time.perf_counter() - t0) * 1000

    return classify


# ---------------------------------------------------------------------------
# Custom classifiers
# ---------------------------------------------------------------------------

def _load_custom_classifier(name: str) -> ClassifierFn:
    """Load a custom classifier by name."""
    research_dir = Path(__file__).parent.parent.parent / "direct"
    indirect_dir = Path(__file__).parent.parent.parent / "indirect"

    if name == "rule_based":
        sys.path.insert(0, str(research_dir))
        from rule_based_detector import detect
        def classify(text):
            t0 = time.perf_counter()
            r = detect(text)
            return r.is_injection, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "hybrid":
        sys.path.insert(0, str(research_dir))
        from hybrid_classifier import classify as hybrid_classify
        def classify(text):
            t0 = time.perf_counter()
            r = hybrid_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "embedding_knn":
        sys.path.insert(0, str(research_dir))
        from embedding_classifier import classify as emb_classify
        def classify(text):
            t0 = time.perf_counter()
            r = emb_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "stackone_repro":
        sys.path.insert(0, str(indirect_dir))
        from stackone_python import classify as so_classify
        def classify(text):
            t0 = time.perf_counter()
            r = so_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "minilm_indirect_v1":
        sys.path.insert(0, str(indirect_dir))
        from train_classifier import classify as mlm_classify
        def classify(text):
            t0 = time.perf_counter()
            r = mlm_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "minilm_indirect_v2":
        sys.path.insert(0, str(indirect_dir))
        from train_v2 import classify as v2_classify
        def classify(text):
            t0 = time.perf_counter()
            r = v2_classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "pattern_detector":
        from simple_defender.pattern_detector import PatternDetector
        detector = PatternDetector()
        def classify(text):
            t0 = time.perf_counter()
            result = detector.analyze(text)
            is_inj = result.suggested_risk in ("high", "critical")
            return is_inj, (time.perf_counter() - t0) * 1000
        return classify

    elif name.startswith("tfidf_"):
        from .traditional_ml import load_tfidf_classifier
        ml_method = name.replace("tfidf_", "")
        model_dir = Path(__file__).parent.parent / "models" / f"tfidf-{ml_method}"
        return load_tfidf_classifier(model_dir)

    elif name.startswith("emb_"):
        from .traditional_ml import make_embedding_classifier
        import pickle
        from sentence_transformers import SentenceTransformer
        ml_method = name.replace("emb_", "")
        model_dir = Path(__file__).parent.parent / "models" / f"emb-{ml_method}"
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        return make_embedding_classifier(embedder, model)

    raise ValueError(f"Unknown custom classifier: {name}")


# ---------------------------------------------------------------------------
# Composable pipelines
# ---------------------------------------------------------------------------

def make_composed_classifier(stage1: ClassifierFn, stage2: ClassifierFn) -> ClassifierFn:
    """Stage1 first; if flagged, short-circuit. Otherwise run Stage2."""
    def classify(text: str) -> tuple[bool, float]:
        is_inj, lat1 = stage1(text)
        if is_inj:
            return True, lat1
        is_inj2, lat2 = stage2(text)
        return is_inj2, lat1 + lat2
    return classify


# ---------------------------------------------------------------------------
# Build classifier from config entry
# ---------------------------------------------------------------------------

def build_classifier(name: str, cfg: dict, all_models: dict) -> ClassifierFn:
    """Build a ClassifierFn from a model config entry."""
    model_type = cfg.get("type", "custom" if "custom" in cfg else "hf")

    if model_type == "hf":
        return make_hf_classifier(cfg["hf_id"], cfg["injection_labels"])
    elif model_type == "custom":
        return _load_custom_classifier(cfg["custom"])
    elif model_type == "pipeline":
        s1_cfg = all_models[cfg["stage1"]]
        s2_cfg = all_models[cfg["stage2"]]
        stage1 = build_classifier(cfg["stage1"], s1_cfg, all_models)
        stage2 = build_classifier(cfg["stage2"], s2_cfg, all_models)
        return make_composed_classifier(stage1, stage2)
    else:
        raise ValueError(f"Unknown model type '{model_type}' for {name}")
