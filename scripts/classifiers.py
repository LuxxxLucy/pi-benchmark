"""Classifier loading: HuggingFace, custom, pattern-detector, and pipelines."""
import importlib.util
import time
import json
import shutil
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

ClassifierFn = Callable[[str], tuple[bool, float]]

INJECTION_LABEL_NAMES = {"injection", "prompt_injection", "jailbreak", "unsafe", "malicious", "harmful"}
SAFE_LABEL_NAMES = {"safe", "benign", "normal", "legitimate"}

MODELS_DIR = Path(__file__).parent.parent / "models"

# Base tokenizer for ModernBERT models that don't ship their own
MODERNBERT_TOKENIZER = "answerdotai/ModernBERT-base"


# ---------------------------------------------------------------------------
# Module import helper (replaces sys.path mutation)
# ---------------------------------------------------------------------------

def _import_module_from_file(module_name: str, file_path: Path):
    """Import a Python module from an absolute file path without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# HuggingFace models — download to local models/ dir
# ---------------------------------------------------------------------------

def _ensure_local_model(hf_id: str) -> Path:
    """Download HF model to local models/ directory if not already present.

    Also fixes known issues (malformed id2label, missing tokenizer files).
    Returns the local directory path.
    """
    from huggingface_hub import snapshot_download

    local_name = hf_id.replace("/", "--")
    local_dir = MODELS_DIR / local_name

    if local_dir.exists() and (local_dir / "config.json").exists():
        return local_dir

    print(f"  Downloading {hf_id} to {local_dir}...")
    cache_dir = snapshot_download(hf_id)

    local_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(cache_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, local_dir / f.name)

    # Fix malformed id2label (e.g., vijil-dome has int values)
    cfg_path = local_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        needs_fix = False
        if "id2label" in cfg:
            for v in cfg["id2label"].values():
                if not isinstance(v, str):
                    needs_fix = True
                    break
        if needs_fix:
            label_names = {0: "benign", 1: "injection"}
            cfg["id2label"] = {k: label_names.get(int(k), str(v)) for k, v in cfg["id2label"].items()}
            cfg["label2id"] = {v: int(k) for k, v in cfg["id2label"].items()}
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"  (fixed malformed id2label)")

    # Check if tokenizer files exist; if not, copy from base model
    tok_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    has_tokenizer = any((local_dir / tf).exists() for tf in tok_files)
    if not has_tokenizer:
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "")
            if model_type == "modernbert":
                base_tok = MODERNBERT_TOKENIZER
            else:
                base_tok = cfg.get("_name_or_path", "")

            if base_tok:
                print(f"  (copying tokenizer from base: {base_tok})")
                base_cache = snapshot_download(base_tok)
                for tf in tok_files + ["vocab.txt", "merges.txt", "sentencepiece.bpe.model"]:
                    src = Path(base_cache) / tf
                    if src.exists():
                        shutil.copy2(src, local_dir / tf)

    # Write README with provenance
    readme = local_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {hf_id}\n\n"
            f"Downloaded from: https://huggingface.co/{hf_id}\n\n"
            f"Command: `snapshot_download(\"{hf_id}\")`\n"
        )

    return local_dir


def _label_is_injection(label, injection_labels: list[int]) -> bool:
    """Check if a pipeline output label means injection."""
    if isinstance(label, str):
        lower = label.lower()
        if lower in INJECTION_LABEL_NAMES:
            return True
        if lower in SAFE_LABEL_NAMES:
            return False
        if label.startswith("LABEL_"):
            return int(label.split("_")[1]) in injection_labels
        return False
    return int(label) in injection_labels


def make_hf_classifier(hf_id: str, injection_labels: list[int],
                       device: str = "cpu", batch_size: int = 1) -> ClassifierFn:
    """Create a classifier function from a HuggingFace model.

    Text truncation is handled by the tokenizer (max_length=512).
    No character-level truncation — let the model see as much as the
    tokenizer allows.
    """
    local_dir = _ensure_local_model(hf_id)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer,
                    device=device, dtype=dtype, batch_size=batch_size)

    def classify(text: str) -> tuple[bool, float]:
        t0 = time.perf_counter()
        try:
            result = pipe(text, truncation=True, max_length=512)
            elapsed = (time.perf_counter() - t0) * 1000
            return _label_is_injection(result[0]["label"], injection_labels), elapsed
        except Exception:
            return False, (time.perf_counter() - t0) * 1000

    def classify_batch(texts: list[str]) -> list[bool]:
        results = pipe(texts, truncation=True, max_length=512)
        return [_label_is_injection(r["label"], injection_labels) for r in results]

    classify.batch = classify_batch
    return classify


# ---------------------------------------------------------------------------
# Custom classifiers
# ---------------------------------------------------------------------------

def _load_custom_classifier(name: str) -> ClassifierFn:
    """Load a custom classifier by name."""
    research_dir = Path(__file__).parent.parent.parent / "direct"
    indirect_dir = Path(__file__).parent.parent.parent / "indirect"

    if name == "rule_based":
        mod = _import_module_from_file("rule_based_detector",
                                       research_dir / "rule_based_detector.py")
        def classify(text):
            t0 = time.perf_counter()
            r = mod.detect(text)
            return r.is_injection, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "hybrid":
        mod = _import_module_from_file("hybrid_classifier",
                                       research_dir / "hybrid_classifier.py")
        def classify(text):
            t0 = time.perf_counter()
            r = mod.classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "embedding_knn":
        mod = _import_module_from_file("embedding_classifier",
                                       research_dir / "embedding_classifier.py")
        def classify(text):
            t0 = time.perf_counter()
            r = mod.classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "stackone_repro":
        mod = _import_module_from_file("stackone_python",
                                       indirect_dir / "stackone_python.py")
        def classify(text):
            t0 = time.perf_counter()
            r = mod.classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "minilm_indirect_v1":
        mod = _import_module_from_file("train_classifier",
                                       indirect_dir / "train_classifier.py")
        def classify(text):
            t0 = time.perf_counter()
            r = mod.classify(text)
            return r, (time.perf_counter() - t0) * 1000
        return classify

    elif name == "minilm_indirect_v2":
        mod = _import_module_from_file("train_v2",
                                       indirect_dir / "train_v2.py")
        def classify(text):
            t0 = time.perf_counter()
            r = mod.classify(text)
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
        model_dir = MODELS_DIR / f"tfidf-{ml_method}"
        return load_tfidf_classifier(model_dir)

    elif name.startswith("emb_"):
        from .traditional_ml import make_embedding_classifier
        import pickle
        from sentence_transformers import SentenceTransformer
        ml_method = name.replace("emb_", "")
        model_dir = MODELS_DIR / f"emb-{ml_method}"
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

def build_classifier(name: str, cfg: dict, all_models: dict,
                     device: str = "cpu", batch_size: int = 1) -> ClassifierFn:
    """Build a ClassifierFn from a model config entry."""
    model_type = cfg.get("type", "custom" if "custom" in cfg else "hf")

    if model_type == "hf":
        return make_hf_classifier(cfg["hf_id"], cfg["injection_labels"],
                                  device=device, batch_size=batch_size)
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
