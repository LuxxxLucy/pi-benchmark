"""Traditional ML baselines for prompt injection detection.

These methods are designed to be fast alternatives to transformer models.
All implement the ClassifierFn interface: (text) -> (is_injection, latency_ms).
"""
import time
import pickle
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# TF-IDF + LightGBM / LogReg
# ---------------------------------------------------------------------------

def train_tfidf_classifier(train_samples: list[dict], method: str = "lgbm",
                           save_path: Path | None = None):
    """Train TF-IDF + classifier. Returns (vectorizer, model)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [s["text"][:2000] for s in train_samples]
    labels = [1 if s["is_injection"] else 0 for s in train_samples]

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        analyzer="char_wb",  # Character n-grams — robust to obfuscation
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    if method == "lgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=200, max_depth=6, num_leaves=31,
                               learning_rate=0.1, n_jobs=1, verbose=-1)
    elif method == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0)
    elif method == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              n_jobs=-1, verbosity=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X, labels)

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open(save_path / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {method} model to {save_path}")

    return vectorizer, model


def make_tfidf_classifier(vectorizer, model):
    """Create ClassifierFn from a trained TF-IDF + classifier."""
    def classify(text: str) -> tuple[bool, float]:
        t0 = time.perf_counter()
        X = vectorizer.transform([text[:2000]])
        pred = model.predict(X)[0]
        return bool(pred), (time.perf_counter() - t0) * 1000
    return classify


def load_tfidf_classifier(model_dir: Path):
    """Load a saved TF-IDF classifier."""
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return make_tfidf_classifier(vectorizer, model)


# ---------------------------------------------------------------------------
# Embedding + classifier (pre-computed MiniLM embeddings)
# ---------------------------------------------------------------------------

def train_embedding_classifier(train_samples: list[dict], method: str = "logreg",
                               save_path: Path | None = None):
    """Train classifier on MiniLM embeddings. Returns (embedder, model)."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [s["text"][:2000] for s in train_samples]
    labels = np.array([1 if s["is_injection"] else 0 for s in train_samples])

    print(f"  Computing embeddings for {len(texts)} samples...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    if method == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0)
    elif method == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              n_jobs=-1, verbosity=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(embeddings, labels)

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Saved embedding+{method} model to {save_path}")

    return embedder, model


def make_embedding_classifier(embedder, model):
    """Create ClassifierFn from embedding + classifier."""
    def classify(text: str) -> tuple[bool, float]:
        t0 = time.perf_counter()
        emb = embedder.encode([text[:2000]])
        pred = model.predict(emb)[0]
        return bool(pred), (time.perf_counter() - t0) * 1000
    return classify


# ---------------------------------------------------------------------------
# CLI for training traditional ML models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "indirect"))
    from train_v2 import prepare_dataset

    parser = argparse.ArgumentParser(description="Train traditional ML baselines")
    parser.add_argument("--methods", nargs="+",
                        default=["tfidf-lgbm", "tfidf-logreg", "emb-logreg", "emb-xgboost"],
                        help="Methods to train")
    parser.add_argument("--max-samples", type=int, help="Limit training samples")
    args = parser.parse_args()

    print("Loading training data...")
    train_data, val_data = prepare_dataset()
    if args.max_samples:
        train_data = train_data[:args.max_samples]

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Training: {method}")
        print(f"{'='*60}")

        save_dir = MODELS_DIR / method

        if method.startswith("tfidf-"):
            ml_method = method.split("-", 1)[1]
            vec, model = train_tfidf_classifier(train_data, ml_method, save_dir)
            clf = make_tfidf_classifier(vec, model)
        elif method.startswith("emb-"):
            ml_method = method.split("-", 1)[1]
            emb, model = train_embedding_classifier(train_data, ml_method, save_dir)
            clf = make_embedding_classifier(emb, model)
        else:
            print(f"Unknown method: {method}")
            continue

        # Quick validation
        correct = 0
        for s in val_data[:200]:
            pred, _ = clf(s["text"])
            if pred == s["is_injection"]:
                correct += 1
        print(f"  Val accuracy ({min(200, len(val_data))} samples): {correct/min(200, len(val_data)):.3f}")
