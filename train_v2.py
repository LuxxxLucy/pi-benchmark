"""
Train v2 indirect prompt injection classifier.

Changes from v1:
- Mixed training data: BIPIA + xxz224 + jayavibhav (train partitions only)
- Increased BIPIA benign ratio to fix 51.5% FPR
- Post-training threshold sweep
- Configurable base model (for lightweight model search)
- ONNX export support
"""
import json
import random
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from split_utils import is_test_partition

BIPIA_DIR = Path(__file__).resolve().parent / "datasets" / "bipia_repo" / "benchmark"
SAVE_DIR_DEFAULT = Path(__file__).parent / "models" / "minilm-indirect-v2"
MAX_LENGTH_DEFAULT = 256
SEED_DEFAULT = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bipia_train():
    """Load BIPIA training data: contexts + attacks → poisoned samples."""
    with open(BIPIA_DIR / "text_attack_train.json") as f:
        attacks_raw = json.load(f)
    attacks = []
    for variants in attacks_raw.values():
        attacks.extend(variants)

    contexts = []
    for category in ["email", "code", "table"]:
        train_file = BIPIA_DIR / category / "train.jsonl"
        if not train_file.exists():
            continue
        with open(train_file) as f:
            for line in f:
                row = json.loads(line)
                ctx = row.get("context", "")
                if isinstance(ctx, list):
                    ctx = "\n".join(str(c) for c in ctx)
                if ctx:
                    contexts.append(ctx)

    samples = []
    random.seed(SEED_DEFAULT)

    # Benign: ALL clean contexts (more benign = lower FPR)
    for ctx in contexts:
        samples.append({"text": ctx, "is_injection": False})

    # Injection: insert attacks into contexts
    for ctx in contexts:
        for attack in random.sample(attacks, min(3, len(attacks))):
            samples.append({"text": ctx + "\n\n" + attack, "is_injection": True})

    print(f"  BIPIA: {sum(1 for s in samples if s['is_injection'])} inj, "
          f"{sum(1 for s in samples if not s['is_injection'])} ben")
    return samples


def load_xxz224_train():
    """Load xxz224 task-hijacking attacks (train partition only)."""
    ds = load_dataset("xxz224/prompt-injection-attack-dataset", split="train")
    attack_cols = ["naive_attack", "escape_attack", "ignore_attack",
                   "fake_comp_attack", "combine_attack"]
    samples = []
    seen_benign = set()
    for r in ds:
        txt = r["target_text"]
        # Skip test partition
        if is_test_partition(txt):
            continue
        if txt not in seen_benign:
            seen_benign.add(txt)
            samples.append({"text": txt, "is_injection": False})
        for col in attack_cols:
            atk = r[col]
            if not is_test_partition(atk):
                samples.append({"text": atk, "is_injection": True})

    n_inj = sum(1 for s in samples if s["is_injection"])
    n_ben = len(samples) - n_inj
    print(f"  xxz224: {n_inj} inj, {n_ben} ben (train partition)")
    return samples


def load_jayavibhav_train():
    """Load jayavibhav prompt-injection train split."""
    ds = load_dataset("jayavibhav/prompt-injection", split="train")
    samples = []
    for r in ds:
        samples.append({"text": r["text"], "is_injection": r["label"] == 1})

    n_inj = sum(1 for s in samples if s["is_injection"])
    n_ben = len(samples) - n_inj
    print(f"  jayavibhav: {n_inj} inj, {n_ben} ben (train split)")
    return samples


def load_external_benign():
    """Load benign samples from deepset + xTRam1 (TRAIN partition only)."""
    benign = []

    ds = load_dataset("deepset/prompt-injections")
    n_kept = 0
    for split in ds:
        for r in ds[split]:
            if r["label"] == 0 and not is_test_partition(r["text"]):
                benign.append({"text": r["text"], "is_injection": False})
                n_kept += 1
    print(f"  deepset benign: {n_kept} (train partition)")

    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    n_kept = 0
    for split in ds:
        for r in ds[split]:
            if r["label"] == 0 and not is_test_partition(r["text"]):
                benign.append({"text": r["text"], "is_injection": False})
                n_kept += 1
    print(f"  xTRam1 benign: {n_kept} (train partition)")

    return benign


def prepare_dataset(max_injection=15000, max_benign=12000):
    """Prepare balanced training dataset from mixed sources.

    v2 key change: higher benign ratio (was 10K:5K, now 15K:12K)
    to fix BIPIA FPR (51.5% → target <10%).
    """
    print("Loading training data...")
    bipia = load_bipia_train()
    xxz224 = load_xxz224_train()
    jayavibhav = load_jayavibhav_train()

    print("\nLoading external benign samples...")
    external_benign = load_external_benign()

    # Combine all injection samples
    all_injection = (
        [s for s in bipia if s["is_injection"]] +
        [s for s in xxz224 if s["is_injection"]] +
        [s for s in jayavibhav if s["is_injection"]]
    )

    # Combine all benign samples
    all_benign = (
        [s for s in bipia if not s["is_injection"]] +
        [s for s in xxz224 if not s["is_injection"]] +
        [s for s in jayavibhav if not s["is_injection"]] +
        external_benign
    )

    print(f"\nRaw totals: {len(all_injection)} inj, {len(all_benign)} ben")

    # Subsample for balance
    random.seed(SEED_DEFAULT)
    random.shuffle(all_injection)
    random.shuffle(all_benign)
    all_injection = all_injection[:max_injection]
    all_benign = all_benign[:max_benign]

    combined = all_injection + all_benign
    random.shuffle(combined)

    # Train/val split (90/10)
    split_idx = int(len(combined) * 0.9)
    train = combined[:split_idx]
    val = combined[split_idx:]

    print(f"\nDataset: {len(train)} train, {len(val)} val")
    print(f"  Train: {sum(1 for s in train if s['is_injection'])} inj, "
          f"{sum(1 for s in train if not s['is_injection'])} ben")
    return train, val


# ---------------------------------------------------------------------------
# PyTorch dataset & model (same architecture as v1, but configurable base)
# ---------------------------------------------------------------------------

class InjectionDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=MAX_LENGTH_DEFAULT):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoded = self.tokenizer(
            s["text"][:2000],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(1.0 if s["is_injection"] else 0.0),
        }


class InjectionClassifier(nn.Module):
    def __init__(self, base_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 trust_remote_code: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
        hidden_size = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(base_model="sentence-transformers/all-MiniLM-L6-v2",
                epochs=5, batch_size=32, lr_bert=2e-5, lr_head=1e-3,
                max_samples=None, save_dir: Path = SAVE_DIR_DEFAULT,
                device: str = "auto", max_length: int = MAX_LENGTH_DEFAULT,
                trust_remote_code: bool = False):
    dev = _pick_device(device)
    use_amp = dev.type == "cuda"
    print(f"\nDevice: {dev}  amp_bf16: {use_amp}")
    print(f"Base model: {base_model}")
    print(f"Save dir: {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    train_data, val_data = prepare_dataset()

    if max_samples:
        train_data = train_data[:max_samples]
        val_data = val_data[:max(50, max_samples // 10)]
        print(f"[test mode] Using {len(train_data)} train, {len(val_data)} val samples")

    train_ds = InjectionDataset(train_data, tokenizer, max_length=max_length)
    val_ds = InjectionDataset(val_data, tokenizer, max_length=max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = InjectionClassifier(base_model, trust_remote_code=trust_remote_code)
    model.to(dev)
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.bert.parameters())
    print(f"Model params: {total_params / 1e6:.1f}M (backbone {backbone_params / 1e6:.1f}M)")

    bert_params = list(model.bert.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr_bert},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict] = []
    best_val_scores: list[float] = []
    best_val_labels: list[float] = []

    # Rough estimate: GPU ~5× faster than CPU at bs=32 for these sizes.
    per_step = 0.04 if dev.type == "cuda" else 0.15
    est_time = len(train_dl) * epochs * per_step
    print(f"Estimated training time: ~{est_time / 60:.0f} min ({len(train_dl)} steps/epoch × {epochs} epochs)")

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullcontext()

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_scores, val_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                labels = batch["label"].to(dev, non_blocking=True)
                with amp_ctx:
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                scores = torch.sigmoid(logits.float())
                preds = (scores >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                val_scores.extend(scores.tolist())
                val_labels.extend(labels.tolist())

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "elapsed_s": elapsed,
        })
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_model": base_model,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, save_dir / "best_model.pt")
            best_val_scores = list(val_scores)
            best_val_labels = list(val_labels)

    print(f"\nBest model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")

    # Threshold sweep on the best checkpoint's validation scores (not last epoch).
    print("\nThreshold sweep on validation set (best checkpoint):")
    best_thresh = sweep_threshold(best_val_scores or val_scores,
                                  best_val_labels or val_labels)

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump({
            "base_model": base_model,
            "threshold": best_thresh,
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "max_length": max_length,
            "params_M": round(total_params / 1e6, 2),
            "backbone_params_M": round(backbone_params / 1e6, 2),
            "device": str(dev),
            "amp_bf16": use_amp,
            "epochs": epochs, "batch_size": batch_size,
            "lr_bert": lr_bert, "lr_head": lr_head,
            "trust_remote_code": trust_remote_code,
        }, f, indent=2)
    with open(save_dir / "history.json", "w") as f:
        json.dump({"history": history}, f, indent=2)

    print(f"Saved to {save_dir}")
    return model, best_thresh


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def sweep_threshold(scores, labels, target_fpr=0.05):
    """Sweep thresholds to find best F1 with FPR < target."""
    best_f1, best_thresh = 0, 0.5
    for thresh in [t / 100 for t in range(20, 80)]:
        tp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 1.0)
        fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 0.0)
        fn = sum(1 for s, l in zip(scores, labels) if s < thresh and l == 1.0)
        tn = sum(1 for s, l in zip(scores, labels) if s < thresh and l == 0.0)

        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0

        marker = " *" if fpr <= target_fpr and f1 > best_f1 else ""
        if thresh in (0.3, 0.4, 0.5, 0.6, 0.7):
            print(f"  t={thresh:.2f}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} FPR={fpr:.3f}{marker}")

        if fpr <= target_fpr and f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n  Best threshold: {best_thresh:.2f} (F1={best_f1:.3f}, FPR≤{target_fpr})")
    return best_thresh


# ---------------------------------------------------------------------------
# Inference (for benchmark integration)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_threshold = 0.5


def _ensure_loaded():
    global _model, _tokenizer, _threshold
    if _model is not None:
        return

    config_path = SAVE_DIR_DEFAULT / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model", "sentence-transformers/all-MiniLM-L6-v2")
        _threshold = cfg.get("threshold", 0.5)
    else:
        base_model = "sentence-transformers/all-MiniLM-L6-v2"

    _tokenizer = AutoTokenizer.from_pretrained(base_model)
    _model = InjectionClassifier(base_model)
    ckpt = torch.load(SAVE_DIR_DEFAULT / "best_model.pt", map_location="cpu", weights_only=True)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()


def classify(text: str) -> bool:
    """Classify text for indirect injection. Returns True if injection."""
    _ensure_loaded()
    encoded = _tokenizer(
        text[:2000], padding="max_length", truncation=True,
        max_length=MAX_LENGTH_DEFAULT, return_tensors="pt",
    )
    with torch.no_grad():
        logit = _model(encoded["input_ids"], encoded["attention_mask"])
        score = torch.sigmoid(logit).item()
    return score >= _threshold


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(output_dir: Path | None = None):
    """Export trained model to ONNX + int8 quantization."""
    _ensure_loaded()
    out = output_dir or (SAVE_DIR_DEFAULT / "onnx")
    out.mkdir(parents=True, exist_ok=True)

    dummy_ids = torch.zeros(1, MAX_LENGTH_DEFAULT, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_LENGTH_DEFAULT, dtype=torch.long)

    onnx_path = out / "model.onnx"
    torch.onnx.export(
        _model,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logit"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logit": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f}MB)")

    # Quantize
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = out / "model_quantized.onnx"
        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
        print(f"Quantized: {quant_path} ({quant_path.stat().st_size / 1e6:.1f}MB)")
    except ImportError:
        print("  [warn] onnxruntime.quantization not available, skipping quantization")

    # Save tokenizer
    _tokenizer.save_pretrained(str(out))
    print(f"Tokenizer saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR_DEFAULT,
                        help="Where to write best_model.pt / config.json / history.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT,
                        help="Tokenizer max_length (default 256). Raise for long-context models.")
    parser.add_argument("--lr-bert", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, help="Limit samples (for smoke tests)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to AutoModel/AutoTokenizer (e.g. gte-base)")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    model, threshold = train_model(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_bert=args.lr_bert,
        lr_head=args.lr_head,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        device=args.device,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
    )

    if args.export_onnx:
        export_onnx(output_dir=args.save_dir / "onnx")
