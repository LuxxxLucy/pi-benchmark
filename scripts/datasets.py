"""Dataset loading with local disk cache.

First load downloads from HuggingFace, processes, and caches to
datasets/cached/{name}.json. Subsequent loads read from cache.
Use --rebuild-cache to force re-download.
"""
import hashlib
import json
import random
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "datasets" / "cached"


# ---------------------------------------------------------------------------
# Deterministic split (inlined from split_utils.py to avoid cross-project dep)
# ---------------------------------------------------------------------------

def _is_test_partition(text: str, test_fraction: float = 0.3, seed: int = 42) -> bool:
    """Deterministic hash-based split. Returns True if sample belongs to test set."""
    h = hashlib.md5(f"{seed}:{text}".encode()).hexdigest()
    return (int(h, 16) % 1000) < int(test_fraction * 1000)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(ds_name: str) -> Path:
    return CACHE_DIR / f"{ds_name}.json"


def _load_from_cache(ds_name: str) -> list[dict] | None:
    path = _cache_path(ds_name)
    if path.exists():
        with open(path) as f:
            samples = json.load(f)
        print(f"  [{ds_name}] loaded {len(samples)} samples from cache")
        return samples
    return None


def _save_to_cache(ds_name: str, samples: list[dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(ds_name), "w") as f:
        json.dump(samples, f)
    print(f"  [{ds_name}] cached {len(samples)} samples → {_cache_path(ds_name)}")


# ---------------------------------------------------------------------------
# HuggingFace loaders
# ---------------------------------------------------------------------------

def _load_binary(hf_id, text_col="text", label_col="label", injection_label=1,
                 split=None, test_only=False):
    from datasets import load_dataset
    if split:
        ds = load_dataset(hf_id, split=split)
        rows = list(ds)
    else:
        ds = load_dataset(hf_id)
        rows = [r for s in ds for r in ds[s]]
    samples = [{"text": r[text_col], "is_injection": r[label_col] == injection_label}
               for r in rows]
    if test_only:
        samples = [s for s in samples if _is_test_partition(s["text"])]
    return samples


def _load_jailbreakbench():
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    return ([{"text": r["Goal"], "is_injection": True} for r in ds["harmful"]] +
            [{"text": r["Goal"], "is_injection": False} for r in ds["benign"]])


def _load_bipia():
    """Load BIPIA indirect injection dataset from local repo.

    Generates poisoned samples by inserting attack strings into benign contexts.
    Uses a fixed seed for reproducibility.
    """
    bipia_dir = Path(__file__).parent.parent / "datasets" / "bipia_repo" / "benchmark"

    with open(bipia_dir / "text_attack_test.json") as f:
        attacks_raw = json.load(f)
    attacks = []
    for variants in attacks_raw.values():
        attacks.extend(variants)

    rng = random.Random(42)
    samples = []
    for category in ["email", "code", "table"]:
        cat_dir = bipia_dir / category
        test_file = cat_dir / "test.jsonl"
        if not test_file.exists():
            continue
        with open(test_file) as f:
            for line in f:
                row = json.loads(line)
                context = row.get("context", "")
                if isinstance(context, list):
                    context = "\n".join(str(c) for c in context)
                if not context:
                    continue
                samples.append({"text": context, "is_injection": False})
                attack = rng.choice(attacks)
                poisoned = context + "\n\n" + attack
                samples.append({"text": poisoned, "is_injection": True})

    rng.shuffle(samples)
    return samples


def _load_xxz224(test_only=False):
    """Load xxz224 prompt-injection-attack-dataset."""
    from datasets import load_dataset
    ds = load_dataset("xxz224/prompt-injection-attack-dataset", split="train")
    attack_cols = ["naive_attack", "escape_attack", "ignore_attack",
                   "fake_comp_attack", "combine_attack"]
    samples = []
    seen_benign = set()
    for r in ds:
        txt = r["target_text"]
        if txt not in seen_benign:
            seen_benign.add(txt)
            samples.append({"text": txt, "is_injection": False})
        for col in attack_cols:
            samples.append({"text": r[col], "is_injection": True})
    if test_only:
        samples = [s for s in samples if _is_test_partition(s["text"])]
    return samples


def _load_notinject():
    """Load NotInject dataset — benign prompts with trigger words (FPR test)."""
    from datasets import load_dataset
    ds = load_dataset("leolee99/NotInject")
    samples = []
    for split in ds:
        for r in ds[split]:
            samples.append({"text": r["prompt"], "is_injection": False})
    return samples


def _load_hackaprompt():
    """Load HackAPrompt competition dataset — real attack submissions."""
    from datasets import load_dataset
    ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
    samples = []
    for r in ds:
        prompt = r.get("user_input", "")
        if not prompt or len(prompt.strip()) < 10:
            continue
        samples.append({"text": prompt, "is_injection": True})
    return samples


def _load_harmbench():
    """Load HarmBench jailbreak behaviors."""
    from datasets import load_dataset
    ds = load_dataset("walledai/HarmBench", split="train")
    samples = []
    for r in ds:
        text = r.get("prompt", "") or r.get("behavior", "")
        if text:
            samples.append({"text": text, "is_injection": True})
    return samples


def _load_generic_hf(ds_cfg: dict) -> list[dict]:
    """Generic HuggingFace dataset loader driven by config fields.

    Config fields:
      hf_id, text_col, label_col, injection_label,
      all_injection, all_benign, split, config_name, max_samples
    """
    from datasets import load_dataset

    hf_id = ds_cfg["hf_id"]
    text_col = ds_cfg.get("text_col", "text")
    label_col = ds_cfg.get("label_col")
    injection_label = ds_cfg.get("injection_label", 1)
    all_injection = ds_cfg.get("all_injection", False)
    all_benign = ds_cfg.get("all_benign", False)
    split = ds_cfg.get("split", "train")
    config_name = ds_cfg.get("config_name")
    max_samples = ds_cfg.get("max_samples")

    if config_name:
        ds = load_dataset(hf_id, config_name, split=split)
    else:
        try:
            ds = load_dataset(hf_id, split=split)
        except Exception:
            ds_full = load_dataset(hf_id)
            rows = [r for s in ds_full for r in ds_full[s]]
            ds = rows

    rows = list(ds) if not isinstance(ds, list) else ds
    samples = []
    for r in rows:
        text = r.get(text_col, "")
        if not text or not isinstance(text, str):
            continue

        if all_injection:
            samples.append({"text": text, "is_injection": True})
        elif all_benign:
            samples.append({"text": text, "is_injection": False})
        elif label_col:
            label = r.get(label_col)
            samples.append({"text": text, "is_injection": label == injection_label})
        else:
            samples.append({"text": text, "is_injection": True})

    if max_samples and len(samples) > max_samples:
        rng = random.Random(42)
        samples = rng.sample(samples, max_samples)

    return samples


def _load_toxic_chat() -> list[dict]:
    """Load lmsys/toxic-chat. Uses jailbreaking column as injection label."""
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    samples = []
    for r in ds:
        text = r.get("user_input", "")
        if not text:
            continue
        is_inj = bool(r.get("jailbreaking", 0))
        samples.append({"text": text, "is_injection": is_inj})
    return samples


def _load_wildguardtest() -> list[dict]:
    """Load walledai/WildGuardTest. Vanilla + adversarial prompts.

    Only uses the 'harmful' label — NOT the 'adversarial' flag, because
    adversarial benign prompts (designed to look harmful but are safe)
    should not be labeled as injection.
    """
    from datasets import load_dataset
    ds = load_dataset("walledai/WildGuardTest", split="train")
    samples = []
    for r in ds:
        text = r.get("prompt", "")
        if not text:
            continue
        is_inj = r.get("label", "") == "harmful"
        samples.append({"text": text, "is_injection": is_inj})
    return samples


def _load_in_the_wild_jailbreak() -> list[dict]:
    """Load TrustAIRLab/in-the-wild-jailbreak-prompts. CCS 2024."""
    from datasets import load_dataset
    ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                      "jailbreak_2023_12_25", split="train")
    samples = []
    for r in ds:
        text = r.get("prompt", "")
        if not text:
            continue
        is_inj = bool(r.get("jailbreak", False))
        samples.append({"text": text, "is_injection": is_inj})
    return samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset_samples(ds_name: str, ds_cfg: dict,
                         rebuild_cache: bool = False) -> list[dict]:
    """Load dataset samples, using local cache when available.

    First call downloads from HuggingFace, processes, and caches to disk.
    Subsequent calls load from cache (fast, no network).
    Pass rebuild_cache=True to force re-download.
    """
    if not rebuild_cache:
        cached = _load_from_cache(ds_name)
        if cached is not None:
            return cached

    print(f"  [{ds_name}] downloading from HuggingFace...")
    loader = ds_cfg["loader"]

    if loader == "binary":
        samples = _load_binary(
            ds_cfg["hf_id"],
            split=ds_cfg.get("split"),
            test_only=ds_cfg.get("test_only", False),
        )
    elif loader == "generic_hf":
        samples = _load_generic_hf(ds_cfg)
    elif loader == "jailbreakbench":
        samples = _load_jailbreakbench()
    elif loader == "bipia":
        samples = _load_bipia()
    elif loader == "xxz224":
        samples = _load_xxz224(test_only=ds_cfg.get("test_only", False))
    elif loader == "notinject":
        samples = _load_notinject()
    elif loader == "hackaprompt":
        samples = _load_hackaprompt()
    elif loader == "harmbench":
        samples = _load_harmbench()
    elif loader == "toxic_chat":
        samples = _load_toxic_chat()
    elif loader == "wildguardtest":
        samples = _load_wildguardtest()
    elif loader == "in_the_wild_jailbreak":
        samples = _load_in_the_wild_jailbreak()
    else:
        raise ValueError(f"Unknown dataset loader '{loader}' for {ds_name}")

    _save_to_cache(ds_name, samples)
    return samples
