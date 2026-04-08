"""Dataset loading functions for prompt injection benchmarks."""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

# Import split utility for consistent train/test partitioning
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "indirect"))
from split_utils import is_test_partition


def _load_binary(hf_id, text_col="text", label_col="label", injection_label=1,
                 split=None, test_only=False):
    if split:
        ds = load_dataset(hf_id, split=split)
        rows = list(ds)
    else:
        ds = load_dataset(hf_id)
        rows = [r for s in ds for r in ds[s]]
    samples = [{"text": r[text_col], "is_injection": r[label_col] == injection_label} for r in rows]
    if test_only:
        samples = [s for s in samples if is_test_partition(s["text"])]
    return samples


def _load_jailbreakbench():
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    return ([{"text": r["Goal"], "is_injection": True} for r in ds["harmful"]] +
            [{"text": r["Goal"], "is_injection": False} for r in ds["benign"]])


def _load_bipia():
    """Load BIPIA indirect injection dataset from local repo.

    Generates poisoned samples by inserting attack strings into benign contexts.
    Uses a fixed seed for reproducibility.
    """
    bipia_dir = Path(__file__).parent.parent.parent / "indirect" / "bipia_repo" / "benchmark"

    with open(bipia_dir / "text_attack_test.json") as f:
        attacks_raw = json.load(f)
    attacks = []
    for variants in attacks_raw.values():
        attacks.extend(variants)

    rng = random.Random(42)  # Fixed seed for reproducibility
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
    """Load xxz224 prompt-injection-attack-dataset (~3747 rows)."""
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
        samples = [s for s in samples if is_test_partition(s["text"])]
    return samples


def _load_notinject():
    """Load NotInject dataset — benign prompts with trigger words (FPR test)."""
    ds = load_dataset("leolee99/NotInject")
    samples = []
    for split in ds:
        for r in ds[split]:
            samples.append({"text": r["prompt"], "is_injection": False})
    return samples


def _load_hackaprompt():
    """Load HackAPrompt competition dataset — real attack submissions."""
    ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
    samples = []
    for r in ds:
        prompt = r.get("user_input", "")
        if not prompt or len(prompt.strip()) < 10:
            continue
        # All submissions are attack attempts
        samples.append({"text": prompt, "is_injection": True})
    # Cap at 5000 to keep benchmark tractable
    if len(samples) > 5000:
        import random
        rng = random.Random(42)
        samples = rng.sample(samples, 5000)
    return samples


def _load_harmbench():
    """Load HarmBench jailbreak behaviors."""
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
      hf_id: HuggingFace dataset ID
      text_col: column name for text (default: "text")
      label_col: column name for label (optional — if absent, all samples are injection or benign per all_injection/all_benign)
      injection_label: value in label_col that means injection (default: 1)
      all_injection: if true, all samples are injection (no label_col needed)
      all_benign: if true, all samples are benign (FPR test)
      split: HF split to load (default: "train")
      config_name: HF dataset config name (optional)
      max_samples: cap sample count (default: 5000 for large datasets)
    """
    hf_id = ds_cfg["hf_id"]
    text_col = ds_cfg.get("text_col", "text")
    label_col = ds_cfg.get("label_col")
    injection_label = ds_cfg.get("injection_label", 1)
    all_injection = ds_cfg.get("all_injection", False)
    all_benign = ds_cfg.get("all_benign", False)
    split = ds_cfg.get("split", "train")
    config_name = ds_cfg.get("config_name")
    max_samples = ds_cfg.get("max_samples", 5000)

    if config_name:
        ds = load_dataset(hf_id, config_name, split=split)
    else:
        try:
            ds = load_dataset(hf_id, split=split)
        except Exception:
            # Some datasets don't have a "train" split
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
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    samples = []
    for r in ds:
        text = r.get("user_input", "")
        if not text:
            continue
        is_inj = bool(r.get("jailbreaking", 0))
        samples.append({"text": text, "is_injection": is_inj})
    if len(samples) > 5000:
        rng = random.Random(42)
        samples = rng.sample(samples, 5000)
    return samples


def _load_wildguardtest() -> list[dict]:
    """Load walledai/WildGuardTest. Vanilla + adversarial prompts."""
    ds = load_dataset("walledai/WildGuardTest", split="train")
    samples = []
    for r in ds:
        text = r.get("prompt", "")
        if not text:
            continue
        # "harmful" label = injection-like behavior
        is_inj = r.get("label", "") == "harmful" or r.get("adversarial", False)
        samples.append({"text": text, "is_injection": is_inj})
    return samples


def _load_in_the_wild_jailbreak() -> list[dict]:
    """Load TrustAIRLab/in-the-wild-jailbreak-prompts. CCS 2024."""
    ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                       "jailbreak_2023_12_25", split="train")
    samples = []
    for r in ds:
        text = r.get("prompt", "")
        if not text:
            continue
        is_inj = bool(r.get("jailbreak", False))
        samples.append({"text": text, "is_injection": is_inj})
    if len(samples) > 5000:
        rng = random.Random(42)
        samples = rng.sample(samples, 5000)
    return samples


def load_dataset_samples(ds_name: str, ds_cfg: dict) -> list[dict]:
    """Load dataset samples from config entry."""
    loader = ds_cfg["loader"]

    if loader == "binary":
        return _load_binary(
            ds_cfg["hf_id"],
            split=ds_cfg.get("split"),
            test_only=ds_cfg.get("test_only", False),
        )
    elif loader == "generic_hf":
        return _load_generic_hf(ds_cfg)
    elif loader == "jailbreakbench":
        return _load_jailbreakbench()
    elif loader == "bipia":
        return _load_bipia()
    elif loader == "xxz224":
        return _load_xxz224(test_only=ds_cfg.get("test_only", False))
    elif loader == "notinject":
        return _load_notinject()
    elif loader == "hackaprompt":
        return _load_hackaprompt()
    elif loader == "harmbench":
        return _load_harmbench()
    elif loader == "toxic_chat":
        return _load_toxic_chat()
    elif loader == "wildguardtest":
        return _load_wildguardtest()
    elif loader == "in_the_wild_jailbreak":
        return _load_in_the_wild_jailbreak()
    else:
        raise ValueError(f"Unknown dataset loader '{loader}' for {ds_name}")
