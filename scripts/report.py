"""Generate markdown report from benchmark results. Tables only, no images."""
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"

CATEGORY_ORDER = ["direct", "indirect", "jailbreak", "fpr", "safety"]
CATEGORY_TITLES = {
    "direct": "Direct Prompt Injection",
    "indirect": "Indirect Prompt Injection",
    "jailbreak": "Jailbreak Detection",
    "fpr": "False Positive Analysis",
    "safety": "Safety / Content Moderation (FPR Test)",
}

KEY_FINDINGS = """\
## Key Findings

1. **No model generalizes across all attack categories.** The best direct-injection detectors
   collapse on indirect injection (BIPIA), and vice versa. Models trained on one distribution
   fail silently on others — the heatmap below makes this stark.

2. **Simple baselines are shockingly competitive.** TF-IDF + logistic regression (<1M params,
   0.3ms latency) matches or beats 66–307M transformer models on 4 of 7 core datasets.
   This challenges the assumption that prompt injection detection requires large models.

3. **False positive rates are catastrophic.** Most high-recall models achieve their scores by
   classifying everything as injection. fmops-distilbert has 72% FPR on NotInject and 100% on
   BIPIA benign. In production, this means blocking legitimate user traffic.

4. **Indirect injection remains unsolved.** On BIPIA (the only indirect injection benchmark),
   most models score near zero. Only models explicitly trained on indirect data show any signal.
"""

RECOMMENDATIONS = """\
## Deployment Recommendations

| Latency Budget | Recommended Model | Tradeoff |
|---|---|---|
| **< 1ms** | tfidf-logreg or pattern-detector | Good on easy datasets, misses subtle attacks |
| **< 15ms** | stackone-repro (where available) | Best avg F1, but 55% FPR on NotInject |
| **< 100ms** | protectai-base-v2 | Best on xTRam1 (F1=0.91), weak on indirect |
| **Indirect PI** | minilm-indirect-v1 | Only model with BIPIA signal (F1=0.77), but 51% FPR |

**No single model is production-ready across all attack categories.** A tiered pipeline
(fast pattern filter → ML classifier) with dataset-specific thresholds is likely needed.

## What's Next

- Benchmark additional pre-trained HF models (testsavantai suite, hlyn-deberta-70m, deepset-deberta, etc.)
- Fine-tune lightweight models (ELECTRA-small 14M, DeBERTa-v3-xsmall 22M) with PEFT/LoRA on mixed data
- Train v2 indirect injection model to fix BIPIA FPR (51.5% → target <10%)
- Add gated datasets (HackAPrompt, WildJailbreak, HarmBench) when access is granted
"""


def _fmt_pct(v):
    """Format a 0-1 float as percentage string."""
    return f"{v*100:.1f}%"


def generate_report(df: pd.DataFrame, ds_categories: dict[str, str]):
    """Generate REPORT.md with tables only."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("# Prompt Injection Detection — Benchmark Report\n")
    lines.append(f"**Updated:** {time.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Hardware:** CPU (Apple Silicon M3)\n")

    lines.append(KEY_FINDINGS)

    # --- Overview: heatmap as a markdown table (exclude FPR-only datasets) ---
    lines.append("## Overview — F1 Heatmap\n")
    fpr_only_datasets = {ds for ds, cat in ds_categories.items() if cat == "fpr"}
    heatmap_df = df[~df["dataset"].isin(fpr_only_datasets)]
    pivot = heatmap_df.pivot_table(index="model", columns="dataset", values="f1", aggfunc="first")
    if not pivot.empty:
        pivot["_avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_avg", ascending=False)
        pivot = pivot.drop(columns=["_avg"])
        fmt = pivot.map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        lines.append(fmt.to_markdown())
        lines.append("")

    # --- Speed-Quality table (exclude FPR-only datasets from F1 average) ---
    lines.append("## Speed–Quality Tradeoff\n")

    # Exclude FPR-only datasets (e.g. notinject) from F1 averages
    fpr_only_datasets = {ds for ds, cat in ds_categories.items() if cat == "fpr"}
    attack_df = df[~df["dataset"].isin(fpr_only_datasets)]

    # Compute per-model composite scores
    attack_avg = attack_df.groupby(["model", "params"]).agg(
        direct_pi_f1=("f1", "mean"),
        min_f1=("f1", "min"),
        avg_latency=("latency_p50_ms", "mean"),
        datasets_tested=("dataset", "count"),
    ).reset_index()

    # Add NotInject FPR separately
    notinject = df[df["dataset"] == "notinject"][["model", "fpr"]].rename(columns={"fpr": "notinject_fpr"})
    avg = attack_avg.merge(notinject, on="model", how="left")
    avg = avg.sort_values("avg_latency")

    # Mark Pareto-optimal (on direct_pi_f1 vs latency)
    best_f1 = -1
    pareto = []
    for _, row in avg.iterrows():
        if row["direct_pi_f1"] > best_f1:
            pareto.append(row["model"])
            best_f1 = row["direct_pi_f1"]

    avg["pareto"] = avg["model"].apply(lambda m: "**yes**" if m in pareto else "")
    avg["direct_pi_f1"] = avg["direct_pi_f1"].apply(lambda v: f"{v:.3f}")
    avg["min_f1"] = avg["min_f1"].apply(lambda v: f"{v:.3f}")
    avg["avg_latency"] = avg["avg_latency"].apply(lambda v: f"{v:.1f}ms")
    avg["notinject_fpr"] = avg["notinject_fpr"].apply(
        lambda v: _fmt_pct(v) if pd.notna(v) else "—")
    cols = ["model", "params", "direct_pi_f1", "min_f1", "avg_latency",
            "notinject_fpr", "datasets_tested", "pareto"]
    lines.append(avg[cols].to_markdown(index=False))
    lines.append("")

    # --- Per-category sections ---
    for cat in CATEGORY_ORDER:
        cat_datasets = [ds for ds, c in ds_categories.items() if c == cat]
        sub = df[df["dataset"].isin(cat_datasets)]
        if sub.empty:
            continue

        title = CATEGORY_TITLES.get(cat, cat)
        lines.append(f"## {title}\n")

        for ds_name in cat_datasets:
            ds_sub = sub[sub["dataset"] == ds_name].sort_values("f1", ascending=False)
            if ds_sub.empty:
                continue
            n_samples = int(ds_sub.iloc[0]["samples"])
            lines.append(f"### {ds_name} ({n_samples} samples)\n")
            cols = ["model", "params", "f1", "precision", "recall", "fpr", "latency_p50_ms"]
            lines.append(ds_sub[cols].to_markdown(index=False))
            lines.append("")

    lines.append(RECOMMENDATIONS)

    with open(RESULTS_DIR / "REPORT.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Report → {RESULTS_DIR / 'REPORT.md'}")


def save_results_json(all_dicts: list[dict], results_json: Path):
    """Save raw results to JSON."""
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(all_dicts, f, indent=2)
    print(f"Results → {results_json}")


def generate_summary(df: pd.DataFrame):
    """Generate legacy SUMMARY.md (per-dataset tables)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "SUMMARY.md", "w") as f:
        f.write("# Prompt Injection Detection Benchmark Results\n\n")
        f.write(f"**Updated:** {time.strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Hardware:** CPU (Apple Silicon M3)\n\n")

        for ds_name in df["dataset"].unique():
            sub = df[df["dataset"] == ds_name].sort_values("f1", ascending=False)
            f.write(f"### {ds_name} ({int(sub.iloc[0]['samples'])} samples)\n\n")
            cols = ["model", "params", "f1", "precision", "recall", "fpr", "latency_p50_ms"]
            f.write(sub[cols].to_markdown(index=False))
            f.write("\n\n")

    print(f"Summary → {RESULTS_DIR / 'SUMMARY.md'}")
