"""Generate markdown report from benchmark results. Tables only, no images."""
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "result"

CATEGORY_ORDER = ["direct", "indirect", "jailbreak", "fpr", "safety"]
CATEGORY_TITLES = {
    "direct": "Direct Prompt Injection",
    "indirect": "Indirect Prompt Injection",
    "jailbreak": "Jailbreak Detection",
    "fpr": "False Positive Analysis",
    "safety": "Safety / Content Moderation (FPR Test)",
}

HARDWARE = "Apple Silicon M3 (MPS GPU)"


def _fmt_pct(v):
    """Format a 0-1 float as percentage string."""
    return f"{v*100:.1f}%"


def _generate_findings(df: pd.DataFrame, ds_categories: dict[str, str]) -> str:
    """Generate Key Findings section dynamically from data."""
    lines = ["## Key Findings\n"]
    # FPR/safety datasets are all-benign — F1 is undefined, exclude from F1-based analysis
    fpr_cats = {"fpr", "safety"}
    fpr_datasets = {ds for ds, cat in ds_categories.items() if cat in fpr_cats}
    attack_df = df[~df["dataset"].isin(fpr_datasets)]

    # 1. Best model per attack category
    for cat in ["direct", "indirect", "jailbreak"]:
        cat_ds = [ds for ds, c in ds_categories.items() if c == cat]
        sub = attack_df[attack_df["dataset"].isin(cat_ds)]
        if sub.empty:
            continue
        avg_f1 = sub.groupby("model")["f1"].mean().sort_values(ascending=False)
        best = avg_f1.index[0]
        title = CATEGORY_TITLES.get(cat, cat)
        lines.append(f"- **Best {title}:** {best} (avg F1 {avg_f1.iloc[0]:.3f})")

    # 2. Lowest FPR on notinject (dedicated FPR dataset)
    ni = df[df["dataset"] == "notinject"]
    if not ni.empty:
        lowest_fpr = ni.sort_values("fpr")
        best_fpr_model = lowest_fpr.iloc[0]
        lines.append(
            f"- **Lowest FPR (notinject):** {best_fpr_model['model']} "
            f"({best_fpr_model['fpr']:.1%})"
        )

    # 3. High FPR warning
    high_fpr = df[df["fpr"] > 0.5][["model", "dataset"]].drop_duplicates()
    if not high_fpr.empty:
        n_models = high_fpr["model"].nunique()
        n_total = df["model"].nunique()
        lines.append(
            f"- **High FPR crisis:** {n_models}/{n_total} models have >50% FPR "
            f"on at least one dataset"
        )

    # 4. Best generalization (highest min F1 across attack datasets only)
    if not attack_df.empty:
        min_f1 = attack_df.groupby("model")["f1"].min().sort_values(ascending=False)
        if not min_f1.empty and min_f1.iloc[0] > 0:
            lines.append(
                f"- **Best generalization:** {min_f1.index[0]} "
                f"(min F1 across attack datasets: {min_f1.iloc[0]:.3f})"
            )
        else:
            lines.append("- **No model generalizes** — every model has F1=0 on at least one attack dataset")

    # 5. Indirect PI (BIPIA) status
    indirect_ds = [ds for ds, c in ds_categories.items() if c == "indirect"]
    indirect_df = df[df["dataset"].isin(indirect_ds)]
    if not indirect_df.empty:
        top = indirect_df.groupby("model")["f1"].mean().sort_values(ascending=False)
        if top.iloc[0] > 0.9:
            lines.append(f"- **Indirect PI solved:** {top.index[0]} (F1 {top.iloc[0]:.3f})")
        else:
            lines.append(
                f"- **Indirect PI unsolved:** best is {top.index[0]} "
                f"(F1 {top.iloc[0]:.3f}), no model exceeds 0.9"
            )

    # 6. Simple baselines competitive?
    simple = attack_df[attack_df["model"].isin(["tfidf-logreg", "pattern-detector"])]
    transformer = attack_df[~attack_df["model"].isin(["tfidf-logreg", "pattern-detector"])]
    if not simple.empty and not transformer.empty:
        simple_avg = simple.groupby("model")["f1"].mean()
        transformer_avg = transformer.groupby("model")["f1"].mean()
        n_beaten = (transformer_avg < simple_avg.max()).sum()
        if n_beaten > 0:
            lines.append(
                f"- **Simple baselines competitive:** tfidf-logreg (avg F1 "
                f"{simple_avg.get('tfidf-logreg', 0):.3f}) beats {n_beaten} "
                f"transformer model(s)"
            )

    lines.append("")
    return "\n".join(lines)


def generate_report(df: pd.DataFrame, ds_categories: dict[str, str]):
    """Generate REPORT.md with tables only."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("# Prompt Injection Detection — Benchmark Report\n")
    lines.append(f"**Updated:** {time.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Hardware:** {HARDWARE}\n")

    lines.append(_generate_findings(df, ds_categories))

    # --- Overview: heatmap as a markdown table (exclude FPR/safety — all-benign, F1 undefined) ---
    lines.append("## Overview — F1 Heatmap\n")
    fpr_safety_datasets = {ds for ds, cat in ds_categories.items() if cat in ("fpr", "safety")}
    heatmap_df = df[~df["dataset"].isin(fpr_safety_datasets)]
    pivot = heatmap_df.pivot_table(index="model", columns="dataset", values="f1", aggfunc="first")
    if not pivot.empty:
        pivot["_avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_avg", ascending=False)
        pivot = pivot.drop(columns=["_avg"])
        fmt = pivot.map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        lines.append(fmt.to_markdown())
        lines.append("")

    # --- Speed-Quality table ---
    lines.append("## Speed–Quality Tradeoff\n")
    attack_df = df[~df["dataset"].isin(fpr_safety_datasets)]

    has_throughput = "throughput_samples_per_s" in df.columns

    agg_dict = {
        "direct_pi_f1": ("f1", "mean"),
        "min_f1": ("f1", "min"),
        "datasets_tested": ("dataset", "count"),
    }
    if has_throughput:
        agg_dict["avg_throughput"] = ("throughput_samples_per_s", "mean")

    attack_avg = attack_df.groupby(["model", "params"]).agg(**agg_dict).reset_index()

    # Add NotInject FPR separately
    notinject = df[df["dataset"] == "notinject"][["model", "fpr"]].rename(
        columns={"fpr": "notinject_fpr"}
    )
    avg = attack_avg.merge(notinject, on="model", how="left")

    if has_throughput:
        avg = avg.sort_values("avg_throughput", ascending=False)
    else:
        avg = avg.sort_values("direct_pi_f1", ascending=False)

    avg["direct_pi_f1"] = avg["direct_pi_f1"].apply(lambda v: f"{v:.3f}")
    avg["min_f1"] = avg["min_f1"].apply(lambda v: f"{v:.3f}")
    avg["notinject_fpr"] = avg["notinject_fpr"].apply(
        lambda v: _fmt_pct(v) if pd.notna(v) else "—"
    )

    if has_throughput:
        avg["avg_throughput"] = avg["avg_throughput"].apply(lambda v: f"{v:.1f}")
        cols = [
            "model", "params", "direct_pi_f1", "min_f1",
            "avg_throughput", "notinject_fpr", "datasets_tested",
        ]
    else:
        cols = [
            "model", "params", "direct_pi_f1", "min_f1",
            "notinject_fpr", "datasets_tested",
        ]

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

        metric_cols = ["model", "params", "f1", "precision", "recall", "fpr"]
        if has_throughput:
            metric_cols.append("throughput_samples_per_s")

        for ds_name in cat_datasets:
            ds_sub = sub[sub["dataset"] == ds_name].sort_values("f1", ascending=False)
            if ds_sub.empty:
                continue
            n_samples = int(ds_sub.iloc[0]["samples"])
            lines.append(f"### {ds_name} ({n_samples} samples)\n")
            available = [c for c in metric_cols if c in ds_sub.columns]
            lines.append(ds_sub[available].to_markdown(index=False))
            lines.append("")

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
    has_throughput = "throughput_samples_per_s" in df.columns

    with open(RESULTS_DIR / "SUMMARY.md", "w") as f:
        f.write("# Prompt Injection Detection Benchmark Results\n\n")
        f.write(f"**Updated:** {time.strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Hardware:** {HARDWARE}\n\n")

        for ds_name in df["dataset"].unique():
            sub = df[df["dataset"] == ds_name].sort_values("f1", ascending=False)
            f.write(f"### {ds_name} ({int(sub.iloc[0]['samples'])} samples)\n\n")
            cols = ["model", "params", "f1", "precision", "recall", "fpr"]
            if has_throughput:
                cols.append("throughput_samples_per_s")
            available = [c for c in cols if c in sub.columns]
            f.write(sub[available].to_markdown(index=False))
            f.write("\n\n")

    print(f"Summary → {RESULTS_DIR / 'SUMMARY.md'}")
