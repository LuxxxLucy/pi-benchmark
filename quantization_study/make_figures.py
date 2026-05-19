"""Emit fig1 (accuracy degrade) and fig2 (speedup) from accuracy.json + latency.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRECISION_ORDER = ["fp32", "bf16", "pt_int8_dynamic", "onnx_int8_dynamic"]
PRECISION_LABELS = {
    "fp32": "fp32",
    "bf16": "PT bf16 autocast",
    "pt_int8_dynamic": "PT int8 dynamic",
    "onnx_int8_dynamic": "ONNX int8 dynamic",
}
PRECISION_COLORS = {
    "fp32": "#bdbdbd",
    "bf16": "#e08a3c",
    "pt_int8_dynamic": "#5a7eb4",
    "onnx_int8_dynamic": "#274472",
}


def _index(rows: list[dict], key: str) -> dict[tuple[str, str], dict]:
    return {(r["model"], r["precision"]): r for r in rows if key in r}


def _grouped_bar(ax, models: list[str], precisions: list[str],
                 values: dict[tuple[str, str], float], ylabel: str, title: str,
                 ymax: float | None = None,
                 annotate: dict[tuple[str, str], str] | None = None) -> None:
    width = 0.8 / max(1, len(precisions))
    x = np.arange(len(models))
    for i, precision in enumerate(precisions):
        ys = [values.get((m, precision), 0.0) for m in models]
        bars = ax.bar(
            x + (i - (len(precisions) - 1) / 2) * width,
            ys, width,
            label=PRECISION_LABELS[precision],
            color=PRECISION_COLORS[precision],
        )
        if annotate is not None:
            for bar, m in zip(bars, models):
                txt = annotate.get((m, precision))
                if txt:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            txt, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")


def fig1_accuracy(accuracy_rows: list[dict], out_path: Path) -> None:
    models = sorted({r["model"] for r in accuracy_rows if "composite_f1" in r})
    composite = _index(accuracy_rows, "composite_f1")
    composite_vals = {k: v["composite_f1"] for k, v in composite.items()}
    annotate = {}
    for (m, p), row in composite.items():
        if p == "fp32":
            continue
        delta = row.get("delta_composite_f1_vs_fp32")
        if delta is not None:
            annotate[(m, p)] = f"{delta:+.3f}"

    fig, (ax_main, ax_cat) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    _grouped_bar(
        ax_main, models, PRECISION_ORDER, composite_vals,
        ylabel="Composite F1 (Direct + Indirect + Jailbreak)",
        title="PI-defender composite F1 by precision",
        ymax=1.0, annotate=annotate,
    )

    categories = ["direct_f1", "indirect_f1", "jailbreak_f1"]
    cat_labels = ["Direct", "Indirect", "Jailbreak"]
    x = np.arange(len(cat_labels))
    width = 0.8 / max(1, len(PRECISION_ORDER))
    for i, precision in enumerate(PRECISION_ORDER):
        ys = []
        for c in categories:
            vals = [r[c] for r in accuracy_rows
                    if r.get("precision") == precision and c in r]
            ys.append(float(np.mean(vals)) if vals else 0.0)
        ax_cat.bar(
            x + (i - (len(PRECISION_ORDER) - 1) / 2) * width, ys, width,
            label=PRECISION_LABELS[precision], color=PRECISION_COLORS[precision],
        )
    ax_cat.set_xticks(x)
    ax_cat.set_xticklabels(cat_labels)
    ax_cat.set_ylim(0, 1.0)
    ax_cat.set_ylabel("Mean per-category F1 across models")
    ax_cat.set_title("Per-category F1 (model-averaged)")
    ax_cat.grid(axis="y", linestyle=":", alpha=0.5)
    ax_cat.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig2_speedup(latency_rows: list[dict], out_path: Path) -> None:
    models = sorted({r["model"] for r in latency_rows if "p50_ms" in r})
    speedup_vals: dict[tuple[str, str], float] = {}
    annotate: dict[tuple[str, str], str] = {}
    fp32_p50 = {r["model"]: r["p50_ms"]
                for r in latency_rows
                if r.get("precision") == "fp32" and "p50_ms" in r}
    for r in latency_rows:
        if "p50_ms" not in r:
            continue
        m, p = r["model"], r["precision"]
        base = fp32_p50.get(m, r["p50_ms"])
        speedup_vals[(m, p)] = base / r["p50_ms"] if r["p50_ms"] > 0 else 0.0
        annotate[(m, p)] = f"{r['p50_ms']:.1f}ms"

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    _grouped_bar(
        ax, models, PRECISION_ORDER, speedup_vals,
        ylabel="Speedup vs fp32 PyTorch  (P50 ratio)",
        title="PI-defender CPU latency speedup at length 256",
        annotate=annotate,
    )
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results",
                    help="Directory holding accuracy.json + latency.json")
    args = ap.parse_args()

    in_dir = (Path(__file__).resolve().parent / args.input).resolve()
    acc_path = in_dir / "accuracy.json"
    lat_path = in_dir / "latency.json"

    if acc_path.exists():
        acc_rows = json.loads(acc_path.read_text()).get("results", [])
        fig1_accuracy(acc_rows, in_dir / "fig1_accuracy_degrade.png")
    else:
        print(f"skip fig1: {acc_path} not found")

    if lat_path.exists():
        lat_rows = json.loads(lat_path.read_text()).get("results", [])
        fig2_speedup(lat_rows, in_dir / "fig2_speedup.png")
    else:
        print(f"skip fig2: {lat_path} not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
