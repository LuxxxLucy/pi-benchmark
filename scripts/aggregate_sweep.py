"""Aggregate lightweight-sweep eval JSONs into a single summary + markdown table.

Loads every <short_name>.json in --results-dir (output of scripts/eval_trained.py),
joins with arch-baseline latency from latency_cuda/results/latency_cuda.json,
and prints a single markdown table optimised for the Pareto question:

    Given fixed batch=1 latency, which architecture trains to the best F1
    and generalizes best to held-out + OOD test sets?
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


LATENCY_JSON = Path(__file__).resolve().parent.parent / "latency_cuda" / "results" / "latency_cuda.json"


def load_latency_map() -> dict[str, dict]:
    """short_name → {params_M, p50_ms@256, p50_ms@512}."""
    if not LATENCY_JSON.exists():
        print(f"[warn] {LATENCY_JSON} not found; latency column will be empty")
        return {}
    data = json.load(open(LATENCY_JSON))
    out = {}
    for r in data["results"]:
        if "lengths" not in r:
            continue
        name = r["name"]
        L = r["lengths"]
        out[name] = {
            "params_M": r.get("params_M"),
            "p50_ms@256": L.get("256", {}).get("p50_ms"),
            "p50_ms@512": L.get("512", {}).get("p50_ms"),
        }
    return out


def format_row(name: str, latency: dict, eval_json: dict | None) -> dict:
    row = {
        "model": name,
        "params_M": latency.get("params_M"),
        "gpu_p50_ms_256": latency.get("p50_ms@256"),
        "gpu_p50_ms_512": latency.get("p50_ms@512"),
    }
    if eval_json is None:
        row["status"] = "no eval json"
        return row
    cfg = eval_json.get("config", {})
    row["threshold"] = cfg.get("threshold")
    row["train_epochs"] = cfg.get("epochs")
    row["val_loss"] = cfg.get("val_loss")

    pd = eval_json.get("per_dataset", {})
    for ds_name in ["bipia_test", "deepset_test", "xxz224_test", "notinject"]:
        d = pd.get(ds_name, {})
        if d.get("skipped"):
            row[f"{ds_name}_f1"] = None
            row[f"{ds_name}_fpr"] = None
            continue
        m = d.get("at_best_threshold") or {}
        row[f"{ds_name}_f1"] = m.get("f1")
        row[f"{ds_name}_fpr"] = m.get("fpr")
        row[f"{ds_name}_n"] = m.get("n")

    # Eval-time latency sanity — CPU is primary, GPU is secondary.
    ls = eval_json.get("latency_sanity", {})
    cpu_ls = ls.get("cpu", {}) if isinstance(ls, dict) else {}
    gpu_ls = ls.get("gpu", {}) if isinstance(ls, dict) else {}
    row["eval_cpu_p50_ms"] = cpu_ls.get("p50_ms")
    row["eval_cpu_p95_ms"] = cpu_ls.get("p95_ms")
    row["eval_gpu_p50_ms"] = gpu_ls.get("p50_ms")
    row["eval_gpu_p95_ms"] = gpu_ls.get("p95_ms")
    # Backward-compat: some older JSONs mirrored the primary at top level.
    row["eval_latency_p50_ms"] = ls.get("p50_ms") or cpu_ls.get("p50_ms") or gpu_ls.get("p50_ms")
    return row


def markdown_table(rows: list[dict]) -> str:
    hdr = ["Model", "M", "CPU p50", "GPU p50", "val_loss",
           "BIPIA F1", "BIPIA FPR",
           "deepset F1", "deepset FPR",
           "xxz224 F1", "xxz224 FPR",
           "NotInject FPR"]
    aligns = ["left", "right", "right", "right", "right",
              "right", "right", "right", "right",
              "right", "right", "right"]

    def fmt(v, kind):
        if v is None:
            return "—"
        if kind == "f":
            return f"{v:.2f}"
        if kind == "f3":
            return f"{v:.3f}"
        if kind == "ms":
            return f"{v:.2f} ms"
        if kind == "pct":
            return f"{v * 100:.1f}%"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(":---" if a == "left" else "---:" for a in aligns) + "|")
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"],
            fmt(r.get("params_M"), "f"),
            fmt(r.get("eval_cpu_p50_ms"), "ms"),
            fmt(r.get("eval_gpu_p50_ms"), "ms"),
            fmt(r.get("val_loss"), "f3"),
            fmt(r.get("bipia_test_f1"), "f3"),
            fmt(r.get("bipia_test_fpr"), "pct"),
            fmt(r.get("deepset_test_f1"), "f3"),
            fmt(r.get("deepset_test_fpr"), "pct"),
            fmt(r.get("xxz224_test_f1"), "f3"),
            fmt(r.get("xxz224_test_fpr"), "pct"),
            fmt(r.get("notinject_fpr"), "pct"),
        ]) + " |")
    return "\n".join(lines)


def insights(rows: list[dict]) -> list[str]:
    """Derive the Pareto + over-defense takeaways so the operator doesn't
    have to eyeball the table."""
    out = []
    usable = [r for r in rows if r.get("bipia_test_f1") is not None]
    if not usable:
        return ["(no rows with eval data — skipping insights)"]

    # Pareto frontier uses CPU latency — A6's primary deployment target.
    # Fall back to GPU latency if CPU wasn't measured.
    def lat(r):
        return r.get("eval_cpu_p50_ms") or r.get("eval_gpu_p50_ms")
    lat_label = "CPU"
    if all(r.get("eval_cpu_p50_ms") is None for r in usable):
        lat_label = "GPU"

    best_f1 = max(usable, key=lambda r: r["bipia_test_f1"])
    out.append(f"Best BIPIA F1: **{best_f1['model']}** "
               f"({best_f1['bipia_test_f1']:.3f}, {best_f1['params_M']}M, "
               f"{(lat(best_f1) or 0):.2f} ms {lat_label}).")

    # Pareto frontier: for each model, is there another model with BOTH smaller latency AND higher F1?
    frontier = []
    for r in usable:
        r_lat = lat(r)
        if r_lat is None:
            continue
        dominated = False
        for r2 in usable:
            if r is r2:
                continue
            r2_lat = lat(r2)
            if (r2_lat is not None
                    and r2_lat <= r_lat
                    and r2["bipia_test_f1"] >= r["bipia_test_f1"]
                    and (r2_lat < r_lat or r2["bipia_test_f1"] > r["bipia_test_f1"])):
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: lat(r) or 0)
    out.append(f"Latency–F1 Pareto frontier (BIPIA test, {lat_label} latency): "
               + ", ".join(f"{r['model']} ({r['bipia_test_f1']:.3f}"
                           f" @ {lat(r):.2f}ms)" for r in frontier))

    # Over-defense: highest NotInject FPR is the most trigger-happy model.
    with_notinject = [r for r in usable if r.get("notinject_fpr") is not None]
    if with_notinject:
        worst_od = max(with_notinject, key=lambda r: r["notinject_fpr"])
        best_od = min(with_notinject, key=lambda r: r["notinject_fpr"])
        out.append(f"Over-defense (NotInject FPR): best **{best_od['model']}** "
                   f"{best_od['notinject_fpr']*100:.1f}%, "
                   f"worst **{worst_od['model']}** {worst_od['notinject_fpr']*100:.1f}%.")

    # Generalization gap: BIPIA (closest to training) F1 vs deepset (OOD-ish) F1.
    gaps = []
    for r in usable:
        bf = r.get("bipia_test_f1")
        df = r.get("deepset_test_f1")
        if bf is not None and df is not None:
            gaps.append((r["model"], bf - df))
    if gaps:
        gaps.sort(key=lambda t: abs(t[1]))
        out.append("Generalization gap (BIPIA F1 − deepset F1), smallest = best: "
                   + ", ".join(f"{m} {g:+.3f}" for m, g in gaps))

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing <short_name>.json files from eval_trained.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the summary.json")
    args = parser.parse_args()

    latency_map = load_latency_map()

    per_model_jsons = sorted(
        p for p in args.results_dir.glob("*.json") if p.name != "summary.json"
    )
    rows = []
    for p in per_model_jsons:
        name = p.stem
        try:
            data = json.load(open(p))
        except Exception as exc:
            print(f"[warn] could not load {p}: {exc}")
            data = None
        row = format_row(name, latency_map.get(name, {}), data)
        rows.append(row)

    # Order rows by params ascending (null last).
    rows.sort(key=lambda r: (r.get("params_M") is None, r.get("params_M") or 1e9))

    table = markdown_table(rows)
    tldr = insights(rows)

    args.output.write_text(json.dumps({"rows": rows, "insights": tldr}, indent=2))
    md_path = args.output.with_suffix(".md")
    md_path.write_text(
        "# lightweight sweep summary\n\n"
        + "\n".join(f"- {line}" for line in tldr)
        + "\n\n"
        + table + "\n"
    )

    print(table)
    print()
    for line in tldr:
        print("- " + line)
    print()
    print(f"Wrote {args.output} + {md_path}")


if __name__ == "__main__":
    main()
