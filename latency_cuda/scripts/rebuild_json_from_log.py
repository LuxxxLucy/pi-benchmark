"""Reconstruct partial results/latency_{cuda,cpu}.json from a run stdout log.

Used when the bench run produced valid results on a remote host but the JSON
files were not preserved — stdout carries p50/p95/mean/throughput per cell,
which is enough for a partial JSON tagged `source: "log_reconstruct"`.

Usage:
    uv run python scripts/rebuild_json_from_log.py results/3090_run_YYYY-MM-DD.log

Reads both the CUDA and CPU halves (split on the `env: {` header whose
`device` field flips from `cuda` to `cpu`) and writes the partial
`results/latency_cuda.json` + `results/latency_cpu.json` next to the log.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import src.models as models  # noqa: E402 — cwd = latency_cuda/ when invoked via uv run


ENV_RE = re.compile(r"^env:\s*\{", re.MULTILINE)
SECTION_RE = re.compile(
    r"^===\s+(?P<name>\S+)\s+(?:\((?P<hf>[^)]+)\)\s+\[(?P<family>[^/]+?)\s*/\s*(?P<group>[^\]]+?)\]\s*)?===(?P<trailer>.*)$",
    re.MULTILINE,
)
LOAD_RE = re.compile(r"loaded\s+(?P<load>[\d.]+)s\s+—\s+(?P<params>[\d.]+)M params,\s+max_pos=(?P<maxpos>\d+),\s+dtype=torch\.(?P<dtype>\w+)")
CELL_RE = re.compile(
    r"len=\s*(?P<target>\d+)(?:\s+\(actual\s+(?P<actual>\d+)\))?:\s+p50=\s*(?P<p50>[\d.]+)ms\s+p95=\s*(?P<p95>[\d.]+)ms\s+mean=\s*(?P<mean>[\d.]+)ms\s+thr=\s*(?P<thr>[\d.]+)\s+rps"
)
FAIL_RE = re.compile(r"^===\s+(\S+)\s+.*===\s*\n(?:.*\n)?FAIL\s+", re.MULTILINE)


def parse_env_block(text: str, start: int) -> tuple[dict, int]:
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    env = json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    # Some logs compact-render lengths on one line; strip the comments none exist.
                    env = json.loads(text[i : j + 1])
                return env, j + 1
    raise ValueError("unbalanced env block")


def split_halves(text: str) -> list[tuple[dict, str]]:
    """Return [(env_dict, body_text), ...] for each env header in the log."""
    halves = []
    positions = [m.start() for m in ENV_RE.finditer(text)]
    positions.append(len(text))
    for i in range(len(positions) - 1):
        env, body_start = parse_env_block(text, positions[i])
        body = text[body_start : positions[i + 1]]
        halves.append((env, body))
    return halves


def parse_half(body: str) -> list[tuple[str, dict | None]]:
    """Return [(short_name, result_dict_or_None)] preserving source order.
    None means the section was a FAIL.
    """
    out: list[tuple[str, dict | None]] = []
    # Walk sections in order.
    for m in SECTION_RE.finditer(body):
        name = m.group("name")
        sect_start = m.end()
        # Next === or EOF bounds this section.
        next_m = SECTION_RE.search(body, sect_start)
        sect_end = next_m.start() if next_m else len(body)
        sect = body[sect_start:sect_end]

        trailer = (m.group("trailer") or "").strip()
        if ("FAIL " in trailer or "FAIL " in sect) and not LOAD_RE.search(sect):
            out.append((name, None))
            continue

        load_m = LOAD_RE.search(sect)
        cells = {}
        for c in CELL_RE.finditer(sect):
            tgt = int(c.group("target"))
            act = c.group("actual")
            cells[c.group("target")] = {
                "target_length": tgt,
                "actual_length": int(act) if act else tgt,
                "p50_ms": float(c.group("p50")),
                "p95_ms": float(c.group("p95")),
                "mean_ms": float(c.group("mean")),
                "throughput_rps": float(c.group("thr")),
            }
        if not cells:
            out.append((name, None))
            continue

        out.append((name, {
            "params_M": float(load_m.group("params")) if load_m else None,
            "max_position_embeddings": int(load_m.group("maxpos")) if load_m else None,
            "load_time_s": float(load_m.group("load")) if load_m else None,
            "dtype": load_m.group("dtype") if load_m else None,
            "lengths": cells,
        }))
    return out


def build_results(entries: list[tuple[str, dict | None]], device: str, dtype: str) -> list[dict]:
    results = []
    for name, data in entries:
        try:
            spec = models.by_name(name)
        except KeyError:
            continue
        if data is None:
            results.append({
                "name": name, "hf_id": spec.hf_id, "family": spec.family, "group": spec.group,
                "note": spec.note, "device": device, "dtype": dtype,
                "random_head": spec.group == "arch-baseline",
                "error": "tokenizer load failed (use_fast=False fallback also failed)",
            })
            continue
        results.append({
            "name": name, "hf_id": spec.hf_id, "family": spec.family, "group": spec.group,
            "note": spec.note,
            "params_M": data["params_M"],
            "max_position_embeddings": data["max_position_embeddings"],
            "device": device,
            "dtype": data["dtype"] or dtype,
            "load_time_s": data["load_time_s"],
            "batch_size": 1,
            "warmup_iters": 10,
            "measure_iters": 50,
            "random_head": spec.group == "arch-baseline",
            "lengths": data["lengths"],
        })
    return results


def main():
    log_path = Path(sys.argv[1]).resolve()
    text = log_path.read_text()
    halves = split_halves(text)
    if len(halves) < 2:
        print(f"expected 2 env blocks (cuda + cpu), got {len(halves)}", file=sys.stderr)
        sys.exit(1)

    out_dir = log_path.parent
    note = (f"Reconstructed from {log_path.name}. Only p50/p95/mean/throughput "
            "retained per (model, length); min/p99/stdev/max were not printed. "
            "Next clean run overwrites with full statistics.")

    for env, body in halves:
        device = env["device"]
        dtype = env["dtype"]
        env = {**env, "source": "log_reconstruct", "source_note": note}
        entries = parse_half(body)
        results = build_results(entries, device, dtype)
        payload = {"env": env, "results": results}
        out_path = out_dir / f"latency_{device}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        n_ok = sum(1 for r in results if "error" not in r)
        n_fail = len(results) - n_ok
        print(f"wrote {out_path.name} — {n_ok} ok, {n_fail} fail")


if __name__ == "__main__":
    main()
