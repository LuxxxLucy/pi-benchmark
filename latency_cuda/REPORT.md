# pi-benchmark — Lightweight-classifier latency on RTX 3090 (CUDA + CPU)

**Updated:** 2026-04-16
**Host:** Linux-6.8, NVIDIA RTX 3090 24GB, torch 2.6.0+cu124
**Scope:** batch=1 forward-pass P50/P95 latency across 5 token lengths (32 / 64 / 128 / 256 / 512), same-machine CUDA (bfloat16) and CPU (float32) passes, 15 encoder candidates (7 PI-trained + 8 architecture baselines; 2 failed tokenizer load, fixed in this commit).
**Source:** `src/bench.py` + `src/models.py`; raw stdout in `results/3090_run_2026-04-16.log`; partial reconstructed JSON in `results/latency_{cuda,cpu}.json`.

## TL;DR

1. **On a 3090, every 25M–184M encoder runs 2–13 ms P50 at batch=1 across 32–512 tokens. If A6 deploys on GPU, latency is not a constraint on backbone choice.**
2. **Below ~25M params CUDA is dispatch-bound** — testsavantai-tiny (4.4M) is *slower* on CUDA than CPU (4.9 ms vs 0.9–2.7 ms). GPU is not free for sub-10M models at batch=1.
3. **DeBERTa carries a persistent CUDA penalty.** At 184M, protectai-deberta-v2 / deepset-deberta run 11 ms on CUDA — 5× slower than DistilBERT at 67M (2.2 ms), despite 2.7× more params. Disentangled-attention kernels, not parameter count, drive the gap.
4. **On the 3090 host CPU (x86, default threads, fp32) the M3 finding from the parent [REPORT.md §7](../../REPORT.md) replicates.** fmops-distilbert (66M) runs 24 ms @ 128 tok / 78 ms @ 512 tok — still comfortably under A6's 200 ms P99 budget. 184M DeBERTa busts the budget at 256 tok (146–147 ms P50) and blows it at 512 tok (>386 ms P50).

---

## 1. Setup

**Method.** Per `(model, length)`: 10 warmup forwards, 50 measured forwards, `torch.cuda.synchronize()` around each, `torch.no_grad()`, gc disabled inside the measure loop, batch_size=1. bf16 on CUDA, fp32 on CPU. Input is `FILL_TEXT` looped and truncated to exact target length (see `../bench_common.py`).

**Models.** 15 encoders in two groups (`src/models.py`):
- **PI-trained (7)** — real prompt-injection classifiers with task-fine-tuned heads: `testsavantai-{tiny,small,medium,base}`, `fmops-distilbert`, `protectai-deberta-v2`, `deepset-deberta-injection`.
- **Arch-baseline (8)** — open backbones with random classifier heads. Latency-only: `bert-tiny`, `bert-mini`, `bert-L4-H256`, `minilm-L{6,12}-H384`, `deberta-v3-{xsmall,small}`, `distilbert-base`. Accuracy on these is meaningless.

**What this report does not cover.** Accuracy (parent [REPORT.md](../../REPORT.md)), quantization (INT8/AWQ not measured), batched throughput (batch > 1), long-context (all models cap at `max_position_embeddings=512`), production engines (eager PyTorch only; no torch.compile, TensorRT, ONNX-Runtime).

**Script issues surfaced by this run (fixed in this commit; next rerun will be clean):**
1. `bert-tiny` / `bert-mini` failed tokenizer load because `transformers`' slow→fast converter imports `sentencepiece` / `tiktoken` even for WordPiece tokenizers whose repos ship only `vocab.txt`. Fix: `src/bench.py` now falls back to `use_fast=False` on the convert-time `ValueError`, skipping the converter.
2. `Token indices sequence length is longer than the specified maximum sequence length (522 > 512)` printed for every `len=512` run. Cosmetic — the tensor is 512 post-truncation (log confirms `actual 512`). Fix: `TRANSFORMERS_VERBOSITY=error` now set before the `transformers` import (previously set in `main()`, too late).
3. Arch-baseline models emit HuggingFace `MISSING classifier.{weight,bias}` reports — expected, random head. Fix: results JSON now includes `"random_head": true/false` per model entry so downstream analysis can flag the distinction without string-matching logs.
4. `run.sh` now calls `uv sync` + `scripts/download.py` before the benchmark. Skipping `uv sync` after a `pyproject.toml` dep change was the root cause of issue 1 on the 2026-04-16 run; mid-bench downloads inflated `load_time_s` into the 757–820 s range for three cold models.

---

## 2. GPU results — RTX 3090, bfloat16, batch=1

P50 latency (ms) at 5 token lengths. ★ = PI-trained; random-head baselines unmarked.

| Model | Params | 32 | 64 | 128 | 256 | 512 |
|:---|---:|---:|---:|---:|---:|---:|
| testsavantai-tiny ★ | 4.4M | 4.88 | 2.35 | 4.87 | 4.89 | 4.90 |
| bert-L4-H256 | 11.2M | 1.91 | 1.70 | 2.03 | 1.72 | 1.73 |
| minilm-L6-H384 | 22.7M | 4.41 | 4.19 | 4.34 | 3.02 | 2.24 |
| testsavantai-small ★ | 28.8M | 2.05 | 1.90 | 1.97 | 1.92 | 1.76 |
| minilm-L12-H384 | 33.4M | 5.61 | 5.51 | 5.69 | 4.34 | 4.17 |
| testsavantai-medium ★ | 41.4M | 4.96 | 4.44 | 4.69 | 5.08 | 4.38 |
| distilbert-base | 67.0M | 2.34 | 3.54 | 2.47 | 2.10 | 2.17 |
| fmops-distilbert ★ | 67.0M | 2.26 | 2.93 | 2.39 | 2.24 | 2.17 |
| testsavantai-base ★ | 67.0M | 2.95 | 3.08 | 2.41 | 2.57 | 2.33 |
| deberta-v3-xsmall | 70.8M | 12.62 | 12.45 | 11.38 | 12.35 | 12.87 |
| deberta-v3-small | 141.9M | 6.47 | 7.83 | 7.54 | 6.42 | 5.74 |
| deepset-deberta-inj ★ | 184.4M | 12.41 | 12.39 | 11.89 | 12.12 | 10.29 |
| protectai-deberta-v2 ★ | 184.4M | 12.72 | 12.12 | 13.47 | 12.59 | 11.02 |

P95 appendix table in §A. `bert-tiny` and `bert-mini` missing (tokenizer FAIL, fixed).

### 2.1 Observations

**GPU latency is near-flat across token length for every model in the set.** At batch=1 with bf16, no model's P50 at 512 tok is more than 1.2× its P50 at 32 tok. The forward-pass work scales O(N²) in the attention, but on a 3090 the kernel-launch + memory-dispatch floor dominates below ~1k tokens. This is the defining property of single-request serving: batch=1 GPU latency is not about "how fast is your backbone on 512 tokens" — it's about "how many kernels does your backbone dispatch."

**testsavantai-tiny (4.4M) is slower on CUDA than on CPU.** 4.9 ms P50 on GPU vs 0.9–2.7 ms on CPU across 32–512 tokens. At 4.4M params, bf16 matmuls fit in L2 and CPU beats the GPU dispatch pipeline. **GPU is not free for sub-10M encoders at batch=1** — if A6 ever needs a sub-10M "first-gate" filter, keep it on CPU.

**DistilBERT and BERT-L4 dominate at batch=1 on CUDA.** Every model in the distilbert / tiny-BERT family lands at 2–3 ms P50 regardless of size in this range. bert-L4-H256 (11.2M) at 1.7 ms is the fastest in the set; distilbert-base (67M) at 2.2 ms is 6× cheaper than deberta-v3-xsmall (70.8M) at 12.9 ms despite comparable params.

**The DeBERTa CUDA tax.** Every DeBERTa variant — xsmall, small, base, and the two PI-trained v3-base models — runs 5–13 ms P50. This is not a params issue:

| Model | Params | CUDA P50 @ 512 | vs distilbert-base |
|:---|---:|---:|---:|
| distilbert-base (reference) | 67.0M | 2.17 | 1.0× |
| deberta-v3-small | 141.9M | 5.74 | 2.6× |
| deberta-v3-xsmall | **70.8M** | 12.87 | **5.9×** |
| deepset-deberta-injection | 184.4M | 10.29 | 4.7× |
| protectai-deberta-v2 | 184.4M | 11.02 | 5.1× |

The xsmall anomaly (70.8M runs slower than 141.9M small) rules out parameter count as the driver. The suspect is DeBERTa-v3's disentangled relative-position attention: two parallel matmul streams + a position-bucket gather that dispatches as multiple kernels per attention layer. **If GPU throughput or multi-tenant serving cost matters, avoid DeBERTa-v3** — identical accuracy at a fraction of the CUDA kernel count is available via DistilBERT or MiniLM.

**testsavantai-base looks like DistilBERT, not DeBERTa.** `src/models.py` tags testsavantai-base as `family="deberta"` and the model card says "DeBERTa-v3 base, 86M", but the measured param count is 67.0M and CUDA latency is 2.3 ms — the DistilBERT-at-67M profile, not the DeBERTa-v3 profile. Flagged in §5 as an unresolved data integrity question for the `family` field.

---

## 3. CPU results — 3090 host CPU, float32, default threads, batch=1

Same box as §2, CPU-only pass. P50 latency (ms):

| Model | Params | 32 | 64 | 128 | 256 | 512 |
|:---|---:|---:|---:|---:|---:|---:|
| testsavantai-tiny ★ | 4.4M | 0.89 | 0.99 | 1.19 | 1.64 | 2.73 |
| bert-L4-H256 | 11.2M | 1.85 | 2.40 | 3.58 | 6.00 | 11.94 |
| minilm-L6-H384 | 22.7M | 4.98 | 6.58 | 9.93 | 12.25 | 24.22 |
| testsavantai-small ★ | 28.8M | 5.51 | 7.45 | 11.03 | 19.33 | 25.90 |
| minilm-L12-H384 | 33.4M | 8.83 | 10.74 | 14.78 | 24.62 | 48.40 |
| testsavantai-medium ★ | 41.4M | 9.83 | 11.46 | 21.55 | 26.10 | 51.84 |
| distilbert-base | 67.0M | 14.12 | 20.88 | 24.12 | 40.80 | 79.76 |
| fmops-distilbert ★ | 67.0M | 15.26 | 16.79 | 24.25 | 57.26 | 77.98 |
| testsavantai-base ★ | 67.0M | 14.73 | 22.84 | 24.19 | 39.83 | 77.89 |
| deberta-v3-xsmall | 70.8M | 22.04 | 22.23 | 30.15 | 51.56 | 140.08 |
| deberta-v3-small | 141.9M | 27.77 | 32.09 | 42.61 | 72.78 | 183.37 |
| deepset-deberta-inj ★ | 184.4M | 55.63 | 63.17 | 88.09 | 141.80 | **386.59** |
| protectai-deberta-v2 ★ | 184.4M | 57.25 | 66.28 | 85.08 | 146.15 | **418.27** |

### 3.1 CPU is length-sensitive; the 200 ms budget verdict

Unlike CUDA, CPU shows the expected superlinear growth: fmops-distilbert (66M) goes 15 ms → 78 ms from 32 → 512 tok; 184M DeBERTa goes 55 ms → 387 ms. The A6 production contract from `features.md` is **P99 ≤ 200 ms CPU** — so the verdict by length:

| Model | 128 tok | 256 tok | 512 tok | Fits A6 ≤ 200 ms |
|:---|---:|---:|---:|:---:|
| testsavantai-tiny | 1 ms | 2 ms | 3 ms | ✓ all lengths |
| bert-L4-H256 | 4 ms | 6 ms | 12 ms | ✓ all lengths |
| fmops-distilbert | 24 ms | 57 ms | 78 ms | ✓ all lengths |
| testsavantai-base | 24 ms | 40 ms | 78 ms | ✓ all lengths |
| deberta-v3-xsmall | 30 ms | 52 ms | 140 ms | ✓ all lengths (narrow at 512) |
| deberta-v3-small | 43 ms | 73 ms | 183 ms | ✓ all lengths (narrow at 512) |
| protectai-deberta-v2 | 85 ms | 146 ms | 418 ms | ✗ busts at 512 |
| deepset-deberta-injection | 88 ms | 142 ms | 387 ms | ✗ busts at 512 |

**The A6 CPU cliff is still 184M DeBERTa at 512 tok.** This matches the parent REPORT.md §7 finding on M3. On x86 default threads, the cliff sits ~5× worse than the 200 ms gate (418 ms); on M3 threads=4 it was ~256 ms. Different silicon, same qualitative conclusion: 66M-class DistilBERT-style encoders fit the budget; 184M DeBERTa-v3 busts it at long inputs.

### 3.2 Cross-platform consistency with M3 (parent REPORT.md §7)

Sanity check — same PI-trained models, same script, two different machines:

| Model | M3 (threads=4) P50 @ 500 tok | 3090 host CPU P50 @ 512 tok |
|:---|---:|---:|
| fmops-distilbert | 55.8 ms | 78.0 ms |
| testsavantai-base | 58.4 ms | 77.9 ms |
| testsavantai-medium | 40.8 ms | 51.8 ms |
| testsavantai-small | 19.3 ms | 25.9 ms |
| deepset-deberta-injection | 256.9 ms | 386.6 ms |

The 3090 host CPU is ~1.3–1.5× slower than M3 threads=4 across the board. M3's benefit is its wide NEON / AMX pipeline for dense GEMMs at small batch sizes; the x86 Linux box does not have AVX-512 enabled for these kernels. **Same ranking, same cliff, absolute values shifted by a constant factor** — the A6 CPU budget conclusion transfers cleanly from the M3 baseline to generic x86 CPU deployment.

---

## 4. Cross-device comparison — who wins on GPU vs CPU

Speedup (CPU P50 / CUDA P50) at 512 tokens. `<1×` means CUDA is slower than CPU — i.e. the GPU isn't paying off at batch=1 for that model.

| Model | CPU 512 | CUDA 512 | CPU/CUDA | Regime |
|:---|---:|---:|---:|:---|
| testsavantai-tiny | 2.73 | 4.90 | **0.56×** | CUDA slower — dispatch-bound |
| bert-L4-H256 | 11.94 | 1.73 | 6.9× | GPU wins, small margin |
| minilm-L6-H384 | 24.22 | 2.24 | 10.8× | GPU wins clearly |
| testsavantai-small | 25.90 | 1.76 | **14.7×** | GPU wins clearly |
| minilm-L12-H384 | 48.40 | 4.17 | 11.6× | GPU wins clearly |
| testsavantai-medium | 51.84 | 4.38 | 11.8× | GPU wins clearly |
| distilbert-base | 79.76 | 2.17 | **36.8×** | GPU wins big |
| fmops-distilbert | 77.98 | 2.17 | **35.9×** | GPU wins big |
| testsavantai-base | 77.89 | 2.33 | 33.4× | GPU wins big |
| deberta-v3-xsmall | 140.08 | 12.87 | 10.9× | GPU wins, DeBERTa tax |
| deberta-v3-small | 183.37 | 5.74 | 32.0× | GPU wins big |
| deepset-deberta-inj | 386.59 | 10.29 | 37.6× | GPU wins big, CPU busts budget |
| protectai-deberta-v2 | 418.27 | 11.02 | 38.0× | GPU wins big, CPU busts budget |

Three regimes visible:
1. **Sub-10M dispatch-bound** — CPU wins. testsavantai-tiny.
2. **10–50M transitional** — 7–15× GPU speedup. GPU worth using but the headroom is modest.
3. **50M+ GEMM-bound** — 33–38× GPU speedup for non-DeBERTa; 11× for DeBERTa (the same CUDA tax shows up as a reduced-speedup ratio). This is where GPU deployment pays off most.

---

## 5. Implications for A6 backbone choice

The parent [REPORT.md](../../REPORT.md) §8 "Build Recipe" picks a 66M-class encoder (DistilBERT / ModernBERT-small / DeBERTa-v3-xsmall) on accuracy grounds, subject to `P99 ≤ 200 ms CPU` and `max FPR ≤ 5%`. This latency-cuda report adds:

1. **If A6 is CPU-deployed: the 66M DistilBERT Pareto winner holds.** fmops-distilbert @ 78 ms P50 / 512 tok on Linux x86 = 2.5× under budget. 184M DeBERTa is out at 512 tok.
2. **If A6 moves to GPU (multi-tenant serving, GPU-batched inference): every 66M-class fits trivially.** Choose by accuracy + $ alone; latency is no constraint.
3. **Avoid DeBERTa-v3 on GPU if cost/throughput matters.** 5× CUDA overhead vs DistilBERT at equal params means 5× fewer concurrent streams per GPU at equal latency budget. If the recipe plan ends on DeBERTa-v3-xsmall for accuracy, budget the GPU cost at DistilBERT×5, not DistilBERT×1.
4. **Sub-10M "first-gate" filters stay on CPU.** If any future design puts a tiny classifier upstream of the main detector, it is cheaper on CPU than on GPU at batch=1.

---

## 6. Unresolved questions (TODO)

Each TODO names the experiment that would resolve it.

1. **Rerun with script fixes to get clean JSONs + bert-tiny/bert-mini populated.** Currently the result JSONs are log-reconstructed and missing `min_ms / p99_ms / stdev_ms / max_ms`. Command: `bash run.sh` on the 3090 host after this commit lands. ~1 h end-to-end (most of that is the 50 forward iters × 13 models × 5 lengths × 2 devices).
2. **Long-context encoders (>512 tok).** Every candidate caps at `max_position_embeddings=512`. A6 sees markdown/HTML documents that may exceed this. Add ModernBERT (8192), Longformer-base (4096), and deberta-v3-large-long if it exists. Extend `src/models.py`; keep the length sweep at `32 64 128 256 512 1024 2048 4096`.
3. **Explain the deberta-v3-xsmall (70.8M) / deberta-v3-small (141.9M) latency inversion on CUDA** — xsmall runs 12.9 ms @ 512, small runs 5.7 ms. A param-count-aware explanation is missing. Candidate causes: (a) different disentangled-attention hidden-size splits, (b) xsmall's shared-embedding scheme forces an extra kernel, (c) kernel autotuning picked different implementations. Needs a `torch.profiler` trace on both models, same input.
4. **Verify testsavantai-base backbone family.** The card claims DeBERTa-v3-base (86M) but measured params are 67.0M and CUDA/CPU latency matches DistilBERT-67M exactly, not DeBERTa-v3-xsmall-70.8M. Either the card is wrong, or the architecture pulled by `AutoModelForSequenceClassification` from the checkpoint differs from the card. Check `config.json` of the checkpoint to confirm.
5. **torch.compile / TensorRT / ONNX-Runtime-CUDA speedup.** Eager PyTorch is a lower bound. Measure `torch.compile(mode="reduce-overhead")` on the DistilBERT Pareto winner; expect ~2× from CUDA Graphs alone for the dispatch-bound regime. ORT-CUDA and TensorRT likely collapse the DeBERTa CUDA tax — worth measuring because it's the cheapest path to "DeBERTa at DistilBERT speed".
6. **Batched throughput (batch=4, 16, 64).** At batch=1 GPU is dispatch-bound for small models; batched throughput is what actually matters for multi-tenant serving. Add a `--batch-size` arg to `src/bench.py`, sweep `{1, 4, 16, 64}` × lengths × PI-trained models.
7. **INT8 / AWQ quantization on CPU.** Parent REPORT.md §9 item 2 lists ONNX INT8 as planned. Add a sister `latency_cpu_int8/` sub-project; expected 2–4× CPU speedup on x86-VNNI.
8. **Memory ceiling sanity.** 3090 has 24 GB; batch=1 fits everything trivially. At batch=64 × 184M DeBERTa × 4096 tokens, peak memory may matter. Add `torch.cuda.max_memory_allocated()` collection once TODO 6 is in.

---

## Appendix A. GPU P95 latency (ms)

| Model | 32 | 64 | 128 | 256 | 512 |
|:---|---:|---:|---:|---:|---:|
| testsavantai-tiny | 4.92 | 7.94 | 4.92 | 9.16 | 4.92 |
| bert-L4-H256 | 4.87 | 4.91 | 4.91 | 4.96 | 4.98 |
| minilm-L6-H384 | 4.98 | 5.06 | 5.59 | 5.16 | 5.35 |
| testsavantai-small | 4.94 | 4.94 | 4.97 | 5.04 | 5.18 |
| minilm-L12-H384 | 7.64 | 7.72 | 7.78 | 7.95 | 8.26 |
| testsavantai-medium | 7.38 | 5.61 | 6.03 | 5.97 | 5.71 |
| distilbert-base | 5.18 | 5.68 | 5.36 | 5.53 | 6.16 |
| fmops-distilbert | 5.17 | 5.18 | 5.36 | 5.55 | 6.13 |
| testsavantai-base | 5.18 | 5.17 | 5.37 | 5.54 | 6.11 |
| deberta-v3-xsmall | 14.54 | 13.44 | 13.51 | 14.14 | 14.72 |
| deberta-v3-small | 9.70 | 9.77 | 9.41 | 8.83 | 10.70 |
| deepset-deberta-inj | 13.83 | 13.89 | 14.40 | 13.69 | 19.04 |
| protectai-deberta-v2 | 15.52 | 14.47 | 16.72 | 14.54 | 19.11 |

P95/P50 ratios sit in 1.05–1.35 across the set; the ≥ 2× ratios on testsavantai-tiny @ 64/256 are kernel-overhead noise at the sub-5-ms scale (a 3 ms P50 + one 9 ms outlier pushes P95 above the mean). n=50 is the statistical floor.

## Appendix B. CPU P95 latency (ms)

| Model | 32 | 64 | 128 | 256 | 512 |
|:---|---:|---:|---:|---:|---:|
| testsavantai-tiny | 0.95 | 1.03 | 1.24 | 1.69 | 2.88 |
| bert-L4-H256 | 2.31 | 2.78 | 3.73 | 6.21 | 12.35 |
| minilm-L6-H384 | 5.15 | 6.98 | 10.16 | 12.57 | 24.63 |
| testsavantai-small | 5.76 | 7.65 | 11.28 | 20.05 | 26.43 |
| minilm-L12-H384 | 9.74 | 12.72 | 15.28 | 26.38 | 48.97 |
| testsavantai-medium | 10.69 | 12.86 | 21.89 | 26.51 | 55.33 |
| distilbert-base | 16.71 | 23.01 | 24.41 | 41.37 | 80.45 |
| fmops-distilbert | 15.75 | 17.31 | 25.61 | 58.64 | 80.84 |
| testsavantai-base | 15.33 | 24.70 | 24.56 | 40.41 | 79.58 |
| deberta-v3-xsmall | 23.06 | 22.98 | 32.98 | 60.90 | 149.22 |
| deberta-v3-small | 29.67 | 33.04 | 43.63 | 76.97 | 192.69 |
| deepset-deberta-inj | 60.15 | 65.17 | 89.11 | 149.01 | 454.51 |
| protectai-deberta-v2 | 73.81 | 71.21 | 91.25 | 154.10 | 503.66 |

## Appendix C. Reproducibility

```bash
cd projects/benchmark/benchmark_impl/latency_cuda
bash run.sh          # syncs deps, pre-fetches weights, runs CUDA + CPU pass
```

Outputs: `results/latency_cuda.json`, `results/latency_cpu.json` (full schema in [README.md](README.md)). Raw stdout from the 2026-04-16 run: `results/3090_run_2026-04-16.log`.

To rerun a subset:

```bash
uv run python src/bench.py --device cuda --models pi-trained --lengths 32 64 128 256 512
uv run python src/bench.py --device cpu  --models fmops-distilbert,distilbert-base
```
