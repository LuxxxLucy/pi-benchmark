# Prompt Injection Detection Benchmark Results

**Date:** 2026-04-01 20:37
**Hardware:** CPU (Apple Silicon M3)

## Results Table

| model                          | params   | dataset        |   samples |   accuracy |   precision |   recall |     f1 |    fpr |   latency_p50_ms |   latency_p95_ms |   latency_mean_ms |   errors |
|:-------------------------------|:---------|:---------------|----------:|-----------:|------------:|---------:|-------:|-------:|-----------------:|-----------------:|------------------:|---------:|
| fmops-distilbert               | 66M      | deepset-all    |       662 |     0.9834 |      1      |   0.9582 | 0.9786 | 0      |            13    |            18.23 |             13.7  |        0 |
| protectai-base-v1              | 86M      | deepset-all    |       662 |     0.7749 |      0.9597 |   0.4525 | 0.615  | 0.0125 |            50.05 |            72.78 |             53.37 |        0 |
| protectai-base-v2              | 86M      | deepset-all    |       662 |     0.7613 |      0.9646 |   0.4144 | 0.5798 | 0.01   |            48.95 |            69.73 |             51.8  |        0 |
| vsr-jailbreak-modernbert       | 149M     | deepset-all    |       662 |     0.6329 |      1      |   0.076  | 0.1413 | 0      |            39.67 |            62.3  |             43.06 |        0 |
| vsr-jailbreak-mmbert32k-merged | 307M     | deepset-all    |       662 |     0.6329 |      0.7273 |   0.1217 | 0.2085 | 0.0301 |            38.07 |            52.77 |             39.91 |        0 |
| fmops-distilbert               | 66M      | xTRam1         |     10296 |     0.4534 |      0.3568 |   0.9828 | 0.5235 | 0.7796 |            14.6  |            47.3  |             18.14 |        0 |
| protectai-base-v1              | 86M      | xTRam1         |     10296 |     0.8125 |      0.9599 |   0.4031 | 0.5677 | 0.0074 |            63.19 |           210.19 |             79.38 |        0 |
| protectai-base-v2              | 86M      | xTRam1         |     10296 |     0.9484 |      0.9951 |   0.8353 | 0.9082 | 0.0018 |            69.84 |           213.78 |             85.68 |        0 |
| vsr-jailbreak-modernbert       | 149M     | xTRam1         |     10296 |     0.8301 |      0.9847 |   0.451  | 0.6187 | 0.0031 |            44.6  |           151.57 |             56.24 |        0 |
| vsr-jailbreak-mmbert32k-merged | 307M     | xTRam1         |     10296 |     0.8407 |      0.9318 |   0.5165 | 0.6646 | 0.0166 |            47.88 |           151.72 |             59.24 |        0 |
| fmops-distilbert               | 66M      | jailbreakbench |       200 |     0.53   |      0.5163 |   0.95   | 0.669  | 0.89   |            15.89 |            18.21 |             16.17 |        0 |
| protectai-base-v1              | 86M      | jailbreakbench |       200 |     0.5    |      0      |   0      | 0      | 0      |            67.59 |            85.67 |             68.88 |        0 |
| protectai-base-v2              | 86M      | jailbreakbench |       200 |     0.495  |      0      |   0      | 0      | 0.01   |            67.29 |            79.26 |             68.2  |        0 |
| vsr-jailbreak-modernbert       | 149M     | jailbreakbench |       200 |     0.5    |      0      |   0      | 0      | 0      |            56.65 |            64.69 |             56.29 |        0 |
| vsr-jailbreak-mmbert32k-merged | 307M     | jailbreakbench |       200 |     0.68   |      0.8    |   0.48   | 0.6    | 0.12   |            53.87 |            69.51 |             54.8  |        0 |

## Key Findings

### Per-dataset winners

| Dataset | Samples | Best F1 | Best Model | Notes |
|---------|---------|---------|-----------|-------|
| **deepset-all** | 662 | **0.979** | fmops-distilbert | Dominant — 96% recall, 0% FPR |
| **xTRam1** | 10,296 | **0.908** | protectai-base-v2 | Reversal — protectai-v2 wins here (83.5% recall) |
| **jailbreakbench** | 200 | **0.669** | fmops-distilbert | All models struggle — fmops overfires (89% FPR) |

### Critical observations

1. **No model wins everywhere.** fmops-distilbert dominates deepset but has 78% FPR on xTRam1 (flags most safe text as injection). protectai-base-v2 is best on xTRam1 but weak on deepset.

2. **Dataset definition matters enormously.** deepset uses a broad "injection" definition (includes roleplay). xTRam1 uses a narrower definition. Models trained on one definition fail on the other.

3. **JailbreakBench exposes all models.** protectai-v1/v2 and vsr-modernbert detect ZERO jailbreaks (F1=0). These jailbreaks are goal-oriented harmful requests ("write defamatory article") not instruction-override patterns.

4. **vSR models improve dramatically on xTRam1** (F1=0.62-0.66) vs deepset (F1=0.14-0.21). Their narrow "jailbreak" definition aligns better with xTRam1's labeling.

5. **protectai-base-v2 is the most balanced model** — F1=0.908 on xTRam1 (10K samples), the largest and most realistic dataset.

### Latency on M3 CPU

| Model | p50 (ms) | p95 (ms) |
|-------|----------|----------|
| fmops-distilbert | 13-16 | 18-47 |
| protectai-base-v1/v2 | 49-70 | 70-214 |
| vsr-modernbert | 40-57 | 62-152 |
| vsr-mmbert32k | 38-54 | 53-152 |
| **rule-based (no ML)** | **0.055** | **0.1** |
