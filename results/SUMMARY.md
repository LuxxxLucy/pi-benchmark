# Prompt Injection Detection Benchmark Results

**Date:** 2026-04-01 19:29
**Hardware:** CPU (Apple Silicon M3)

## Results Table

| model                          | params   | dataset                   |   samples |   accuracy |   precision |   recall |     f1 |    fpr |   latency_p50_ms |   latency_p95_ms |   latency_mean_ms |   errors |
|:-------------------------------|:---------|:--------------------------|----------:|-----------:|------------:|---------:|-------:|-------:|-----------------:|-----------------:|------------------:|---------:|
| fmops-distilbert               | 66M      | deepset/prompt-injections |       116 |     0.9052 |         1   |   0.8167 | 0.8991 | 0      |            13.32 |            20.17 |             14.07 |        0 |
| protectai-base-v1              | 86M      | deepset/prompt-injections |       116 |     0.6983 |         1   |   0.4167 | 0.5882 | 0      |            52.14 |            77.37 |             55.38 |        0 |
| protectai-base-v2              | 86M      | deepset/prompt-injections |       116 |     0.6724 |         1   |   0.3667 | 0.5366 | 0      |            55.32 |            78.07 |             57.98 |        0 |
| vsr-jailbreak-modernbert       | 149M     | deepset/prompt-injections |       116 |     0.5172 |         1   |   0.0667 | 0.125  | 0      |            38.77 |            60.97 |             42.86 |        0 |
| vsr-jailbreak-mmbert32k-merged | 307M     | deepset/prompt-injections |       116 |     0.5086 |         0.8 |   0.0667 | 0.1231 | 0.0179 |            43.18 |            60    |             44.63 |        0 |

## Key Findings

- **Best F1:** fmops-distilbert (0.8991)
- **Fastest:** fmops-distilbert (p50: 13.3ms)
- **Best accuracy:** fmops-distilbert (0.9052)
- **Lowest FPR:** All models except mmbert32k show 0.0000 FPR (very conservative)

## Important Context

**Low recall across all models is expected.** The deepset/prompt-injections dataset uses a broad definition of "injection" that includes roleplay prompts ("act as an interviewer") and indirect attacks. Most classifiers are trained on a narrower definition (explicit instruction override patterns like "ignore previous instructions").

**vSR models (6-12% recall):** These are jailbreak detectors, not general injection detectors. They only flag explicit jailbreak patterns. The deepset dataset contains many injection types that don't match jailbreak patterns.

**ProtectAI (37-42% recall):** Trained on a more specific injection dataset. Perfect precision (no false positives) but misses many deepset-labeled injections.

**Fmops DistilBERT (82% recall):** Trained on a broader definition of injection, best match for deepset's labeling scheme.

**Next steps:** Test on additional datasets (xTRam1, GenTel-Bench) where label definitions may better match model training. Also need to get access to gated models (Prompt Guard 2).
