# Prompt Injection Detection Benchmark Results

**Updated:** 2026-04-01 21:26  
**Hardware:** CPU (Apple Silicon M3)

### deepset-all (662 samples)

| model                          | params   |     f1 |   precision |   recall |    fpr |   latency_p50_ms |
|:-------------------------------|:---------|-------:|------------:|---------:|-------:|-----------------:|
| fmops-distilbert               | 66M      | 0.9786 |      1      |   0.9582 | 0      |            13    |
| rule-based                     | 0        | 0.7044 |      1      |   0.5437 | 0      |             0.05 |
| protectai-base-v1              | 86M      | 0.615  |      0.9597 |   0.4525 | 0.0125 |            50.05 |
| hybrid                         | ~86M     | 0.6011 |      1      |   0.4297 | 0      |             0.05 |
| protectai-base-v2              | 86M      | 0.5798 |      0.9646 |   0.4144 | 0.01   |            48.95 |
| vsr-jailbreak-mmbert32k-merged | 307M     | 0.2085 |      0.7273 |   0.1217 | 0.0301 |            38.07 |
| vsr-jailbreak-modernbert       | 149M     | 0.1413 |      1      |   0.076  | 0      |            39.67 |
| embedding-knn                  | ~23M     | 0.0803 |      1      |   0.0418 | 0      |             5.91 |

### xTRam1 (10296 samples)

| model                          | params   |     f1 |   precision |   recall |    fpr |   latency_p50_ms |
|:-------------------------------|:---------|-------:|------------:|---------:|-------:|-----------------:|
| protectai-base-v2              | 86M      | 0.9082 |      0.9951 |   0.8353 | 0.0018 |            69.84 |
| vsr-jailbreak-mmbert32k-merged | 307M     | 0.6646 |      0.9318 |   0.5165 | 0.0166 |            47.88 |
| hybrid                         | ~86M     | 0.6236 |      0.9856 |   0.4561 | 0.0029 |             0.13 |
| vsr-jailbreak-modernbert       | 149M     | 0.6187 |      0.9847 |   0.451  | 0.0031 |            44.6  |
| protectai-base-v1              | 86M      | 0.5677 |      0.9599 |   0.4031 | 0.0074 |            63.19 |
| fmops-distilbert               | 66M      | 0.5235 |      0.3568 |   0.9828 | 0.7796 |            14.6  |
| rule-based                     | 0        | 0.4659 |      0.8675 |   0.3185 | 0.0214 |             0.1  |
| embedding-knn                  | ~23M     | 0.2186 |      1      |   0.1227 | 0      |             6.49 |

### jailbreakbench (200 samples)

| model                          | params   |     f1 |   precision |   recall |   fpr |   latency_p50_ms |
|:-------------------------------|:---------|-------:|------------:|---------:|------:|-----------------:|
| fmops-distilbert               | 66M      | 0.669  |      0.5163 |     0.95 |  0.89 |            15.89 |
| vsr-jailbreak-mmbert32k-merged | 307M     | 0.6    |      0.8    |     0.48 |  0.12 |            53.87 |
| rule-based                     | 0        | 0.0392 |      1      |     0.02 |  0    |             0.1  |
| hybrid                         | ~86M     | 0.0198 |      1      |     0.01 |  0    |             0.06 |
| protectai-base-v1              | 86M      | 0      |      0      |     0    |  0    |            67.59 |
| protectai-base-v2              | 86M      | 0      |      0      |     0    |  0.01 |            67.29 |
| vsr-jailbreak-modernbert       | 149M     | 0      |      0      |     0    |  0    |            56.65 |
| embedding-knn                  | ~23M     | 0      |      0      |     0    |  0    |             5.83 |

