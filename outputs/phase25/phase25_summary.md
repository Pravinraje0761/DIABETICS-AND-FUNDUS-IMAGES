# Phase 25 - dataset shift vs. site generalisation

_Generated 2026-11-06T02:04:49+00:00; B=2000, seed=42; predictions from `outputs/phase6` (frozen checkpoints)._

## Grade mix (%)

| Dataset | 0 | 1 | 2 | 3 | 4 | referable |
|---|--:|--:|--:|--:|--:|--:|
| APTOS-internal | 49.2 | 10.1 | 27.3 | 5.3 | 8.0 | 40.7 |
| Messidor-2 | 58.3 | 15.5 | 19.9 | 4.3 | 2.0 | 26.2 |
| IDRiD | 28.4 | 4.8 | 34.3 | 18.5 | 14.1 | 66.8 |

## A/C/D. QWK drop (internal − external) under each correction, 95% CI

Reweighted = external images importance-weighted to the APTOS-test grade mix (removes prevalence shift). 4-level = grades 0 and 1 merged (removes the mild-boundary reference-standard disagreement).

| Model | External | raw | reweighted | 4-level | 4-level + reweighted |
|---|---|---|---|---|---|
| EfficientNet-B3 | Messidor-2 | 0.377 [0.327, 0.431] | 0.293 [0.237, 0.351] | 0.200 [0.126, 0.276] | 0.141 [0.052, 0.234] |
| EfficientNet-B3 | IDRiD | 0.116 [0.070, 0.169] | 0.115 [0.064, 0.175] | 0.062 [-0.011, 0.135] | 0.006 [-0.068, 0.085] |
| ResNet-50 | Messidor-2 | 0.440 [0.385, 0.496] | 0.379 [0.309, 0.454] | 0.248 [0.169, 0.325] | 0.211 [0.115, 0.310] |
| ResNet-50 | IDRiD | 0.167 [0.116, 0.220] | 0.144 [0.097, 0.190] | 0.128 [0.054, 0.205] | 0.060 [-0.010, 0.128] |
| RETFound linear probe | Messidor-2 | 0.453 [0.392, 0.515] | 0.296 [0.251, 0.348] | 0.306 [0.228, 0.386] | 0.123 [0.057, 0.192] |
| RETFound linear probe | IDRiD | 0.221 [0.163, 0.279] | 0.311 [0.244, 0.378] | 0.098 [0.026, 0.169] | 0.153 [0.074, 0.229] |
| RETFound full fine-tune | Messidor-2 | 0.395 [0.345, 0.453] | 0.284 [0.239, 0.333] | 0.202 [0.134, 0.278] | 0.117 [0.044, 0.194] |
| RETFound full fine-tune | IDRiD | 0.139 [0.091, 0.189] | 0.161 [0.103, 0.223] | 0.097 [0.026, 0.166] | 0.084 [0.011, 0.157] |

### Share of the raw QWK drop removed by each correction

| Model | External | by reweighting | by 4-level | by both | residual drop |
|---|---|--:|--:|--:|--:|
| EfficientNet-B3 | Messidor-2 | 22% | 47% | 63% | 0.141 |
| EfficientNet-B3 | IDRiD | 1% | 47% | 95% | 0.006 |
| ResNet-50 | Messidor-2 | 14% | 44% | 52% | 0.211 |
| ResNet-50 | IDRiD | 14% | 23% | 64% | 0.060 |
| RETFound linear probe | Messidor-2 | 35% | 32% | 73% | 0.123 |
| RETFound linear probe | IDRiD | -41% | 56% | 31% | 0.153 |
| RETFound full fine-tune | Messidor-2 | 28% | 49% | 70% | 0.117 |
| RETFound full fine-tune | IDRiD | -16% | 30% | 39% | 0.084 |

### Does the primary comparison survive the corrections? Difference in drop vs EfficientNet-B3

| Model | External | raw | reweighted | 4-level + reweighted |
|---|---|---|---|---|
| ResNet-50 | Messidor-2 | +0.063 [+0.018, +0.107] | +0.086 [+0.017, +0.155] | +0.070 [-0.024, +0.163] |
| ResNet-50 | IDRiD | +0.051 [+0.001, +0.100] | +0.029 [-0.029, +0.082] | +0.054 [-0.019, +0.125] |
| RETFound linear probe | Messidor-2 | +0.076 [+0.020, +0.131] | +0.004 [-0.060, +0.069] | -0.019 [-0.106, +0.069] |
| RETFound linear probe | IDRiD | +0.105 [+0.042, +0.167] | +0.196 [+0.117, +0.275] | +0.148 [+0.057, +0.238] |
| RETFound full fine-tune | Messidor-2 | +0.018 [-0.022, +0.059] | -0.009 [-0.059, +0.040] | -0.024 [-0.096, +0.042] |
| RETFound full fine-tune | IDRiD | +0.022 [-0.034, +0.078] | +0.046 [-0.028, +0.119] | +0.079 [-0.009, +0.166] |

## B. Prevalence-invariant performance

### Per-grade recall

| Model | Dataset | 0 | 1 | 2 | 3 | 4 |
|---|---|--:|--:|--:|--:|--:|
| EfficientNet-B3 | APTOS-internal | 0.96 | 0.53 | 0.79 | 0.49 | 0.44 |
| EfficientNet-B3 | Messidor-2 | 0.97 | 0.01 | 0.22 | 0.13 | 0.43 |
| EfficientNet-B3 | IDRiD | 0.49 | 0.77 | 0.68 | 0.25 | 0.30 |
| ResNet-50 | APTOS-internal | 0.96 | 0.66 | 0.73 | 0.51 | 0.29 |
| ResNet-50 | Messidor-2 | 0.95 | 0.01 | 0.20 | 0.28 | 0.26 |
| ResNet-50 | IDRiD | 0.50 | 0.86 | 0.52 | 0.38 | 0.12 |
| RETFound linear probe | APTOS-internal | 0.95 | 0.61 | 0.58 | 0.51 | 0.51 |
| RETFound linear probe | Messidor-2 | 0.90 | 0.00 | 0.18 | 0.41 | 0.63 |
| RETFound linear probe | IDRiD | 0.22 | 0.45 | 0.18 | 0.75 | 0.38 |
| RETFound full fine-tune | APTOS-internal | 0.95 | 0.59 | 0.69 | 0.51 | 0.49 |
| RETFound full fine-tune | Messidor-2 | 0.97 | 0.00 | 0.22 | 0.16 | 0.40 |
| RETFound full fine-tune | IDRiD | 0.51 | 0.86 | 0.33 | 0.69 | 0.16 |

### Referable DR: sensitivity / specificity (prevalence-invariant), and PPV at own vs APTOS prevalence

| Model | Dataset | prev. | sens [CI] | spec [CI] | PPV own | PPV @APTOS prev |
|---|---|--:|---|---|--:|--:|
| EfficientNet-B3 | APTOS-internal | 0.41 | 0.90 [0.86, 0.93] | 0.93 [0.90, 0.95] | 0.90 | 0.90 |
| EfficientNet-B3 | Messidor-2 | 0.26 | 0.39 [0.34, 0.43] | 0.97 [0.96, 0.98] | 0.84 | 0.91 |
| EfficientNet-B3 | IDRiD | 0.67 | 0.86 [0.81, 0.89] | 0.96 [0.92, 0.98] | 0.98 | 0.94 |
| ResNet-50 | APTOS-internal | 0.41 | 0.87 [0.83, 0.90] | 0.95 [0.93, 0.97] | 0.93 | 0.93 |
| ResNet-50 | Messidor-2 | 0.26 | 0.36 [0.31, 0.40] | 0.96 [0.94, 0.97] | 0.74 | 0.85 |
| ResNet-50 | IDRiD | 0.67 | 0.74 [0.69, 0.79] | 0.99 [0.95, 1.00] | 0.99 | 0.97 |
| RETFound linear probe | APTOS-internal | 0.41 | 0.86 [0.81, 0.89] | 0.95 [0.93, 0.97] | 0.92 | 0.92 |
| RETFound linear probe | Messidor-2 | 0.26 | 0.43 [0.38, 0.47] | 0.90 [0.88, 0.92] | 0.61 | 0.75 |
| RETFound linear probe | IDRiD | 0.67 | 0.84 [0.80, 0.88] | 0.64 [0.56, 0.71] | 0.82 | 0.61 |
| RETFound full fine-tune | APTOS-internal | 0.41 | 0.91 [0.87, 0.94] | 0.95 [0.93, 0.97] | 0.93 | 0.93 |
| RETFound full fine-tune | Messidor-2 | 0.26 | 0.39 [0.35, 0.44] | 0.98 [0.97, 0.98] | 0.85 | 0.92 |
| RETFound full fine-tune | IDRiD | 0.67 | 0.76 [0.71, 0.80] | 0.93 [0.87, 0.96] | 0.95 | 0.88 |

### Where reference-standard grade-1 (mild) images are predicted

| Model | Dataset | n | →0 | →1 | →2 | →3 | →4 |
|---|---|--:|--:|--:|--:|--:|--:|
| EfficientNet-B3 | APTOS-internal | 74 | 8 | 39 | 26 | 0 | 1 |
| EfficientNet-B3 | Messidor-2 | 270 | 259 | 4 | 6 | 0 | 1 |
| EfficientNet-B3 | IDRiD | 22 | 4 | 17 | 1 | 0 | 0 |
| ResNet-50 | APTOS-internal | 74 | 8 | 49 | 17 | 0 | 0 |
| ResNet-50 | Messidor-2 | 270 | 255 | 2 | 13 | 0 | 0 |
| ResNet-50 | IDRiD | 22 | 2 | 19 | 1 | 0 | 0 |
| RETFound linear probe | APTOS-internal | 74 | 10 | 45 | 14 | 1 | 4 |
| RETFound linear probe | Messidor-2 | 270 | 242 | 0 | 17 | 0 | 11 |
| RETFound linear probe | IDRiD | 22 | 1 | 10 | 6 | 5 | 0 |
| RETFound full fine-tune | APTOS-internal | 74 | 10 | 44 | 16 | 3 | 1 |
| RETFound full fine-tune | Messidor-2 | 270 | 266 | 0 | 2 | 1 | 1 |
| RETFound full fine-tune | IDRiD | 22 | 1 | 19 | 1 | 1 | 0 |
