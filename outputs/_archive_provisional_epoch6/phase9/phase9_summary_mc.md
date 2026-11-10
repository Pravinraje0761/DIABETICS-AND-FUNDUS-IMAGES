# Phase 9 - Statistical analysis (mc)

_Generated 2026-09-10T04:42:38+00:00. Percentile bootstrap, B=2000, seed=42, two-sided alpha=0.05. Predictions: **mc** - mean softmax over T MC-dropout passes (`outputs\phase8\mc/`); compare against the single-pass `phase9_summary.md`._

## 1. Headline metrics with 95% CI

| Model | Dataset | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |
|---|---|---|---|---|---|---|
| baseline_efficientnet_b3 | APTOS-internal | 0.809 (0.780–0.838) | 0.881 (0.854–0.906) | 0.657 (0.608–0.701) | 0.657 (0.608–0.705) | 0.930 (0.914–0.945) |
| baseline_efficientnet_b3 | Messidor-2 | 0.626 (0.603–0.647) | 0.499 (0.449–0.547) | 0.340 (0.300–0.377) | 0.360 (0.322–0.396) | 0.785 (0.772–0.798) |
| baseline_efficientnet_b3 | IDRiD | 0.508 (0.462–0.556) | 0.764 (0.720–0.803) | 0.453 (0.406–0.500) | 0.508 (0.458–0.561) | 0.843 (0.818–0.867) |
| baseline_resnet50 | APTOS-internal | 0.787 (0.757–0.817) | 0.874 (0.847–0.897) | 0.615 (0.566–0.661) | 0.629 (0.584–0.676) | 0.924 (0.909–0.939) |
| baseline_resnet50 | Messidor-2 | 0.614 (0.591–0.636) | 0.435 (0.382–0.482) | 0.361 (0.318–0.405) | 0.345 (0.310–0.385) | 0.754 (0.738–0.769) |
| baseline_resnet50 | IDRiD | 0.451 (0.402–0.497) | 0.710 (0.665–0.754) | 0.404 (0.359–0.451) | 0.478 (0.430–0.526) | 0.841 (0.818–0.862) |
| retfound_linprobe | APTOS-internal | 0.749 (0.718–0.780) | 0.829 (0.798–0.859) | 0.599 (0.553–0.642) | 0.632 (0.583–0.682) | 0.903 (0.882–0.922) |
| retfound_linprobe | Messidor-2 | 0.577 (0.554–0.599) | 0.370 (0.313–0.422) | 0.336 (0.303–0.366) | 0.427 (0.386–0.466) | 0.757 (0.738–0.774) |
| retfound_linprobe | IDRiD | 0.369 (0.327–0.415) | 0.651 (0.600–0.698) | 0.359 (0.317–0.401) | 0.427 (0.373–0.483) | 0.776 (0.748–0.803) |
| retfound_full | APTOS-internal | 0.775 (0.744–0.805) | 0.868 (0.839–0.893) | 0.627 (0.579–0.670) | 0.641 (0.591–0.690) | 0.921 (0.904–0.937) |
| retfound_full | Messidor-2 | 0.594 (0.570–0.617) | 0.403 (0.346–0.456) | 0.379 (0.335–0.420) | 0.362 (0.322–0.403) | 0.754 (0.737–0.768) |
| retfound_full | IDRiD | 0.387 (0.343–0.431) | 0.694 (0.643–0.737) | 0.396 (0.350–0.438) | 0.476 (0.428–0.522) | 0.813 (0.788–0.839) |

## 2. Internal->external drop with 95% CI (QWK)

| Model | External set | QWK internal | QWK external | Drop (95% CI) |
|---|---|--:|--:|---|
| baseline_efficientnet_b3 | Messidor-2 | 0.881 | 0.499 | 0.382 (0.327–0.438) |
| baseline_efficientnet_b3 | IDRiD | 0.881 | 0.764 | 0.116 (0.069–0.166) |
| baseline_resnet50 | Messidor-2 | 0.874 | 0.435 | 0.439 (0.387–0.498) |
| baseline_resnet50 | IDRiD | 0.874 | 0.710 | 0.164 (0.114–0.218) |
| retfound_linprobe | Messidor-2 | 0.829 | 0.370 | 0.459 (0.399–0.523) |
| retfound_linprobe | IDRiD | 0.829 | 0.651 | 0.178 (0.120–0.236) |
| retfound_full | Messidor-2 | 0.868 | 0.403 | 0.464 (0.405–0.526) |
| retfound_full | IDRiD | 0.868 | 0.694 | 0.174 (0.124–0.229) |

## 3. Difference in QWK drop vs the EfficientNet-B3 baseline

Positive = the model lost *more* ordinal agreement than the baseline on that external set. CI excluding 0 = a statistically distinguishable difference in robustness.

| Model | External set | Drop (model) | Drop (baseline) | Difference (95% CI) | Excludes 0 |
|---|---|--:|--:|---|:--:|
| baseline_resnet50 | Messidor-2 | 0.439 | 0.382 | 0.057 (0.010–0.108) | yes |
| baseline_resnet50 | IDRiD | 0.164 | 0.116 | 0.047 (-0.001–0.096) | no |
| retfound_linprobe | Messidor-2 | 0.459 | 0.382 | 0.078 (0.018–0.136) | yes |
| retfound_linprobe | IDRiD | 0.178 | 0.116 | 0.061 (-0.004–0.123) | no |
| retfound_full | Messidor-2 | 0.464 | 0.382 | 0.083 (0.033–0.132) | yes |
| retfound_full | IDRiD | 0.174 | 0.116 | 0.058 (0.000–0.114) | yes |

## 4. DeLong paired-AUC test (vs EfficientNet-B3 baseline)

Referable-DR (ICDR grade >= 2) binarisation; score = P(grade>=2).

| Dataset | Model | AUC baseline | AUC model | dAUC | z | p | sig. |
|---|---|--:|--:|--:|--:|--:|:--:|
| APTOS-internal | baseline_resnet50 | 0.982 | 0.978 | -0.004 | 1.34 | 0.181 | ns |
| APTOS-internal | retfound_linprobe | 0.982 | 0.976 | -0.006 | 1.51 | 0.132 | ns |
| APTOS-internal | retfound_full | 0.982 | 0.978 | -0.004 | 0.93 | 0.354 | ns |
| Messidor-2 | baseline_resnet50 | 0.844 | 0.789 | -0.056 | 5.73 | 0.000 | * |
| Messidor-2 | retfound_linprobe | 0.844 | 0.773 | -0.072 | 5.71 | 0.000 | * |
| Messidor-2 | retfound_full | 0.844 | 0.780 | -0.065 | 5.49 | 0.000 | * |
| IDRiD | baseline_resnet50 | 0.968 | 0.949 | -0.019 | 2.66 | 0.008 | * |
| IDRiD | retfound_linprobe | 0.968 | 0.913 | -0.055 | 4.78 | 0.000 | * |
| IDRiD | retfound_full | 0.968 | 0.875 | -0.093 | 6.14 | 0.000 | * |

Full per-grade one-vs-rest DeLong results: `outputs\phase9\delong_mc.csv`.
