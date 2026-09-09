# Phase 9 - Statistical analysis

_Generated 2026-09-08T11:14:30+00:00. Percentile bootstrap, B=2000, seed=42, two-sided alpha=0.05._

## 1. Headline metrics with 95% CI

| Model | Dataset | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |
|---|---|---|---|---|---|---|
| baseline_efficientnet_b3 | APTOS-internal | 0.804 (0.776–0.832) | 0.876 (0.849–0.901) | 0.644 (0.594–0.688) | 0.641 (0.592–0.690) | 0.928 (0.911–0.943) |
| baseline_efficientnet_b3 | Messidor-2 | 0.626 (0.603–0.648) | 0.499 (0.452–0.543) | 0.360 (0.316–0.400) | 0.353 (0.315–0.390) | 0.775 (0.761–0.789) |
| baseline_efficientnet_b3 | IDRiD | 0.497 (0.451–0.545) | 0.760 (0.715–0.798) | 0.441 (0.395–0.488) | 0.497 (0.448–0.550) | 0.830 (0.805–0.856) |
| baseline_resnet50 | APTOS-internal | 0.789 (0.759–0.819) | 0.875 (0.848–0.898) | 0.616 (0.567–0.663) | 0.630 (0.585–0.677) | 0.924 (0.908–0.939) |
| baseline_resnet50 | Messidor-2 | 0.614 (0.591–0.635) | 0.435 (0.381–0.481) | 0.354 (0.309–0.397) | 0.339 (0.304–0.377) | 0.754 (0.738–0.769) |
| baseline_resnet50 | IDRiD | 0.448 (0.400–0.497) | 0.707 (0.663–0.751) | 0.403 (0.358–0.450) | 0.477 (0.428–0.525) | 0.841 (0.818–0.862) |
| retfound_linprobe | APTOS-internal | 0.759 (0.728–0.789) | 0.847 (0.819–0.873) | 0.607 (0.562–0.649) | 0.633 (0.588–0.682) | 0.908 (0.889–0.927) |
| retfound_linprobe | Messidor-2 | 0.589 (0.565–0.612) | 0.394 (0.338–0.447) | 0.347 (0.314–0.377) | 0.423 (0.379–0.462) | 0.758 (0.738–0.774) |
| retfound_linprobe | IDRiD | 0.336 (0.297–0.376) | 0.626 (0.572–0.676) | 0.325 (0.284–0.365) | 0.395 (0.342–0.451) | 0.778 (0.751–0.805) |
| retfound_full | APTOS-internal | 0.769 (0.739–0.798) | 0.869 (0.842–0.893) | 0.616 (0.571–0.658) | 0.629 (0.582–0.678) | 0.915 (0.896–0.934) |
| retfound_full | Messidor-2 | 0.602 (0.579–0.625) | 0.368 (0.311–0.422) | 0.362 (0.316–0.402) | 0.346 (0.307–0.387) | 0.758 (0.742–0.772) |
| retfound_full | IDRiD | 0.420 (0.376–0.466) | 0.720 (0.673–0.759) | 0.423 (0.378–0.466) | 0.480 (0.427–0.534) | 0.821 (0.796–0.847) |

## 2. Internal->external drop with 95% CI (QWK)

| Model | External set | QWK internal | QWK external | Drop (95% CI) |
|---|---|--:|--:|---|
| baseline_efficientnet_b3 | Messidor-2 | 0.876 | 0.499 | 0.377 (0.323–0.433) |
| baseline_efficientnet_b3 | IDRiD | 0.876 | 0.760 | 0.116 (0.068–0.166) |
| baseline_resnet50 | Messidor-2 | 0.875 | 0.435 | 0.440 (0.388–0.499) |
| baseline_resnet50 | IDRiD | 0.875 | 0.707 | 0.167 (0.117–0.222) |
| retfound_linprobe | Messidor-2 | 0.847 | 0.394 | 0.453 (0.393–0.516) |
| retfound_linprobe | IDRiD | 0.847 | 0.626 | 0.221 (0.163–0.285) |
| retfound_full | Messidor-2 | 0.869 | 0.368 | 0.501 (0.441–0.563) |
| retfound_full | IDRiD | 0.869 | 0.720 | 0.150 (0.101–0.202) |

## 3. Difference in QWK drop vs the EfficientNet-B3 baseline

Positive = the model lost *more* ordinal agreement than the baseline on that external set. CI excluding 0 = a statistically distinguishable difference in robustness.

| Model | External set | Drop (model) | Drop (baseline) | Difference (95% CI) | Excludes 0 |
|---|---|--:|--:|---|:--:|
| baseline_resnet50 | Messidor-2 | 0.440 | 0.377 | 0.063 (0.017–0.109) | yes |
| baseline_resnet50 | IDRiD | 0.167 | 0.116 | 0.051 (0.003–0.102) | yes |
| retfound_linprobe | Messidor-2 | 0.453 | 0.377 | 0.076 (0.019–0.132) | yes |
| retfound_linprobe | IDRiD | 0.221 | 0.116 | 0.105 (0.038–0.170) | yes |
| retfound_full | Messidor-2 | 0.501 | 0.377 | 0.124 (0.074–0.175) | yes |
| retfound_full | IDRiD | 0.150 | 0.116 | 0.033 (-0.022–0.089) | no |

## 4. DeLong paired-AUC test (vs EfficientNet-B3 baseline)

Referable-DR (ICDR grade >= 2) binarisation; score = P(grade>=2).

| Dataset | Model | AUC baseline | AUC model | dAUC | z | p | sig. |
|---|---|--:|--:|--:|--:|--:|:--:|
| APTOS-internal | baseline_resnet50 | 0.980 | 0.978 | -0.002 | 0.63 | 0.530 | ns |
| APTOS-internal | retfound_linprobe | 0.980 | 0.976 | -0.003 | 0.72 | 0.474 | ns |
| APTOS-internal | retfound_full | 0.980 | 0.978 | -0.002 | 0.44 | 0.660 | ns |
| Messidor-2 | baseline_resnet50 | 0.842 | 0.789 | -0.053 | 5.19 | 0.000 | * |
| Messidor-2 | retfound_linprobe | 0.842 | 0.781 | -0.060 | 4.74 | 0.000 | * |
| Messidor-2 | retfound_full | 0.842 | 0.789 | -0.052 | 4.49 | 0.000 | * |
| IDRiD | baseline_resnet50 | 0.962 | 0.950 | -0.013 | 1.77 | 0.076 | ns |
| IDRiD | retfound_linprobe | 0.962 | 0.903 | -0.059 | 4.88 | 0.000 | * |
| IDRiD | retfound_full | 0.962 | 0.873 | -0.090 | 5.97 | 0.000 | * |

Full per-grade one-vs-rest DeLong results: `outputs/phase9/delong.csv`.
