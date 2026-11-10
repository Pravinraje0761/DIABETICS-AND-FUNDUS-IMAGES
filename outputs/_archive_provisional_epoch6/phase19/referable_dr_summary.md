# Phase 19 -- referable-DR (ICDR grade >= 2) secondary task

_Generated 2026-09-11T05:53:18+00:00. Referable = ICDR grade >= 2. Decision-based metrics use the 5-class model's own argmax (y_pred >= 2); AUROC/AUPRC use summed P(grade>=2). Percentile bootstrap, B=2000, seed=42._

## 1. Headline metrics with 95% CI

| Model | Dataset | n | n-ref | Sens | Spec | AUROC | AUPRC | Sens@85%spec | Spec@85%sens |
|---|---|--:|--:|---|---|---|---|--:|--:|
| baseline_efficientnet_b3 | APTOS-internal | 733 | 298 | 0.903 (0.868-0.934) | 0.931 (0.907-0.954) | 0.980 (0.971-0.987) | 0.968 (0.953-0.980) | 0.987 | 0.947 |
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 457 | 0.387 (0.345-0.433) | 0.974 (0.966-0.983) | 0.842 (0.819-0.864) | 0.732 (0.694-0.771) | 0.702 | 0.576 |
| baseline_efficientnet_b3 | IDRiD | 455 | 304 | 0.855 (0.815-0.894) | 0.960 (0.927-0.987) | 0.962 (0.946-0.977) | 0.984 (0.976-0.990) | 0.934 | 0.967 |
| baseline_resnet50 | APTOS-internal | 733 | 298 | 0.869 (0.829-0.905) | 0.954 (0.935-0.973) | 0.978 (0.968-0.986) | 0.963 (0.944-0.979) | 0.983 | 0.954 |
| baseline_resnet50 | Messidor-2 | 1744 | 457 | 0.357 (0.312-0.399) | 0.956 (0.944-0.966) | 0.789 (0.764-0.814) | 0.630 (0.585-0.675) | 0.589 | 0.487 |
| baseline_resnet50 | IDRiD | 455 | 304 | 0.743 (0.694-0.795) | 0.987 (0.967-1.000) | 0.950 (0.930-0.967) | 0.978 (0.968-0.986) | 0.905 | 0.980 |
| retfound_linprobe | APTOS-internal | 733 | 298 | 0.859 (0.820-0.896) | 0.952 (0.930-0.971) | 0.976 (0.966-0.986) | 0.969 (0.955-0.981) | 0.977 | 0.956 |
| retfound_linprobe | Messidor-2 | 1744 | 457 | 0.429 (0.383-0.473) | 0.902 (0.885-0.918) | 0.781 (0.754-0.806) | 0.616 (0.571-0.661) | 0.549 | 0.498 |
| retfound_linprobe | IDRiD | 455 | 304 | 0.842 (0.799-0.883) | 0.636 (0.559-0.712) | 0.903 (0.874-0.929) | 0.956 (0.941-0.970) | 0.786 | 0.768 |
| retfound_full | APTOS-internal | 733 | 298 | 0.879 (0.840-0.915) | 0.938 (0.913-0.961) | 0.978 (0.967-0.987) | 0.972 (0.960-0.983) | 0.987 | 0.956 |
| retfound_full | Messidor-2 | 1744 | 457 | 0.219 (0.182-0.257) | 0.984 (0.976-0.990) | 0.789 (0.765-0.815) | 0.632 (0.589-0.676) | 0.530 | 0.552 |
| retfound_full | IDRiD | 455 | 304 | 0.668 (0.615-0.719) | 0.901 (0.850-0.946) | 0.873 (0.839-0.904) | 0.947 (0.930-0.961) | 0.773 | 0.675 |

## 2. Internal->external AUROC drop with 95% CI

Mirrors the primary-task QWK-drop table (Phase 9 sec. 2) for the referable-DR binarisation, so the manuscript can state whether the generalisation gap persists beyond 5-class ordinal grading.

| Model | External set | AUROC internal | AUROC external | Drop (95% CI) |
|---|---|--:|--:|---|
| baseline_efficientnet_b3 | Messidor-2 | 0.980 | 0.842 | 0.138 (0.115-0.161) |
| baseline_efficientnet_b3 | IDRiD | 0.980 | 0.962 | 0.018 (0.001-0.036) |
| baseline_resnet50 | Messidor-2 | 0.978 | 0.789 | 0.189 (0.162-0.215) |
| baseline_resnet50 | IDRiD | 0.978 | 0.950 | 0.028 (0.008-0.050) |
| retfound_linprobe | Messidor-2 | 0.976 | 0.781 | 0.195 (0.169-0.223) |
| retfound_linprobe | IDRiD | 0.976 | 0.903 | 0.074 (0.045-0.103) |
| retfound_full | Messidor-2 | 0.978 | 0.789 | 0.188 (0.161-0.215) |
| retfound_full | IDRiD | 0.978 | 0.873 | 0.105 (0.073-0.139) |

## 3. Between-model AUROC difference (significance test)

Not recomputed here -- see the existing DeLong paired-AUC test in `outputs/phase9/delong.csv` (target = `referable_DR(grade>=2)`), which uses this exact binarisation and score.
