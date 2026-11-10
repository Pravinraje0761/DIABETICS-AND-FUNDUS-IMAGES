# Phase 19 -- referable-DR (ICDR grade >= 2) secondary task

_Generated 2026-11-06T09:44:13+00:00. Referable = ICDR grade >= 2. Decision-based metrics use the 5-class model's own argmax (y_pred >= 2); AUROC/AUPRC use summed P(grade>=2). Percentile bootstrap, B=2000, seed=42._

## 1. Headline metrics with 95% CI

| Model | Dataset | n | n-ref | Sens | Spec | AUROC | AUPRC | Sens@85%spec | Spec@85%sens |
|---|---|--:|--:|---|---|---|---|--:|--:|
| baseline_efficientnet_b3 | APTOS-internal | 733 | 298 | 0.903 (0.868-0.934) | 0.931 (0.907-0.954) | 0.980 (0.971-0.987) | 0.968 (0.953-0.980) | 0.987 | 0.947 |
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 457 | 0.387 (0.345-0.433) | 0.974 (0.966-0.983) | 0.842 (0.819-0.864) | 0.732 (0.694-0.771) | 0.702 | 0.576 |
| baseline_efficientnet_b3 | IDRiD | 516 | 323 | 0.827 (0.783-0.866) | 0.907 (0.865-0.945) | 0.927 (0.904-0.948) | 0.961 (0.946-0.973) | 0.854 | 0.855 |
| baseline_resnet50 | APTOS-internal | 733 | 298 | 0.869 (0.829-0.905) | 0.954 (0.935-0.973) | 0.978 (0.968-0.986) | 0.963 (0.944-0.979) | 0.983 | 0.954 |
| baseline_resnet50 | Messidor-2 | 1744 | 457 | 0.357 (0.312-0.399) | 0.956 (0.944-0.966) | 0.789 (0.764-0.814) | 0.630 (0.585-0.675) | 0.589 | 0.487 |
| baseline_resnet50 | IDRiD | 516 | 323 | 0.715 (0.667-0.764) | 0.953 (0.920-0.980) | 0.912 (0.887-0.936) | 0.951 (0.933-0.966) | 0.842 | 0.839 |
| retfound_linprobe | APTOS-internal | 733 | 298 | 0.859 (0.820-0.896) | 0.952 (0.930-0.971) | 0.976 (0.966-0.986) | 0.969 (0.955-0.981) | 0.977 | 0.956 |
| retfound_linprobe | Messidor-2 | 1744 | 457 | 0.429 (0.383-0.473) | 0.902 (0.885-0.918) | 0.781 (0.754-0.806) | 0.616 (0.571-0.661) | 0.549 | 0.498 |
| retfound_linprobe | IDRiD | 516 | 323 | 0.827 (0.784-0.868) | 0.658 (0.586-0.726) | 0.882 (0.853-0.909) | 0.936 (0.916-0.952) | 0.759 | 0.710 |
| retfound_full | APTOS-internal | 733 | 298 | 0.909 (0.877-0.940) | 0.952 (0.931-0.972) | 0.967 (0.950-0.983) | 0.968 (0.954-0.981) | 0.973 | 0.968 |
| retfound_full | Messidor-2 | 1744 | 457 | 0.392 (0.347-0.434) | 0.975 (0.966-0.984) | 0.817 (0.791-0.840) | 0.716 (0.678-0.752) | 0.659 | 0.521 |
| retfound_full | IDRiD | 516 | 323 | 0.734 (0.684-0.781) | 0.927 (0.888-0.959) | 0.876 (0.844-0.906) | 0.942 (0.925-0.957) | 0.814 | 0.642 |

## 2. Internal->external AUROC drop with 95% CI

Mirrors the primary-task QWK-drop table (Phase 9 sec. 2) for the referable-DR binarisation, so the manuscript can state whether the generalisation gap persists beyond 5-class ordinal grading.

| Model | External set | AUROC internal | AUROC external | Drop (95% CI) |
|---|---|--:|--:|---|
| baseline_efficientnet_b3 | Messidor-2 | 0.980 | 0.842 | 0.138 (0.115-0.161) |
| baseline_efficientnet_b3 | IDRiD | 0.980 | 0.927 | 0.053 (0.031-0.076) |
| baseline_resnet50 | Messidor-2 | 0.978 | 0.789 | 0.189 (0.162-0.215) |
| baseline_resnet50 | IDRiD | 0.978 | 0.912 | 0.065 (0.040-0.094) |
| retfound_linprobe | Messidor-2 | 0.976 | 0.781 | 0.195 (0.169-0.223) |
| retfound_linprobe | IDRiD | 0.976 | 0.882 | 0.094 (0.066-0.125) |
| retfound_full | Messidor-2 | 0.967 | 0.817 | 0.150 (0.122-0.180) |
| retfound_full | IDRiD | 0.967 | 0.876 | 0.091 (0.056-0.126) |

## 3. Between-model AUROC difference (significance test)

Not recomputed here -- see the existing DeLong paired-AUC test in `outputs/phase9/delong.csv` (target = `referable_DR(grade>=2)`), which uses this exact binarisation and score.
