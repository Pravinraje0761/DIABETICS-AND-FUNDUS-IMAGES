# Phase 11 - review-response analyses

Verification: 60 metric cells recomputed from prediction files; **45 mismatch(es)** vs `master_results.csv` (tolerance 5e-4).

- baseline_efficientnet_b3/APTOS-internal/accuracy: recomputed 0.803547 vs csv nan
- baseline_efficientnet_b3/APTOS-internal/quadratic_weighted_kappa: recomputed 0.875958 vs csv nan
- baseline_efficientnet_b3/APTOS-internal/macro_f1: recomputed 0.644189 vs csv nan
- baseline_efficientnet_b3/APTOS-internal/balanced_accuracy: recomputed 0.641221 vs csv nan
- baseline_efficientnet_b3/APTOS-internal/auc_ovr_macro: recomputed 0.927897 vs csv nan
- baseline_efficientnet_b3/Messidor-2/accuracy: recomputed 0.625573 vs csv nan
- baseline_efficientnet_b3/Messidor-2/quadratic_weighted_kappa: recomputed 0.498811 vs csv nan
- baseline_efficientnet_b3/Messidor-2/macro_f1: recomputed 0.359567 vs csv nan
- baseline_efficientnet_b3/Messidor-2/balanced_accuracy: recomputed 0.352672 vs csv nan
- baseline_efficientnet_b3/Messidor-2/auc_ovr_macro: recomputed 0.775205 vs csv nan
- baseline_efficientnet_b3/IDRiD/accuracy: recomputed 0.496703 vs csv nan
- baseline_efficientnet_b3/IDRiD/quadratic_weighted_kappa: recomputed 0.759687 vs csv nan
- baseline_efficientnet_b3/IDRiD/macro_f1: recomputed 0.440984 vs csv nan
- baseline_efficientnet_b3/IDRiD/balanced_accuracy: recomputed 0.497492 vs csv nan
- baseline_efficientnet_b3/IDRiD/auc_ovr_macro: recomputed 0.830223 vs csv nan
- baseline_resnet50/APTOS-internal/accuracy: recomputed 0.78854 vs csv nan
- baseline_resnet50/APTOS-internal/quadratic_weighted_kappa: recomputed 0.874669 vs csv nan
- baseline_resnet50/APTOS-internal/macro_f1: recomputed 0.616447 vs csv nan
- baseline_resnet50/APTOS-internal/balanced_accuracy: recomputed 0.630313 vs csv nan
- baseline_resnet50/APTOS-internal/auc_ovr_macro: recomputed 0.923767 vs csv nan
- baseline_resnet50/Messidor-2/accuracy: recomputed 0.613532 vs csv nan
- baseline_resnet50/Messidor-2/quadratic_weighted_kappa: recomputed 0.434668 vs csv nan
- baseline_resnet50/Messidor-2/macro_f1: recomputed 0.354488 vs csv nan
- baseline_resnet50/Messidor-2/balanced_accuracy: recomputed 0.33924 vs csv nan
- baseline_resnet50/Messidor-2/auc_ovr_macro: recomputed 0.753679 vs csv nan
- baseline_resnet50/IDRiD/accuracy: recomputed 0.448352 vs csv nan
- baseline_resnet50/IDRiD/quadratic_weighted_kappa: recomputed 0.707343 vs csv nan
- baseline_resnet50/IDRiD/macro_f1: recomputed 0.403428 vs csv nan
- baseline_resnet50/IDRiD/balanced_accuracy: recomputed 0.476989 vs csv nan
- baseline_resnet50/IDRiD/auc_ovr_macro: recomputed 0.840792 vs csv nan
- retfound_linprobe/APTOS-internal/accuracy: recomputed 0.758527 vs csv nan
- retfound_linprobe/APTOS-internal/quadratic_weighted_kappa: recomputed 0.846888 vs csv nan
- retfound_linprobe/APTOS-internal/macro_f1: recomputed 0.60667 vs csv nan
- retfound_linprobe/APTOS-internal/balanced_accuracy: recomputed 0.633462 vs csv nan
- retfound_linprobe/APTOS-internal/auc_ovr_macro: recomputed 0.908234 vs csv nan
- retfound_linprobe/Messidor-2/accuracy: recomputed 0.58945 vs csv nan
- retfound_linprobe/Messidor-2/quadratic_weighted_kappa: recomputed 0.393775 vs csv nan
- retfound_linprobe/Messidor-2/macro_f1: recomputed 0.346873 vs csv nan
- retfound_linprobe/Messidor-2/balanced_accuracy: recomputed 0.423284 vs csv nan
- retfound_linprobe/Messidor-2/auc_ovr_macro: recomputed 0.757561 vs csv nan
- retfound_linprobe/IDRiD/accuracy: recomputed 0.336264 vs csv nan
- retfound_linprobe/IDRiD/quadratic_weighted_kappa: recomputed 0.625833 vs csv nan
- retfound_linprobe/IDRiD/macro_f1: recomputed 0.3255 vs csv nan
- retfound_linprobe/IDRiD/balanced_accuracy: recomputed 0.395217 vs csv nan
- retfound_linprobe/IDRiD/auc_ovr_macro: recomputed 0.778393 vs csv nan

## Calibration (ECE / MCE / Brier / NLL)

| Model | Dataset | ECE (95% CI) | MCE | Brier | NLL |
|---|---|---|---|---|---|
| baseline_efficientnet_b3 | APTOS-internal | 0.101 (0.080-0.129) | 0.295 | 0.300 | 0.645 |
| baseline_efficientnet_b3 | Messidor-2 | 0.279 (0.259-0.302) | 0.439 | 0.644 | 1.742 |
| baseline_efficientnet_b3 | IDRiD | 0.294 (0.253-0.342) | 0.437 | 0.764 | 1.667 |
| baseline_resnet50 | APTOS-internal | 0.058 (0.046-0.091) | 0.307 | 0.311 | 0.608 |
| baseline_resnet50 | Messidor-2 | 0.265 (0.245-0.290) | 0.400 | 0.650 | 1.583 |
| baseline_resnet50 | IDRiD | 0.260 (0.220-0.312) | 0.459 | 0.778 | 1.445 |
| retfound_linprobe | APTOS-internal | 0.182 (0.159-0.211) | 0.249 | 0.395 | 0.815 |
| retfound_linprobe | Messidor-2 | 0.121 (0.100-0.145) | 0.155 | 0.594 | 1.181 |
| retfound_linprobe | IDRiD | 0.127 (0.092-0.172) | 0.419 | 0.758 | 1.462 |
| retfound_full | APTOS-internal | 0.179 (0.161-0.213) | 0.258 | 0.369 | 0.748 |
| retfound_full | Messidor-2 | 0.095 (0.075-0.118) | 0.230 | 0.563 | 1.158 |
| retfound_full | IDRiD | 0.129 (0.101-0.184) | 0.449 | 0.715 | 1.377 |

Files: `per_class_ci.csv`, `referable_roc_pr.csv`, `multiplicity.csv`, `training_accounting.csv`, `domain_shift.md`.
