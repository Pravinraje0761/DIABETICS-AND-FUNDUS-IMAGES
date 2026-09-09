# Phase 11 - review-response analyses

Verification: 60 metric cells recomputed from prediction files; **0 mismatch(es)** vs `master_results.csv` (tolerance 5e-4).

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
| retfound_full | APTOS-internal | 0.177 (0.156-0.209) | 0.275 | 0.380 | 0.777 |
| retfound_full | Messidor-2 | 0.160 (0.139-0.181) | 0.245 | 0.594 | 1.185 |
| retfound_full | IDRiD | 0.098 (0.071-0.149) | 0.745 | 0.702 | 1.302 |

Files: `per_class_ci.csv`, `referable_roc_pr.csv`, `multiplicity.csv`, `training_accounting.csv`, `domain_shift.md`.
