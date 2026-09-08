# Phase 6 - Master results

_Generated 2026-09-08T08:57:40+00:00_

## Master table (model x dataset)

| Model | Dataset | n | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |
|---|---|--:|--:|--:|--:|--:|--:|
| baseline_efficientnet_b3 | APTOS-internal | 733 | 0.804 | 0.876 | 0.644 | 0.641 | 0.928 |
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 0.626 | 0.499 | 0.360 | 0.353 | 0.775 |
| baseline_efficientnet_b3 | IDRiD | 455 | 0.497 | 0.760 | 0.441 | 0.497 | 0.830 |
| baseline_resnet50 | APTOS-internal | 733 | 0.789 | 0.875 | 0.616 | 0.630 | 0.924 |
| baseline_resnet50 | Messidor-2 | 1744 | 0.614 | 0.435 | 0.354 | 0.339 | 0.754 |
| baseline_resnet50 | IDRiD | 455 | 0.448 | 0.707 | 0.403 | 0.477 | 0.841 |
| retfound_linprobe | APTOS-internal | 733 | 0.759 | 0.847 | 0.607 | 0.633 | 0.908 |
| retfound_linprobe | Messidor-2 | 1744 | 0.589 | 0.394 | 0.347 | 0.423 | 0.758 |
| retfound_linprobe | IDRiD | 455 | 0.336 | 0.626 | 0.325 | 0.395 | 0.778 |
| retfound_full | APTOS-internal | 733 | 0.769 | 0.869 | 0.616 | 0.629 | 0.915 |
| retfound_full | Messidor-2 | 1744 | 0.602 | 0.368 | 0.362 | 0.346 | 0.758 |
| retfound_full | IDRiD | 455 | 0.420 | 0.720 | 0.423 | 0.480 | 0.821 |

## Internal -> external drop (QWK)

| Model | External set | QWK internal | QWK external | Drop |
|---|---|--:|--:|--:|
| baseline_efficientnet_b3 | Messidor-2 | 0.876 | 0.499 | 0.377 |
| baseline_efficientnet_b3 | IDRiD | 0.876 | 0.760 | 0.116 |
| baseline_resnet50 | Messidor-2 | 0.875 | 0.435 | 0.440 |
| baseline_resnet50 | IDRiD | 0.875 | 0.707 | 0.167 |
| retfound_linprobe | Messidor-2 | 0.847 | 0.394 | 0.453 |
| retfound_linprobe | IDRiD | 0.847 | 0.626 | 0.221 |
| retfound_full | Messidor-2 | 0.869 | 0.368 | 0.501 |
| retfound_full | IDRiD | 0.869 | 0.720 | 0.150 |

Raw per-image predictions: `outputs/phase6/<model>__<dataset>_preds.csv` (kept for Phase 8 uncertainty + Phase 9 bootstrap/DeLong).