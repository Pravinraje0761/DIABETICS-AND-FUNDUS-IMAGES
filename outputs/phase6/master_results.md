# Phase 6 - master results (rebuilt 2026-10-21 after the completed RETFound full fine-tune)

| Model | Dataset | n | Acc | QWK | Macro-F1 | Bal. acc | AUC |
|---|---|---|---|---|---|---|---|
| baseline_efficientnet_b3 | APTOS-internal | 733 | 0.804 | 0.876 | 0.644 | 0.641 | 0.928 |
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 0.626 | 0.499 | 0.360 | 0.353 | 0.775 |
| baseline_efficientnet_b3 | IDRiD | 455 | 0.497 | 0.760 | 0.441 | 0.497 | 0.830 |
| baseline_resnet50 | APTOS-internal | 733 | 0.789 | 0.875 | 0.616 | 0.630 | 0.924 |
| baseline_resnet50 | Messidor-2 | 1744 | 0.614 | 0.435 | 0.354 | 0.339 | 0.754 |
| baseline_resnet50 | IDRiD | 455 | 0.448 | 0.707 | 0.403 | 0.477 | 0.841 |
| retfound_linprobe | APTOS-internal | 733 | 0.759 | 0.847 | 0.607 | 0.633 | 0.908 |
| retfound_linprobe | Messidor-2 | 1744 | 0.589 | 0.394 | 0.347 | 0.423 | 0.758 |
| retfound_linprobe | IDRiD | 455 | 0.336 | 0.626 | 0.325 | 0.395 | 0.778 |
| retfound_full | APTOS-internal | 733 | 0.783 | 0.882 | 0.632 | 0.647 | 0.900 |
| retfound_full | Messidor-2 | 1744 | 0.624 | 0.487 | 0.368 | 0.350 | 0.750 |
| retfound_full | IDRiD | 455 | 0.451 | 0.744 | 0.419 | 0.511 | 0.819 |

## Internal -> external drop (QWK)

| Model | External | QWK drop |
|---|---|---|
| baseline_efficientnet_b3 | Messidor-2 | 0.377 |
| baseline_efficientnet_b3 | IDRiD | 0.116 |
| baseline_resnet50 | Messidor-2 | 0.440 |
| baseline_resnet50 | IDRiD | 0.167 |
| retfound_linprobe | Messidor-2 | 0.453 |
| retfound_linprobe | IDRiD | 0.221 |
| retfound_full | Messidor-2 | 0.395 |
| retfound_full | IDRiD | 0.139 |
