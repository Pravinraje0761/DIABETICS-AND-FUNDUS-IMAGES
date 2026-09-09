# Phase 15 -- degradation ablation (Variants A + B)

QWK by model x dataset x variant:

| Model | Dataset | variant | QWK | acc | referable AUC |
|---|---|---|--:|--:|--:|
| baseline_efficientnet_b3 | APTOS-internal | baseline | 0.876 | 0.803 | 0.980 |
| baseline_efficientnet_b3 | Messidor-2 | baseline | 0.499 | 0.626 | 0.842 |
| baseline_efficientnet_b3 | IDRiD | baseline | 0.760 | 0.497 | 0.962 |
| baseline_efficientnet_b3 | APTOS-internal | clahe | 0.807 | 0.640 | 0.977 |
| baseline_efficientnet_b3 | Messidor-2 | clahe | 0.458 | 0.606 | 0.792 |
| baseline_efficientnet_b3 | IDRiD | clahe | 0.697 | 0.448 | 0.966 |
| baseline_efficientnet_b3 | APTOS-internal | colour_xfer | 0.816 | 0.704 | 0.979 |
| baseline_efficientnet_b3 | Messidor-2 | colour_xfer | 0.512 | 0.608 | 0.809 |
| baseline_efficientnet_b3 | IDRiD | colour_xfer | 0.689 | 0.464 | 0.960 |
| retfound_full | APTOS-internal | baseline | 0.869 | 0.769 | 0.978 |
| retfound_full | Messidor-2 | baseline | 0.368 | 0.602 | 0.789 |
| retfound_full | IDRiD | baseline | 0.720 | 0.420 | 0.873 |
| retfound_full | APTOS-internal | clahe | 0.599 | 0.555 | 0.966 |
| retfound_full | Messidor-2 | clahe | 0.065 | 0.585 | 0.693 |
| retfound_full | IDRiD | clahe | 0.297 | 0.341 | 0.892 |
| retfound_full | APTOS-internal | colour_xfer | 0.751 | 0.598 | 0.947 |
| retfound_full | Messidor-2 | colour_xfer | 0.122 | 0.591 | 0.738 |
| retfound_full | IDRiD | colour_xfer | 0.545 | 0.415 | 0.938 |

Internal->external QWK drop, and change vs the baseline variant (negative = the variant *recovers* generalisation):

| Model | External | variant | QWK drop | delta vs baseline |
|---|---|---|--:|--:|
| baseline_efficientnet_b3 | Messidor-2 | baseline | 0.377 | +0.000 |
| baseline_efficientnet_b3 | Messidor-2 | clahe | 0.349 | -0.028 |
| baseline_efficientnet_b3 | Messidor-2 | colour_xfer | 0.304 | -0.073 |
| baseline_efficientnet_b3 | IDRiD | baseline | 0.116 | +0.000 |
| baseline_efficientnet_b3 | IDRiD | clahe | 0.110 | -0.006 |
| baseline_efficientnet_b3 | IDRiD | colour_xfer | 0.127 | +0.011 |
| retfound_full | Messidor-2 | baseline | 0.501 | +0.000 |
| retfound_full | Messidor-2 | clahe | 0.534 | +0.033 |
| retfound_full | Messidor-2 | colour_xfer | 0.628 | +0.127 |
| retfound_full | IDRiD | baseline | 0.150 | +0.000 |
| retfound_full | IDRiD | clahe | 0.302 | +0.152 |
| retfound_full | IDRiD | colour_xfer | 0.205 | +0.056 |
