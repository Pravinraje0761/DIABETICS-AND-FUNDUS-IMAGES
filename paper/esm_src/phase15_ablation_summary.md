# Phase 15 -- degradation ablation (Variants A + B)

QWK by model x dataset x variant:

| Model | Dataset | variant | QWK | acc | referable AUC |
|---|---|---|--:|--:|--:|
| baseline_efficientnet_b3 | APTOS-internal | baseline | 0.876 | 0.803 | 0.980 |
| baseline_efficientnet_b3 | Messidor-2 | baseline | 0.499 | 0.626 | 0.842 |
| baseline_efficientnet_b3 | IDRiD | baseline | 0.760 | 0.497 | 0.962 |
| baseline_efficientnet_b3 | APTOS-internal | clahe | 0.807 | 0.641 | 0.977 |
| baseline_efficientnet_b3 | Messidor-2 | clahe | 0.458 | 0.606 | 0.792 |
| baseline_efficientnet_b3 | IDRiD | clahe | 0.697 | 0.448 | 0.966 |
| baseline_efficientnet_b3 | APTOS-internal | colour_xfer | 0.816 | 0.704 | 0.979 |
| baseline_efficientnet_b3 | Messidor-2 | colour_xfer | 0.512 | 0.608 | 0.809 |
| baseline_efficientnet_b3 | IDRiD | colour_xfer | 0.689 | 0.464 | 0.960 |
| retfound_full | APTOS-internal | baseline | 0.882 | 0.783 | 0.967 |
| retfound_full | Messidor-2 | baseline | 0.487 | 0.624 | 0.817 |
| retfound_full | IDRiD | baseline | 0.744 | 0.451 | 0.885 |
| retfound_full | APTOS-internal | clahe | 0.703 | 0.570 | 0.966 |
| retfound_full | Messidor-2 | clahe | 0.096 | 0.588 | 0.733 |
| retfound_full | IDRiD | clahe | 0.410 | 0.380 | 0.938 |
| retfound_full | APTOS-internal | colour_xfer | 0.810 | 0.689 | 0.944 |
| retfound_full | Messidor-2 | colour_xfer | 0.284 | 0.608 | 0.742 |
| retfound_full | IDRiD | colour_xfer | 0.693 | 0.505 | 0.936 |

Internal->external QWK drop, and change vs the baseline variant (negative = the variant *recovers* generalisation):

| Model | External | variant | QWK drop | delta vs baseline |
|---|---|---|--:|--:|
| baseline_efficientnet_b3 | Messidor-2 | baseline | 0.377 | +0.000 |
| baseline_efficientnet_b3 | Messidor-2 | clahe | 0.350 | -0.028 |
| baseline_efficientnet_b3 | Messidor-2 | colour_xfer | 0.304 | -0.073 |
| baseline_efficientnet_b3 | IDRiD | baseline | 0.116 | +0.000 |
| baseline_efficientnet_b3 | IDRiD | clahe | 0.111 | -0.006 |
| baseline_efficientnet_b3 | IDRiD | colour_xfer | 0.127 | +0.011 |
| retfound_full | Messidor-2 | baseline | 0.395 | +0.000 |
| retfound_full | Messidor-2 | clahe | 0.606 | +0.211 |
| retfound_full | Messidor-2 | colour_xfer | 0.526 | +0.131 |
| retfound_full | IDRiD | baseline | 0.139 | +0.000 |
| retfound_full | IDRiD | clahe | 0.293 | +0.154 |
| retfound_full | IDRiD | colour_xfer | 0.118 | -0.021 |
