# Phase 15 Variant C -- input resolution 224 vs 384

Inference-only; RETFound was trained at 224 and is run at 384 with a
bicubically interpolated positional embedding (off-distribution test).

## QWK by model x dataset x resolution

| Model | Dataset | res | QWK | acc | macro-F1 | referable AUC |
|---|---|--:|--:|--:|--:|--:|
| baseline_efficientnet_b3 | APTOS-internal | 224 | 0.876 | 0.803 | 0.644 | 0.980 |
| baseline_efficientnet_b3 | Messidor-2 | 224 | 0.499 | 0.626 | 0.360 | 0.842 |
| baseline_efficientnet_b3 | IDRiD | 224 | 0.760 | 0.497 | 0.441 | 0.962 |
| baseline_efficientnet_b3 | APTOS-internal | 384 | 0.824 | 0.756 | 0.535 | 0.977 |
| baseline_efficientnet_b3 | Messidor-2 | 384 | 0.425 | 0.616 | 0.330 | 0.866 |
| baseline_efficientnet_b3 | IDRiD | 384 | 0.731 | 0.600 | 0.435 | 0.984 |
| retfound_full | APTOS-internal | 224 | 0.869 | 0.769 | 0.616 | 0.978 |
| retfound_full | Messidor-2 | 224 | 0.368 | 0.602 | 0.362 | 0.789 |
| retfound_full | IDRiD | 224 | 0.720 | 0.420 | 0.423 | 0.873 |
| retfound_full | APTOS-internal | 384 | 0.872 | 0.764 | 0.614 | 0.976 |
| retfound_full | Messidor-2 | 384 | 0.393 | 0.607 | 0.373 | 0.803 |
| retfound_full | IDRiD | 384 | 0.683 | 0.396 | 0.394 | 0.854 |

## Internal->external QWK drop, and change vs 224

| Model | External | res | QWK drop | delta vs 224 |
|---|---|--:|--:|--:|
| baseline_efficientnet_b3 | Messidor-2 | 224 | 0.377 | +0.000 |
| baseline_efficientnet_b3 | IDRiD | 224 | 0.116 | +0.000 |
| baseline_efficientnet_b3 | Messidor-2 | 384 | 0.399 | +0.022 |
| baseline_efficientnet_b3 | IDRiD | 384 | 0.093 | -0.024 |
| retfound_full | Messidor-2 | 224 | 0.501 | +0.000 |
| retfound_full | IDRiD | 224 | 0.150 | +0.000 |
| retfound_full | Messidor-2 | 384 | 0.480 | -0.022 |
| retfound_full | IDRiD | 384 | 0.189 | +0.040 |
