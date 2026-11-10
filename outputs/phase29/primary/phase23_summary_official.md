# Phase 23 - primary-outcome inference (official)

Generated 2026-11-06T09:42:48+00:00; B=2000, seed=42, preds=outputs/phase29/preds_official

## Confirmatory family (primary): difference in QWK drop vs EfficientNet-B3

| Model | External | dDrop | 95% CI | Bonferroni 98.75% CI | p (boot) | p (Holm) |
|---|---|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.018 | [-0.024, 0.058] | [-0.036, 0.068] | 0.3938 | 0.6337 |
| RETFound full fine-tune | IDRiD | +0.030 | [-0.026, 0.086] | [-0.041, 0.099] | 0.3168 | 0.6337 |
| RETFound linear probe | Messidor-2 | +0.076 | [0.019, 0.132] | [0.004, 0.146] | 0.0100 | 0.0300 |
| RETFound linear probe | IDRiD | +0.108 | [0.043, 0.169] | [0.027, 0.192] | 0.0040 | 0.0160 |

## Secondary: ResNet-50 vs EfficientNet-B3

| External | dDrop | 95% CI | p (boot) |
|---|---|---|---|
| Messidor-2 | +0.063 | [0.017, 0.109] | 0.0050 |
| IDRiD | +0.063 | [0.013, 0.116] | 0.0170 |

## Paired QWK difference (model - EfficientNet-B3)

| Model | Dataset | QWK model | QWK base | diff | 95% CI |
|---|---|---|---|---|---|
| ResNet-50 | APTOS-internal | 0.875 | 0.876 | -0.001 | [-0.021, 0.019] |
| ResNet-50 | Messidor-2 | 0.435 | 0.499 | -0.064 | [-0.107, -0.023] |
| ResNet-50 | IDRiD | 0.662 | 0.726 | -0.064 | [-0.115, -0.018] |
| RETFound linear probe | APTOS-internal | 0.847 | 0.876 | -0.029 | [-0.054, -0.005] |
| RETFound linear probe | Messidor-2 | 0.394 | 0.499 | -0.105 | [-0.157, -0.053] |
| RETFound linear probe | IDRiD | 0.590 | 0.726 | -0.137 | [-0.197, -0.081] |
| RETFound full fine-tune | APTOS-internal | 0.882 | 0.876 | +0.006 | [-0.014, 0.029] |
| RETFound full fine-tune | Messidor-2 | 0.487 | 0.499 | -0.011 | [-0.047, 0.025] |
| RETFound full fine-tune | IDRiD | 0.703 | 0.726 | -0.024 | [-0.074, 0.026] |

## Worst-case non-independence (m = n/2)

| Model | External | dDrop | 95% CI | excludes 0 |
|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.018 | [-0.042, 0.078] | False |
| RETFound full fine-tune | IDRiD | +0.030 | [-0.048, 0.113] | False |
| RETFound linear probe | Messidor-2 | +0.076 | [-0.004, 0.157] | False |
| RETFound linear probe | IDRiD | +0.108 | [0.018, 0.197] | True |
| ResNet-50 | Messidor-2 | +0.063 | [-0.002, 0.126] | False |
| ResNet-50 | IDRiD | +0.063 | [-0.009, 0.143] | False |

## IDRiD reweighted to the official 516-image grade mix

| Model | QWK mirror | QWK reweighted | dDrop reweighted (95% CI) |
|---|---|---|---|
| EfficientNet-B3 | 0.726 | 0.726 | --- |
| ResNet-50 | 0.662 | 0.662 | +0.063 [0.013, 0.116] |
| RETFound linear probe | 0.590 | 0.590 | +0.108 [0.043, 0.169] |
| RETFound full fine-tune | 0.703 | 0.703 | +0.030 [-0.026, 0.086] |
