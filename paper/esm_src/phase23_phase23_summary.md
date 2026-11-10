# Phase 23 - primary-outcome inference (singlepass)

Generated 2026-10-21T06:16:34+00:00; B=2000, seed=42, preds=outputs/phase6

## Confirmatory family (primary): difference in QWK drop vs EfficientNet-B3

| Model | External | dDrop | 95% CI | Bonferroni 98.75% CI | p (boot) | p (Holm) |
|---|---|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.018 | [-0.024, 0.058] | [-0.036, 0.068] | 0.3938 | 0.7876 |
| RETFound full fine-tune | IDRiD | +0.022 | [-0.033, 0.083] | [-0.051, 0.096] | 0.4548 | 0.7876 |
| RETFound linear probe | Messidor-2 | +0.076 | [0.019, 0.132] | [0.004, 0.146] | 0.0100 | 0.0300 |
| RETFound linear probe | IDRiD | +0.105 | [0.038, 0.170] | [0.021, 0.195] | 0.0040 | 0.0160 |

## Secondary: ResNet-50 vs EfficientNet-B3

| External | dDrop | 95% CI | p (boot) |
|---|---|---|---|
| Messidor-2 | +0.063 | [0.017, 0.109] | 0.0050 |
| IDRiD | +0.051 | [0.003, 0.102] | 0.0410 |

## Paired QWK difference (model - EfficientNet-B3)

| Model | Dataset | QWK model | QWK base | diff | 95% CI |
|---|---|---|---|---|---|
| ResNet-50 | APTOS-internal | 0.875 | 0.876 | -0.001 | [-0.021, 0.019] |
| ResNet-50 | Messidor-2 | 0.435 | 0.499 | -0.064 | [-0.107, -0.023] |
| ResNet-50 | IDRiD | 0.707 | 0.760 | -0.052 | [-0.098, -0.004] |
| RETFound linear probe | APTOS-internal | 0.847 | 0.876 | -0.029 | [-0.054, -0.005] |
| RETFound linear probe | Messidor-2 | 0.394 | 0.499 | -0.105 | [-0.157, -0.053] |
| RETFound linear probe | IDRiD | 0.626 | 0.760 | -0.134 | [-0.192, -0.075] |
| RETFound full fine-tune | APTOS-internal | 0.882 | 0.876 | +0.006 | [-0.014, 0.029] |
| RETFound full fine-tune | Messidor-2 | 0.487 | 0.499 | -0.011 | [-0.047, 0.025] |
| RETFound full fine-tune | IDRiD | 0.744 | 0.760 | -0.016 | [-0.068, 0.038] |

## Worst-case non-independence (m = n/2)

| Model | External | dDrop | 95% CI | excludes 0 |
|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.018 | [-0.042, 0.078] | False |
| RETFound full fine-tune | IDRiD | +0.022 | [-0.056, 0.105] | False |
| RETFound linear probe | Messidor-2 | +0.076 | [-0.004, 0.157] | False |
| RETFound linear probe | IDRiD | +0.105 | [0.020, 0.199] | True |
| ResNet-50 | Messidor-2 | +0.063 | [-0.002, 0.126] | False |
| ResNet-50 | IDRiD | +0.051 | [-0.020, 0.122] | False |

## IDRiD reweighted to the official 516-image grade mix

| Model | QWK mirror | QWK reweighted | dDrop reweighted (95% CI) |
|---|---|---|---|
| EfficientNet-B3 | 0.760 | 0.765 | --- |
| ResNet-50 | 0.707 | 0.722 | +0.042 [-0.006, 0.093] |
| RETFound linear probe | 0.626 | 0.619 | +0.117 [0.049, 0.184] |
| RETFound full fine-tune | 0.744 | 0.750 | +0.022 [-0.034, 0.081] |
