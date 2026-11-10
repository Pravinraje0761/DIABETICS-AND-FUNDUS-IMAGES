# Phase 23 - primary-outcome inference (mc_lpdet)

Generated 2026-10-21T07:18:34+00:00; B=2000, seed=42, preds=outputs/phase8/mc_lp_deterministic

## Confirmatory family (primary): difference in QWK drop vs EfficientNet-B3

| Model | External | dDrop | 95% CI | Bonferroni 98.75% CI | p (boot) | p (Holm) |
|---|---|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.019 | [-0.024, 0.062] | [-0.036, 0.073] | 0.3988 | 0.4278 |
| RETFound full fine-tune | IDRiD | +0.035 | [-0.020, 0.092] | [-0.035, 0.107] | 0.2139 | 0.4278 |
| RETFound linear probe | Messidor-2 | +0.071 | [0.015, 0.126] | [0.002, 0.139] | 0.0130 | 0.0390 |
| RETFound linear probe | IDRiD | +0.105 | [0.040, 0.169] | [0.020, 0.195] | 0.0040 | 0.0160 |

## Secondary: ResNet-50 vs EfficientNet-B3

| External | dDrop | 95% CI | p (boot) |
|---|---|---|---|
| Messidor-2 | +0.057 | [0.010, 0.108] | 0.0210 |
| IDRiD | +0.047 | [-0.001, 0.096] | 0.0570 |

## Paired QWK difference (model - EfficientNet-B3)

| Model | Dataset | QWK model | QWK base | diff | 95% CI |
|---|---|---|---|---|---|
| ResNet-50 | APTOS-internal | 0.874 | 0.881 | -0.007 | [-0.028, 0.015] |
| ResNet-50 | Messidor-2 | 0.435 | 0.499 | -0.064 | [-0.108, -0.022] |
| ResNet-50 | IDRiD | 0.710 | 0.764 | -0.054 | [-0.099, -0.007] |
| RETFound linear probe | APTOS-internal | 0.847 | 0.881 | -0.034 | [-0.058, -0.008] |
| RETFound linear probe | Messidor-2 | 0.394 | 0.499 | -0.105 | [-0.156, -0.054] |
| RETFound linear probe | IDRiD | 0.626 | 0.764 | -0.139 | [-0.197, -0.081] |
| RETFound full fine-tune | APTOS-internal | 0.887 | 0.881 | +0.006 | [-0.013, 0.026] |
| RETFound full fine-tune | Messidor-2 | 0.486 | 0.499 | -0.013 | [-0.050, 0.027] |
| RETFound full fine-tune | IDRiD | 0.735 | 0.764 | -0.029 | [-0.082, 0.022] |

## Worst-case non-independence (m = n/2)

| Model | External | dDrop | 95% CI | excludes 0 |
|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.019 | [-0.040, 0.079] | False |
| RETFound full fine-tune | IDRiD | +0.035 | [-0.043, 0.115] | False |
| RETFound linear probe | Messidor-2 | +0.071 | [-0.008, 0.151] | False |
| RETFound linear probe | IDRiD | +0.105 | [0.019, 0.199] | True |
| ResNet-50 | Messidor-2 | +0.057 | [-0.009, 0.126] | False |
| ResNet-50 | IDRiD | +0.047 | [-0.024, 0.117] | False |

## IDRiD reweighted to the official 516-image grade mix

| Model | QWK mirror | QWK reweighted | dDrop reweighted (95% CI) |
|---|---|---|---|
| EfficientNet-B3 | 0.764 | 0.770 | --- |
| ResNet-50 | 0.710 | 0.725 | +0.039 [-0.008, 0.086] |
| RETFound linear probe | 0.626 | 0.619 | +0.117 [0.051, 0.184] |
| RETFound full fine-tune | 0.735 | 0.741 | +0.035 [-0.022, 0.092] |
