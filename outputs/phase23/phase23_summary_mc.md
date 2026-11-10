# Phase 23 - primary-outcome inference (mc)

Generated 2026-10-21T06:47:29+00:00; B=2000, seed=42, preds=outputs/phase8/mc

## Confirmatory family (primary): difference in QWK drop vs EfficientNet-B3

| Model | External | dDrop | 95% CI | Bonferroni 98.75% CI | p (boot) | p (Holm) |
|---|---|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.019 | [-0.024, 0.062] | [-0.036, 0.073] | 0.3988 | 0.4278 |
| RETFound full fine-tune | IDRiD | +0.035 | [-0.020, 0.092] | [-0.035, 0.107] | 0.2139 | 0.4278 |
| RETFound linear probe | Messidor-2 | +0.078 | [0.018, 0.136] | [0.001, 0.153] | 0.0130 | 0.0520 |
| RETFound linear probe | IDRiD | +0.061 | [-0.004, 0.123] | [-0.025, 0.138] | 0.0670 | 0.2009 |

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
| RETFound linear probe | APTOS-internal | 0.829 | 0.881 | -0.052 | [-0.081, -0.023] |
| RETFound linear probe | Messidor-2 | 0.370 | 0.499 | -0.130 | [-0.180, -0.077] |
| RETFound linear probe | IDRiD | 0.651 | 0.764 | -0.113 | [-0.167, -0.054] |
| RETFound full fine-tune | APTOS-internal | 0.887 | 0.881 | +0.006 | [-0.013, 0.026] |
| RETFound full fine-tune | Messidor-2 | 0.486 | 0.499 | -0.013 | [-0.050, 0.027] |
| RETFound full fine-tune | IDRiD | 0.735 | 0.764 | -0.029 | [-0.082, 0.022] |

## Worst-case non-independence (m = n/2)

| Model | External | dDrop | 95% CI | excludes 0 |
|---|---|---|---|---|
| RETFound full fine-tune | Messidor-2 | +0.019 | [-0.040, 0.079] | False |
| RETFound full fine-tune | IDRiD | +0.035 | [-0.043, 0.115] | False |
| RETFound linear probe | Messidor-2 | +0.078 | [-0.005, 0.163] | False |
| RETFound linear probe | IDRiD | +0.061 | [-0.029, 0.154] | False |
| ResNet-50 | Messidor-2 | +0.057 | [-0.009, 0.126] | False |
| ResNet-50 | IDRiD | +0.047 | [-0.024, 0.117] | False |

## IDRiD reweighted to the official 516-image grade mix

| Model | QWK mirror | QWK reweighted | dDrop reweighted (95% CI) |
|---|---|---|---|
| EfficientNet-B3 | 0.764 | 0.770 | --- |
| ResNet-50 | 0.710 | 0.725 | +0.039 [-0.008, 0.086] |
| RETFound linear probe | 0.651 | 0.648 | +0.070 [0.003, 0.134] |
| RETFound full fine-tune | 0.735 | 0.741 | +0.035 [-0.022, 0.092] |
