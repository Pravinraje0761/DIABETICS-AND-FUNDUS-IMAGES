# Phase 29b - calibration (official IDRiD) and label-smoothing control

ECE: 15 equal-width bins on max-probability, 95% bootstrap CI (B=500). Error AUROC: grade error.

| Model | Dataset | QWK | ECE [CI] | Brier | mean entropy | mean max-prob | AUROC(err) entropy | AUROC(err) MSP |
|---|---|--:|---|--:|--:|--:|--:|--:|
| EfficientNet-B3 | APTOS-internal | 0.876 | 0.101 [0.081, 0.126] | 0.300 | 0.266 | 0.903 | 0.841 | 0.840 |
| EfficientNet-B3 | Messidor-2 | 0.499 | 0.279 [0.259, 0.304] | 0.644 | 0.277 | 0.905 | 0.681 | 0.682 |
| EfficientNet-B3 | IDRiD | 0.726 | 0.320 [0.277, 0.362] | 0.818 | 0.554 | 0.785 | 0.585 | 0.585 |
| EfficientNet-B3 + label smoothing 0.1 | APTOS-internal | 0.867 | 0.196 [0.177, 0.229] | 0.381 | 1.034 | 0.630 | 0.502 | 0.655 |
| EfficientNet-B3 + label smoothing 0.1 | Messidor-2 | 0.464 | 0.041 [0.029, 0.070] | 0.577 | 1.210 | 0.577 | 0.639 | 0.639 |
| EfficientNet-B3 + label smoothing 0.1 | IDRiD | 0.681 | 0.185 [0.151, 0.232] | 0.778 | 1.081 | 0.579 | 0.452 | 0.467 |
| ResNet-50 | APTOS-internal | 0.875 | 0.058 [0.046, 0.091] | 0.311 | 0.459 | 0.826 | 0.788 | 0.800 |
| ResNet-50 | Messidor-2 | 0.435 | 0.264 [0.247, 0.288] | 0.650 | 0.362 | 0.878 | 0.652 | 0.654 |
| ResNet-50 | IDRiD | 0.662 | 0.282 [0.246, 0.331] | 0.837 | 0.760 | 0.704 | 0.500 | 0.499 |
| RETFound linear probe | APTOS-internal | 0.847 | 0.182 [0.159, 0.209] | 0.395 | 1.108 | 0.583 | 0.695 | 0.761 |
| RETFound linear probe | Messidor-2 | 0.394 | 0.121 [0.102, 0.145] | 0.594 | 1.355 | 0.468 | 0.649 | 0.655 |
| RETFound linear probe | IDRiD | 0.590 | 0.154 [0.115, 0.191] | 0.794 | 1.270 | 0.461 | 0.600 | 0.579 |
| RETFound full fine-tune | APTOS-internal | 0.882 | 0.179 [0.165, 0.211] | 0.369 | 0.972 | 0.656 | 0.516 | 0.635 |
| RETFound full fine-tune | Messidor-2 | 0.487 | 0.095 [0.076, 0.115] | 0.563 | 1.285 | 0.530 | 0.633 | 0.653 |
| RETFound full fine-tune | IDRiD | 0.703 | 0.170 [0.135, 0.215] | 0.762 | 1.104 | 0.552 | 0.503 | 0.518 |

## Difference in internal-to-external QWK drop (paired bootstrap, B=2000)

| External | Comparison | ΔDrop | 95% CI | p |
|---|---|--:|---|--:|
| Messidor-2 | EffB3+LS minus EfficientNet-B3 | +0.026 | [-0.014, +0.067] | 0.176 |
| Messidor-2 | EffB3+LS minus RETFound full fine-tune | +0.008 | [-0.036, +0.054] | 0.702 |
| IDRiD | EffB3+LS minus EfficientNet-B3 | +0.037 | [-0.004, +0.079] | 0.070 |
| IDRiD | EffB3+LS minus RETFound full fine-tune | +0.007 | [-0.053, +0.065] | 0.825 |
