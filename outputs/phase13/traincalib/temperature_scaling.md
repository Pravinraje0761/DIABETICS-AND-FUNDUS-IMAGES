# Phase 13 -- post-hoc temperature scaling

Calibration mode: **held-out:APTOS-traincalib**  
Bins: 15 equal-width, top-1 confidence  
Eval predictions: `outputs/phase6`

## Fitted temperature (per model)

| Model | T | reading |
|---|--:|---|
| EfficientNet-B3 (baseline) | 0.788 (bootstrap 95% CI 0.62-0.91) | under-confident -> sharpened (T<1) |
| ResNet-50 (baseline) | 0.824 (bootstrap 95% CI 0.72-0.92) | under-confident -> sharpened (T<1) |
| RETFound (linear probe) | 0.474 (bootstrap 95% CI 0.43-0.51) | under-confident -> sharpened (T<1) |
| RETFound (full fine-tune) | 0.487 (bootstrap 95% CI 0.45-0.53) | under-confident -> sharpened (T<1) |

## ECE before -> after, by dataset

| Model | Dataset | n | ECE raw | ECE scaled | Brier raw | Brier scaled | NLL raw | NLL scaled |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| EfficientNet-B3 (baseline) | APTOS-internal | 733 | 0.101 | 0.124 | 0.300 | 0.314 | 0.645 | 0.757 |
| EfficientNet-B3 (baseline) | Messidor-2 | 1744 | 0.279 | 0.302 | 0.644 | 0.665 | 1.742 | 2.146 |
| EfficientNet-B3 (baseline) | IDRiD | 455 | 0.294 | 0.340 | 0.764 | 0.807 | 1.667 | 1.986 |
| ResNet-50 (baseline) | APTOS-internal | 733 | 0.058 | 0.084 | 0.311 | 0.317 | 0.608 | 0.649 |
| ResNet-50 (baseline) | Messidor-2 | 1744 | 0.265 | 0.291 | 0.650 | 0.670 | 1.583 | 1.852 |
| ResNet-50 (baseline) | IDRiD | 455 | 0.260 | 0.309 | 0.778 | 0.818 | 1.445 | 1.604 |
| RETFound (linear probe) | APTOS-internal | 733 | 0.182 | 0.053 | 0.395 | 0.345 | 0.815 | 0.765 |
| RETFound (linear probe) | Messidor-2 | 1744 | 0.121 | 0.119 | 0.594 | 0.597 | 1.181 | 1.209 |
| RETFound (linear probe) | IDRiD | 455 | 0.127 | 0.311 | 0.758 | 0.889 | 1.462 | 1.859 |
| RETFound (full fine-tune) | APTOS-internal | 733 | 0.177 | 0.056 | 0.380 | 0.327 | 0.777 | 0.701 |
| RETFound (full fine-tune) | Messidor-2 | 1744 | 0.160 | 0.086 | 0.594 | 0.570 | 1.185 | 1.125 |
| RETFound (full fine-tune) | IDRiD | 455 | 0.098 | 0.208 | 0.702 | 0.777 | 1.302 | 1.449 |

_Temperature scaling is monotonic in the logits, so accuracy and QWK are
unchanged (verified: `QWK_raw == QWK_scaled`). Only the probability
calibration changes._
