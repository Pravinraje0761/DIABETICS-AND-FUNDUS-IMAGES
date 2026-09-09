# Phase 13 -- post-hoc temperature scaling

Calibration mode: **held-out:APTOS-val**  
Bins: 15 equal-width, top-1 confidence  
Eval predictions: `outputs/phase6`

## Fitted temperature (per model)

| Model | T | reading |
|---|--:|---|
| EfficientNet-B3 (baseline) | 1.523 | over-confident -> softened (T>1) |
| ResNet-50 (baseline) | 1.160 | over-confident -> softened (T>1) |
| RETFound (linear probe) | 0.503 | under-confident -> sharpened (T<1) |
| RETFound (full fine-tune) | 0.496 | under-confident -> sharpened (T<1) |

## ECE before -> after, by dataset

| Model | Dataset | n | ECE raw | ECE scaled | Brier raw | Brier scaled | NLL raw | NLL scaled |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| EfficientNet-B3 (baseline) | APTOS-internal | 733 | 0.101 | 0.043 | 0.300 | 0.280 | 0.645 | 0.549 |
| EfficientNet-B3 (baseline) | Messidor-2 | 1744 | 0.279 | 0.213 | 0.644 | 0.594 | 1.742 | 1.277 |
| EfficientNet-B3 (baseline) | IDRiD | 455 | 0.294 | 0.195 | 0.764 | 0.692 | 1.667 | 1.328 |
| ResNet-50 (baseline) | APTOS-internal | 733 | 0.058 | 0.053 | 0.311 | 0.309 | 0.608 | 0.593 |
| ResNet-50 (baseline) | Messidor-2 | 1744 | 0.265 | 0.240 | 0.650 | 0.633 | 1.583 | 1.420 |
| ResNet-50 (baseline) | IDRiD | 455 | 0.260 | 0.223 | 0.778 | 0.749 | 1.445 | 1.357 |
| RETFound (linear probe) | APTOS-internal | 733 | 0.182 | 0.045 | 0.395 | 0.344 | 0.815 | 0.753 |
| RETFound (linear probe) | Messidor-2 | 1744 | 0.121 | 0.100 | 0.594 | 0.591 | 1.181 | 1.190 |
| RETFound (linear probe) | IDRiD | 455 | 0.127 | 0.299 | 0.758 | 0.874 | 1.462 | 1.804 |
| RETFound (full fine-tune) | APTOS-internal | 733 | 0.177 | 0.053 | 0.380 | 0.327 | 0.777 | 0.698 |
| RETFound (full fine-tune) | Messidor-2 | 1744 | 0.160 | 0.081 | 0.594 | 0.569 | 1.185 | 1.122 |
| RETFound (full fine-tune) | IDRiD | 455 | 0.098 | 0.211 | 0.702 | 0.774 | 1.302 | 1.440 |

_Temperature scaling is monotonic in the logits, so accuracy and QWK are
unchanged (verified: `QWK_raw == QWK_scaled`). Only the probability
calibration changes._
