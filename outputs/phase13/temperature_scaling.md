# Phase 13 -- post-hoc temperature scaling

Calibration mode: **in-sample:APTOS-internal-test**  
Bins: 15 equal-width, top-1 confidence  
Eval predictions: `outputs/phase6`

## Fitted temperature (per model)

| Model | T | reading |
|---|--:|---|
| EfficientNet-B3 (baseline) | 1.711 | over-confident -> softened (T>1) |
| ResNet-50 (baseline) | 1.273 | over-confident -> softened (T>1) |
| RETFound (linear probe) | 0.616 | under-confident -> sharpened (T<1) |
| RETFound (full fine-tune) | 0.596 | under-confident -> sharpened (T<1) |

## ECE before -> after, by dataset

| Model | Dataset | n | ECE raw | ECE scaled | Brier raw | Brier scaled | NLL raw | NLL scaled |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| EfficientNet-B3 (baseline) | APTOS-internal | 733 | 0.101 | 0.048 | 0.300 | 0.278 | 0.645 | 0.544 |
| EfficientNet-B3 (baseline) | Messidor-2 | 1744 | 0.279 | 0.189 | 0.644 | 0.579 | 1.742 | 1.197 |
| EfficientNet-B3 (baseline) | IDRiD | 455 | 0.294 | 0.157 | 0.764 | 0.675 | 1.667 | 1.276 |
| ResNet-50 (baseline) | APTOS-internal | 733 | 0.058 | 0.039 | 0.311 | 0.309 | 0.608 | 0.591 |
| ResNet-50 (baseline) | Messidor-2 | 1744 | 0.265 | 0.223 | 0.650 | 0.621 | 1.583 | 1.336 |
| ResNet-50 (baseline) | IDRiD | 455 | 0.260 | 0.199 | 0.778 | 0.732 | 1.445 | 1.316 |
| RETFound (linear probe) | APTOS-internal | 733 | 0.182 | 0.063 | 0.395 | 0.346 | 0.815 | 0.737 |
| RETFound (linear probe) | Messidor-2 | 1744 | 0.121 | 0.056 | 0.594 | 0.578 | 1.181 | 1.151 |
| RETFound (linear probe) | IDRiD | 455 | 0.127 | 0.246 | 0.758 | 0.830 | 1.462 | 1.651 |
| RETFound (full fine-tune) | APTOS-internal | 733 | 0.177 | 0.053 | 0.380 | 0.326 | 0.777 | 0.683 |
| RETFound (full fine-tune) | Messidor-2 | 1744 | 0.160 | 0.039 | 0.594 | 0.560 | 1.185 | 1.108 |
| RETFound (full fine-tune) | IDRiD | 455 | 0.098 | 0.158 | 0.702 | 0.747 | 1.302 | 1.368 |

_Temperature scaling is monotonic in the logits, so accuracy and QWK are
unchanged (verified: `QWK_raw == QWK_scaled`). Only the probability
calibration changes._
