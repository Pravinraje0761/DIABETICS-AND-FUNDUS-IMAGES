# Phase 17 -- class-weighting / loss-function robustness

_Generated 2026-09-10T06:33:21+00:00. Variant checkpoints scored unmodified on the Phase-6 code path. `weighted_ce` = the primary model._

## QWK by family x loss x dataset

| Family | Loss | Internal | Messidor-2 | IDRiD |
|---|---|--:|--:|--:|
| efficientnet_b3 | weighted_ce | 0.876 | 0.499 | 0.760 |
| efficientnet_b3 | ce_noweight | 0.881 | 0.549 | 0.706 |
| efficientnet_b3 | focal_g2 | 0.877 | 0.543 | 0.769 |
| retfound_linprobe | weighted_ce | 0.847 | 0.394 | 0.626 |
| retfound_linprobe | ce_noweight | 0.851 | 0.316 | 0.751 |
| retfound_linprobe | focal_g2 | 0.858 | 0.404 | 0.636 |

## Internal->external QWK drop by family x loss

| Family | Loss | Messidor-2 drop | IDRiD drop |
|---|---|--:|--:|
| efficientnet_b3 | weighted_ce | 0.377 | 0.116 |
| efficientnet_b3 | ce_noweight | 0.332 | 0.175 |
| efficientnet_b3 | focal_g2 | 0.334 | 0.108 |
| retfound_linprobe | weighted_ce | 0.453 | 0.221 |
| retfound_linprobe | ce_noweight | 0.535 | 0.100 |
| retfound_linprobe | focal_g2 | 0.454 | 0.222 |

Per-image predictions: `outputs/phase17/<family>__<loss>__<dataset>_preds.csv`.
