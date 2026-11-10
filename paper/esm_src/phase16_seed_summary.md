# Phase 16 -- training-seed sensitivity

_Generated 2026-09-09T14:21:05+00:00_

Extra full-fine-tune runs scored unmodified on the Phase-6 code path. Primary model = seed 42.

## QWK across seeds (mean +/- SD [min, max])

| Family | Dataset | seeds | QWK mean | SD | min | max | range |
|---|---|--:|--:|--:|--:|--:|--:|
| efficientnet_b3 | APTOS-internal | 4 | 0.879 | 0.003 | 0.876 | 0.883 | 0.007 |
| efficientnet_b3 | Messidor-2 | 4 | 0.500 | 0.024 | 0.469 | 0.527 | 0.058 |
| efficientnet_b3 | IDRiD | 4 | 0.749 | 0.015 | 0.732 | 0.764 | 0.033 |
| retfound_linprobe | APTOS-internal | 4 | 0.855 | 0.008 | 0.847 | 0.865 | 0.018 |
| retfound_linprobe | Messidor-2 | 4 | 0.395 | 0.015 | 0.374 | 0.408 | 0.034 |
| retfound_linprobe | IDRiD | 4 | 0.623 | 0.012 | 0.605 | 0.631 | 0.026 |

## Internal->external QWK drop across seeds

| Family | External | seeds | drop mean | SD | min | max |
|---|---|--:|--:|--:|--:|--:|
| efficientnet_b3 | Messidor-2 | 4 | 0.379 | 0.025 | 0.353 | 0.413 |
| efficientnet_b3 | IDRiD | 4 | 0.130 | 0.015 | 0.116 | 0.148 |
| retfound_linprobe | Messidor-2 | 4 | 0.461 | 0.017 | 0.444 | 0.484 |
| retfound_linprobe | IDRiD | 4 | 0.233 | 0.019 | 0.221 | 0.260 |

Per-seed detail: `seed_master.csv`, `seed_drop.csv`.
