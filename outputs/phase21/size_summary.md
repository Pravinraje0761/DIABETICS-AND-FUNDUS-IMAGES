# Phase 21 -- training-data-size sensitivity

_Generated 2026-09-11T06:05:29+00:00. Stratified subsamples of the APTOS train split (fixed seed = --seed at train time); val/test always use the full split. frac100 = the primary model (no retrain)._

## QWK by family x train-fraction x dataset

| Family | Train % | Internal | Messidor-2 | IDRiD |
|---|--:|--:|--:|--:|
| efficientnet_b3 | 100 | 0.876 | 0.499 | 0.760 |

## Internal->external QWK drop by family x train-fraction

If RETFound needs more labelled data before its advantage over the CNN baseline appears, its drop should shrink faster than the baseline's as train_frac_pct increases.

| Family | Train % | Messidor-2 drop | IDRiD drop |
|---|--:|--:|--:|
| efficientnet_b3 | 100 | 0.377 | 0.116 |

Per-image predictions: `outputs/phase21/<family>__frac<NN>__<dataset>_preds.csv`.
