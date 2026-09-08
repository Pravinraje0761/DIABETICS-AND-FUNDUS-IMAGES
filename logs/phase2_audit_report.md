# Phase 2 - Data Acquisition Audit

_Generated: 2026-09-08T06:23:00+00:00_

## 1. File-level consistency

| Dataset | Label rows | Image files | Labels w/o image | Images w/o label | Dup label IDs |
|---|---|---|---|---|---|
| APTOS 2019 (train.csv) | 3662 | 3662 | 0 | 0 | 0 |
| Messidor-2 | 1744 | 1744 | 0 | 0 | 0 |
| IDRiD (Disease Grading) | 455 | 455 | 0 | 0 | 0 |

> APTOS Kaggle holdout `test_images/`: 1928 images, has_labels=False. Kaggle competition holdout - NO public labels. Not usable as an internal test set. Internal test is carved from train.csv in Phase 3.

## 2. DR-grade class distribution

| Dataset | Subset | 0 No DR | 1 Mild | 2 Moderate | 3 Severe | 4 Prolif. | n | Majority | Max/Min ratio |
|---|---|---|---|---|---|---|---|---|---|
| APTOS 2019 | train.csv (all labeled) | 1805 (49.29%) | 370 (10.1%) | 999 (27.28%) | 193 (5.27%) | 295 (8.06%) | 3662 | 0 (No DR) | 9.35 |
| Messidor-2 | all | 1017 (58.31%) | 270 (15.48%) | 347 (19.9%) | 75 (4.3%) | 35 (2.01%) | 1744 | 0 (No DR) | 29.06 |
| Messidor-2 | gradable only | 1017 (58.31%) | 270 (15.48%) | 347 (19.9%) | 75 (4.3%) | 35 (2.01%) | 1744 | 0 (No DR) | 29.06 |
| IDRiD | all | 129 (28.35%) | 22 (4.84%) | 156 (34.29%) | 84 (18.46%) | 64 (14.07%) | 455 | 2 (Moderate) | 7.09 |

## 3. Notes carried into later phases

- **Metric choice (Phase 4+):** all three datasets are class-imbalanced (APTOS/Messidor-2 dominated by grade 0; IDRiD's modal class is grade 2), and imbalanced differently. Use quadratic-weighted kappa and per-class sensitivity/specificity as primary metrics; raw accuracy is misleading here.
- **Messidor-2:** Recommend restricting the external test to adjudicated_gradable == 1. (0 ungradable rows present.)
- **IDRiD:** Official IDRiD Disease Grading subset is 516 images (413 train + 103 test). This mirror provides 455 images / 455 labels - confirm completeness against the source before the Data Availability statement.
- **Split isolation:** APTOS 2019 -> train/val/test only. Messidor-2 and IDRiD -> external test only, never seen in training or hyperparameter tuning.
