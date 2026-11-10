# Phase 7 - blinded saliency rating (DEMO / SYNTHETIC)

_Generated 2026-09-10T04:39:52+00:00. Raters: DEMO_rater_synthetic (160 maps with a usable rating)._

> **These numbers come from a fabricated rating sheet** (proxy verdict flipped with p=0.15), used only to exercise the scoring pipeline. Replace with real clinician ratings.

Expert (rater consensus) vs. automated proxy - overall agreement **84.4%**, Cohen's kappa **0.590** (n=160).

| Model | Dataset | Correct? | n | expert-plausible % | proxy-plausible % | agree % | kappa |
|---|---|:--:|--:|--:|--:|--:|--:|
| baseline_efficientnet_b3 | IDRiD | no | 10 | 90% | 100% | 90% | 0.000 |
| baseline_efficientnet_b3 | IDRiD | yes | 10 | 100% | 100% | 100% | nan |
| baseline_efficientnet_b3 | Messidor-2 | no | 10 | 80% | 100% | 80% | 0.000 |
| baseline_efficientnet_b3 | Messidor-2 | yes | 10 | 80% | 90% | 90% | 0.615 |
| baseline_resnet50 | IDRiD | no | 10 | 50% | 90% | 60% | 0.200 |
| baseline_resnet50 | IDRiD | yes | 10 | 90% | 100% | 90% | 0.000 |
| baseline_resnet50 | Messidor-2 | no | 10 | 80% | 100% | 80% | 0.000 |
| baseline_resnet50 | Messidor-2 | yes | 10 | 80% | 90% | 90% | 0.615 |
| retfound_full | IDRiD | no | 10 | 70% | 70% | 100% | 1.000 |
| retfound_full | IDRiD | yes | 10 | 70% | 90% | 80% | 0.412 |
| retfound_full | Messidor-2 | no | 10 | 50% | 30% | 80% | 0.600 |
| retfound_full | Messidor-2 | yes | 10 | 60% | 70% | 90% | 0.783 |
| retfound_linprobe | IDRiD | no | 10 | 60% | 80% | 60% | 0.091 |
| retfound_linprobe | IDRiD | yes | 10 | 60% | 50% | 90% | 0.800 |
| retfound_linprobe | Messidor-2 | no | 10 | 50% | 30% | 80% | 0.600 |
| retfound_linprobe | Messidor-2 | yes | 10 | 60% | 70% | 90% | 0.783 |

Per-cell table: `outputs/phase7/rating_agreement_DEMO.csv`; key: `outputs/phase7/rating_packet/KEY.csv`.
