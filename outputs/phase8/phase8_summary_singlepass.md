# Phase 8 - Predictive uncertainty (singlepass)

_Generated 2026-09-08T09:56:47+00:00. Source: `outputs\phase6`._

Uncertainty score = predictive entropy of the mean softmax. Mann-Whitney U is one-sided (incorrect > correct); rank-biserial positive = incorrect predictions are more uncertain.

| Model | Dataset | n | Acc | mean H (correct) | mean H (incorrect) | MWU p | rank-biserial | AUROC(err) | Acc@90%cov | err-capture@90% |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 0.626 | 0.193 | 0.418 | 3.43e-37 | +0.363 | 0.681 | 0.654 | 0.167 |
| baseline_efficientnet_b3 | IDRiD | 455 | 0.497 | 0.492 | 0.590 | 3.28e-04 | +0.185 | 0.592 | 0.515 | 0.131 |
| baseline_efficientnet_b3 | APTOS-internal | 733 | 0.804 | 0.189 | 0.579 | 3.68e-37 | +0.681 | 0.841 | 0.847 | 0.299 |
| baseline_resnet50 | Messidor-2 | 1744 | 0.614 | 0.283 | 0.488 | 3.83e-27 | +0.305 | 0.652 | 0.636 | 0.151 |
| baseline_resnet50 | IDRiD | 455 | 0.448 | 0.728 | 0.783 | 4.10e-02 | +0.095 | 0.547 | 0.454 | 0.108 |
| baseline_resnet50 | APTOS-internal | 733 | 0.789 | 0.370 | 0.791 | 1.23e-28 | +0.577 | 0.788 | 0.814 | 0.206 |
| retfound_linprobe | Messidor-2 | 1744 | 0.589 | 1.324 | 1.400 | 2.02e-26 | +0.297 | 0.649 | 0.612 | 0.149 |
| retfound_linprobe | IDRiD | 455 | 0.336 | 1.186 | 1.300 | 1.15e-07 | +0.297 | 0.648 | 0.356 | 0.126 |
| retfound_linprobe | APTOS-internal | 733 | 0.759 | 1.075 | 1.210 | 2.98e-15 | +0.389 | 0.695 | 0.794 | 0.232 |
| retfound_full | Messidor-2 | 1744 | 0.602 | 1.393 | 1.455 | 6.42e-34 | +0.341 | 0.671 | 0.636 | 0.176 |
| retfound_full | IDRiD | 455 | 0.420 | 1.310 | 1.314 | 5.23e-01 | -0.003 | 0.498 | 0.412 | 0.087 |
| retfound_full | APTOS-internal | 733 | 0.769 | 1.084 | 1.165 | 2.41e-07 | +0.255 | 0.628 | 0.803 | 0.231 |

Selective-prediction curve (all coverages): `outputs/phase8/selective_prediction_singlepass.csv`; figure: `outputs/phase8/fig_uncertainty_singlepass.png`.
