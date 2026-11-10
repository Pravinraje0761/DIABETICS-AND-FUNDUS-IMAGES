# Phase 8 - Predictive uncertainty (singlepass)

_Generated 2026-10-21T06:20:56+00:00. Source: `outputs\phase6`._

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
| retfound_full | Messidor-2 | 1744 | 0.624 | 1.259 | 1.326 | 6.26e-21 | +0.266 | 0.633 | 0.652 | 0.165 |
| retfound_full | IDRiD | 455 | 0.451 | 1.104 | 1.095 | 6.34e-01 | -0.019 | 0.491 | 0.459 | 0.112 |
| retfound_full | APTOS-internal | 733 | 0.783 | 0.972 | 0.970 | 2.64e-01 | +0.033 | 0.516 | 0.808 | 0.201 |

Selective-prediction curve (all coverages): `outputs/phase8/selective_prediction_singlepass.csv`; figure: `outputs/phase8/fig_uncertainty_singlepass.png`.

## Per-class predictive uncertainty

Mean predictive entropy (nats) and accuracy per ICDR grade on the external sets. `H` = mean entropy; accuracy in brackets; grouped by **true** grade. Full breakdown (incl. grouping by predicted grade, median entropy, correct/incorrect split): `outputs/phase8/uncertainty_by_class_singlepass.csv`; figure: `outputs/phase8/fig_uncertainty_byclass_singlepass.png`.

### Messidor-2 (by true grade)

| Model | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| baseline_efficientnet_b3 | H=0.151 (0.97, n=1017) | H=0.176 (0.01, n=270) | H=0.552 (0.22, n=347) | H=0.827 (0.13, n=75) | H=0.820 (0.43, n=35) |
| baseline_resnet50 | H=0.247 (0.95, n=1017) | H=0.261 (0.01, n=270) | H=0.601 (0.20, n=347) | H=0.900 (0.28, n=75) | H=0.978 (0.26, n=35) |
| retfound_linprobe | H=1.337 (0.90, n=1017) | H=1.356 (0.00, n=270) | H=1.433 (0.18, n=347) | H=1.361 (0.41, n=75) | H=1.106 (0.63, n=35) |
| retfound_full | H=1.262 (0.97, n=1017) | H=1.274 (0.00, n=270) | H=1.380 (0.22, n=347) | H=1.228 (0.16, n=75) | H=1.198 (0.40, n=35) |

### IDRiD (by true grade)

| Model | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| baseline_efficientnet_b3 | H=0.476 (0.49, n=129) | H=0.501 (0.77, n=22) | H=0.522 (0.68, n=156) | H=0.661 (0.25, n=84) | H=0.578 (0.30, n=64) |
| baseline_resnet50 | H=0.564 (0.50, n=129) | H=0.546 (0.86, n=22) | H=0.826 (0.52, n=156) | H=0.907 (0.38, n=84) | H=0.864 (0.12, n=64) |
| retfound_linprobe | H=1.398 (0.22, n=129) | H=1.380 (0.45, n=22) | H=1.310 (0.18, n=156) | H=1.145 (0.75, n=84) | H=0.983 (0.38, n=64) |
| retfound_full | H=1.268 (0.51, n=129) | H=1.261 (0.86, n=22) | H=1.109 (0.33, n=156) | H=0.964 (0.69, n=84) | H=0.858 (0.16, n=64) |
