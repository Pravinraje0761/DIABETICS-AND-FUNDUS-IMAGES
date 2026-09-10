# Phase 8 - Predictive uncertainty (mc)

_Generated 2026-09-10T04:42:30+00:00. Source: `outputs\phase8\mc`._

Uncertainty score = predictive entropy of the mean softmax. Mann-Whitney U is one-sided (incorrect > correct); rank-biserial positive = incorrect predictions are more uncertain.

| Model | Dataset | n | Acc | mean H (correct) | mean H (incorrect) | MWU p | rank-biserial | AUROC(err) | Acc@90%cov | err-capture@90% |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 0.626 | 0.251 | 0.494 | 1.86e-38 | +0.369 | 0.685 | 0.652 | 0.162 |
| baseline_efficientnet_b3 | IDRiD | 455 | 0.508 | 0.614 | 0.732 | 3.18e-05 | +0.217 | 0.608 | 0.527 | 0.134 |
| baseline_efficientnet_b3 | APTOS-internal | 733 | 0.809 | 0.271 | 0.710 | 2.29e-36 | +0.681 | 0.840 | 0.852 | 0.300 |
| baseline_resnet50 | Messidor-2 | 1744 | 0.614 | 0.283 | 0.488 | 6.91e-27 | +0.303 | 0.652 | 0.636 | 0.152 |
| baseline_resnet50 | IDRiD | 455 | 0.451 | 0.733 | 0.783 | 5.75e-02 | +0.086 | 0.543 | 0.456 | 0.108 |
| baseline_resnet50 | APTOS-internal | 733 | 0.787 | 0.371 | 0.794 | 7.10e-29 | +0.578 | 0.789 | 0.811 | 0.199 |
| retfound_linprobe | Messidor-2 | 1744 | 0.577 | 1.394 | 1.452 | 2.58e-24 | +0.283 | 0.641 | 0.594 | 0.137 |
| retfound_linprobe | IDRiD | 455 | 0.369 | 1.265 | 1.375 | 1.97e-08 | +0.308 | 0.654 | 0.388 | 0.125 |
| retfound_linprobe | APTOS-internal | 733 | 0.749 | 1.196 | 1.303 | 2.33e-14 | +0.371 | 0.686 | 0.785 | 0.228 |
| retfound_full | Messidor-2 | 1744 | 0.594 | 1.450 | 1.498 | 1.01e-33 | +0.339 | 0.670 | 0.622 | 0.161 |
| retfound_full | IDRiD | 455 | 0.387 | 1.374 | 1.387 | 3.71e-01 | +0.018 | 0.509 | 0.376 | 0.082 |
| retfound_full | APTOS-internal | 733 | 0.775 | 1.199 | 1.270 | 1.05e-07 | +0.265 | 0.633 | 0.800 | 0.200 |

Selective-prediction curve (all coverages): `outputs/phase8/selective_prediction_mc.csv`; figure: `outputs/phase8/fig_uncertainty_mc.png`.

## Per-class predictive uncertainty

Mean predictive entropy (nats) and accuracy per ICDR grade on the external sets. `H` = mean entropy; accuracy in brackets; grouped by **true** grade. Full breakdown (incl. grouping by predicted grade, median entropy, correct/incorrect split): `outputs/phase8/uncertainty_by_class_mc.csv`; figure: `outputs/phase8/fig_uncertainty_byclass_mc.png`.

### Messidor-2 (by true grade)

| Model | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| baseline_efficientnet_b3 | H=0.201 (0.97, n=1017) | H=0.239 (0.01, n=270) | H=0.645 (0.21, n=347) | H=0.959 (0.09, n=75) | H=0.914 (0.51, n=35) |
| baseline_resnet50 | H=0.247 (0.95, n=1017) | H=0.260 (0.01, n=270) | H=0.600 (0.20, n=347) | H=0.902 (0.28, n=75) | H=0.981 (0.29, n=35) |
| retfound_linprobe | H=1.410 (0.88, n=1017) | H=1.425 (0.00, n=270) | H=1.475 (0.17, n=347) | H=1.387 (0.40, n=75) | H=1.139 (0.69, n=35) |
| retfound_full | H=1.452 (0.95, n=1017) | H=1.463 (0.03, n=270) | H=1.513 (0.07, n=347) | H=1.525 (0.33, n=75) | H=1.494 (0.43, n=35) |

### IDRiD (by true grade)

| Model | No DR | Mild | Moderate | Severe | Proliferative |
|---|---|---|---|---|---|
| baseline_efficientnet_b3 | H=0.579 (0.50, n=129) | H=0.689 (0.77, n=22) | H=0.680 (0.68, n=156) | H=0.782 (0.27, n=84) | H=0.691 (0.31, n=64) |
| baseline_resnet50 | H=0.565 (0.50, n=129) | H=0.550 (0.86, n=22) | H=0.827 (0.53, n=156) | H=0.909 (0.38, n=84) | H=0.868 (0.12, n=64) |
| retfound_linprobe | H=1.454 (0.29, n=129) | H=1.459 (0.45, n=22) | H=1.390 (0.19, n=156) | H=1.234 (0.71, n=84) | H=1.049 (0.48, n=64) |
| retfound_full | H=1.462 (0.41, n=129) | H=1.450 (0.86, n=22) | H=1.403 (0.23, n=156) | H=1.339 (0.60, n=84) | H=1.198 (0.28, n=64) |
