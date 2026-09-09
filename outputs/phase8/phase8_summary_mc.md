# Phase 8 - Predictive uncertainty (mc)

_Generated 2026-09-08T15:32:13+00:00. Source: `outputs\phase8\mc`._

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
