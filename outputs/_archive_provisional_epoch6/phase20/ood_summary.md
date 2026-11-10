# Phase 20 -- domain-shift OOD detection vs. entropy error detection

_Generated 2026-09-11T06:01:00+00:00. OOD score = min-over-class squared Mahalanobis distance in a frozen ImageNet EfficientNet-B3 feature space (class-conditional means + Ledoit-Wolf shared covariance fit on APTOS-train only; same extractor as Phase 14). Paired bootstrap on the difference in error-detection AUROC, B=2000, seed=42._

| Model | Dataset | n | errors | AUROC(OOD) | AUROC(entropy) | diff (OOD-entropy), 95% CI | verdict |
|---|---|--:|--:|--:|--:|---|---|
| baseline_efficientnet_b3 | APTOS-internal | 733 | 144 | 0.652 | 0.841 | -0.189 (-0.243,-0.136) | entropy better |
| baseline_efficientnet_b3 | Messidor-2 | 1744 | 653 | 0.472 | 0.681 | -0.210 (-0.244,-0.176) | entropy better |
| baseline_efficientnet_b3 | IDRiD | 455 | 229 | 0.486 | 0.592 | -0.106 (-0.174,-0.032) | entropy better |
| baseline_resnet50 | APTOS-internal | 733 | 155 | 0.644 | 0.788 | -0.145 (-0.197,-0.091) | entropy better |
| baseline_resnet50 | Messidor-2 | 1744 | 674 | 0.474 | 0.652 | -0.178 (-0.209,-0.146) | entropy better |
| baseline_resnet50 | IDRiD | 455 | 251 | 0.506 | 0.547 | -0.041 (-0.106,+0.018) | no sig. difference |
| retfound_linprobe | APTOS-internal | 733 | 177 | 0.663 | 0.695 | -0.032 (-0.101,+0.030) | no sig. difference |
| retfound_linprobe | Messidor-2 | 1744 | 716 | 0.479 | 0.649 | -0.170 (-0.206,-0.135) | entropy better |
| retfound_linprobe | IDRiD | 455 | 302 | 0.427 | 0.648 | -0.221 (-0.322,-0.123) | entropy better |
| retfound_full | APTOS-internal | 733 | 169 | 0.662 | 0.627 | +0.035 (-0.038,+0.100) | no sig. difference |
| retfound_full | Messidor-2 | 1744 | 694 | 0.447 | 0.671 | -0.223 (-0.255,-0.193) | entropy better |
| retfound_full | IDRiD | 455 | 264 | 0.498 | 0.498 | -0.001 (-0.090,+0.092) | no sig. difference |

APTOS-internal rows are a control: the OOD detector is fit on this same distribution, so a large advantage there (if any) reflects in-domain behaviour, not domain-shift detection.
