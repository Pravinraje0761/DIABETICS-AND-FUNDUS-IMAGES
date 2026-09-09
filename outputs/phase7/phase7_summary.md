# Phase 7 - Explainability (automated plausibility proxy)

_Generated 2026-09-08T16:51:42+00:00. 520 saliency maps; CNN = Grad-CAM++, ViT = attention rollout (fallback noted per image)._

Proxy-plausible = peak inside fundus, outside the estimated optic-disc disk and outside the outer border annulus, with >=50% of top-decile saliency mass in that 'lesion zone'. Automated screen, not expert rating (see Limitations).

| Model | Dataset | Correct? | n | proxy-plausible % | top-mass lesion-zone | in disc | in border |
|---|---|:--:|--:|--:|--:|--:|--:|
| baseline_efficientnet_b3 | IDRiD | no | 25 | 100% | 0.98 | 0.02 | 0.00 |
| baseline_efficientnet_b3 | IDRiD | yes | 40 | 100% | 0.98 | 0.01 | 0.01 |
| baseline_efficientnet_b3 | Messidor-2 | no | 25 | 96% | 0.95 | 0.04 | 0.00 |
| baseline_efficientnet_b3 | Messidor-2 | yes | 40 | 92% | 0.94 | 0.04 | 0.02 |
| baseline_resnet50 | IDRiD | no | 25 | 92% | 0.97 | 0.03 | 0.00 |
| baseline_resnet50 | IDRiD | yes | 40 | 98% | 0.95 | 0.04 | 0.01 |
| baseline_resnet50 | Messidor-2 | no | 25 | 100% | 0.98 | 0.02 | 0.00 |
| baseline_resnet50 | Messidor-2 | yes | 40 | 90% | 0.92 | 0.04 | 0.04 |
| retfound_full | IDRiD | no | 25 | 76% | 0.79 | 0.02 | 0.18 |
| retfound_full | IDRiD | yes | 40 | 90% | 0.92 | 0.00 | 0.08 |
| retfound_full | Messidor-2 | no | 25 | 36% | 0.51 | 0.03 | 0.46 |
| retfound_full | Messidor-2 | yes | 40 | 70% | 0.75 | 0.01 | 0.25 |
| retfound_linprobe | IDRiD | no | 25 | 76% | 0.85 | 0.01 | 0.14 |
| retfound_linprobe | IDRiD | yes | 40 | 70% | 0.81 | 0.02 | 0.17 |
| retfound_linprobe | Messidor-2 | no | 25 | 24% | 0.47 | 0.03 | 0.50 |
| retfound_linprobe | Messidor-2 | yes | 40 | 60% | 0.70 | 0.01 | 0.29 |

Saliency method usage: {'gradcampp': 260, 'vit_gradcam': 260}

Per-image detail: `outputs/phase7/saliency_per_image.csv`; overlays: `outputs/phase7/maps/<model>__<dataset>/`.
