# Phase 26 - saliency localisation vs. IDRiD expert lesion masks

_Generated 2026-11-06T02:36:00+00:00; n = 81 images (IDRiD Segmentation subset), pointing tolerance 4.0 px at 224, bootstrap B=2000._

Lift = share of saliency mass on lesions / share of the field of view that is lesion (uniform map = 1). Cross-image = the model's map for a different image.

| Map | energy on lesions | lift (any) | pointing hit | pixel AUROC | top-10% precision | lift MA | lift HE | lift EX | lift SE | energy on OD |
|---|---|---|---|---|---|---|---|---|---|---|
| Centre-Gaussian control | 0.09 [0.07, 0.12] | 1.28 [1.19, 1.38] | 0.36 [0.25, 0.47] | 0.59 [0.56, 0.61] | 0.11 [0.08, 0.14] | 1.23 | 0.90 | 1.80 | 0.70 | 0.03 |
| EfficientNet-B3 (cross-image) | 0.08 [0.06, 0.11] | 1.17 [1.08, 1.25] | 0.31 [0.21, 0.41] | 0.55 [0.53, 0.58] | 0.09 [0.06, 0.12] | 1.18 | 0.91 | 1.49 | 0.70 | 0.03 |
| ResNet-50 (cross-image) | 0.08 [0.06, 0.11] | 1.15 [1.08, 1.22] | 0.33 [0.23, 0.44] | 0.56 [0.53, 0.58] | 0.09 [0.07, 0.12] | 1.18 | 0.88 | 1.46 | 0.80 | 0.03 |
| RETFound full fine-tune (cross-image) | 0.07 [0.05, 0.10] | 1.02 [0.94, 1.11] | 0.25 [0.16, 0.35] | 0.51 [0.49, 0.53] | 0.08 [0.05, 0.11] | 1.08 | 0.91 | 1.13 | 0.81 | 0.04 |
| RETFound linear probe (cross-image) | 0.08 [0.06, 0.11] | 1.12 [1.04, 1.21] | 0.36 [0.26, 0.47] | 0.55 [0.52, 0.57] | 0.09 [0.06, 0.12] | 1.10 | 1.03 | 1.30 | 0.83 | 0.03 |
| EfficientNet-B3 | 0.09 [0.07, 0.13] | 1.37 [1.28, 1.48] | 0.48 [0.37, 0.59] | 0.61 [0.58, 0.63] | 0.12 [0.09, 0.15] | 1.21 | 1.03 | 1.73 | 0.87 | 0.03 |
| ResNet-50 | 0.09 [0.07, 0.12] | 1.32 [1.25, 1.41] | 0.49 [0.38, 0.60] | 0.61 [0.59, 0.64] | 0.12 [0.09, 0.15] | 1.21 | 1.02 | 1.61 | 0.95 | 0.03 |
| RETFound full fine-tune | 0.12 [0.09, 0.15] | 1.79 [1.63, 1.95] | 0.64 [0.53, 0.74] | 0.67 [0.63, 0.70] | 0.19 [0.15, 0.23] | 1.24 | 1.46 | 1.83 | 2.24 | 0.02 |
| RETFound linear probe | 0.12 [0.09, 0.16] | 1.67 [1.54, 1.82] | 0.60 [0.51, 0.72] | 0.67 [0.64, 0.70] | 0.19 [0.15, 0.23] | 1.20 | 1.46 | 1.74 | 1.77 | 0.02 |

Mean lesion share of the field of view: 0.071.

## Paired comparisons (Wilcoxon signed-rank over images)

| Model | vs | metric | model | comparator | Δ | images better | p |
|---|---|---|--:|--:|--:|--:|--:|
| EfficientNet-B3 | centre | lift_any | 1.372 | 1.283 | +0.090 | 0.57 | 0.099 |
| EfficientNet-B3 | centre | auroc_lesion | 0.606 | 0.587 | +0.020 | 0.55 | 0.082 |
| EfficientNet-B3 | centre | energy_any | 0.094 | 0.090 | +0.004 | 0.57 | 0.068 |
| EfficientNet-B3 | centre | top10_precision | 0.121 | 0.107 | +0.013 | 0.53 | 0.18 |
| EfficientNet-B3 | centre | energy_od | 0.026 | 0.027 | -0.001 | 0.42 | 0.62 |
| EfficientNet-B3 | cross_image | lift_any | 1.372 | 1.165 | +0.207 | 0.77 | 1.6e-08 |
| EfficientNet-B3 | cross_image | auroc_lesion | 0.606 | 0.555 | +0.051 | 0.75 | 2.3e-08 |
| EfficientNet-B3 | cross_image | energy_any | 0.094 | 0.082 | +0.012 | 0.77 | 1.2e-08 |
| EfficientNet-B3 | cross_image | top10_precision | 0.121 | 0.091 | +0.030 | 0.72 | 1.2e-06 |
| EfficientNet-B3 | cross_image | energy_od | 0.026 | 0.032 | -0.006 | 0.38 | 0.0088 |
| ResNet-50 | centre | lift_any | 1.324 | 1.283 | +0.041 | 0.54 | 0.45 |
| ResNet-50 | centre | auroc_lesion | 0.612 | 0.587 | +0.025 | 0.62 | 0.0078 |
| ResNet-50 | centre | energy_any | 0.090 | 0.090 | -0.001 | 0.54 | 0.75 |
| ResNet-50 | centre | top10_precision | 0.119 | 0.107 | +0.012 | 0.60 | 0.024 |
| ResNet-50 | centre | energy_od | 0.032 | 0.027 | +0.004 | 0.57 | 0.023 |
| ResNet-50 | cross_image | lift_any | 1.324 | 1.147 | +0.177 | 0.72 | 2.1e-07 |
| ResNet-50 | cross_image | auroc_lesion | 0.612 | 0.555 | +0.057 | 0.82 | 4.8e-10 |
| ResNet-50 | cross_image | energy_any | 0.090 | 0.080 | +0.010 | 0.72 | 6.9e-07 |
| ResNet-50 | cross_image | top10_precision | 0.119 | 0.091 | +0.028 | 0.56 | 0.0033 |
| ResNet-50 | cross_image | energy_od | 0.032 | 0.031 | +0.001 | 0.51 | 0.6 |
| ResNet-50 | EfficientNet-B3 | lift_any | 1.324 | 1.372 | -0.049 | 0.42 | 0.17 |
| ResNet-50 | EfficientNet-B3 | auroc_lesion | 0.612 | 0.606 | +0.006 | 0.51 | 0.23 |
| ResNet-50 | EfficientNet-B3 | energy_any | 0.090 | 0.094 | -0.004 | 0.42 | 0.048 |
| ResNet-50 | EfficientNet-B3 | top10_precision | 0.119 | 0.121 | -0.002 | 0.53 | 0.8 |
| ResNet-50 | EfficientNet-B3 | energy_od | 0.032 | 0.026 | +0.005 | 0.77 | 1.4e-05 |
| RETFound linear probe | centre | lift_any | 1.672 | 1.283 | +0.389 | 0.73 | 3e-06 |
| RETFound linear probe | centre | auroc_lesion | 0.670 | 0.587 | +0.083 | 0.72 | 1.1e-05 |
| RETFound linear probe | centre | energy_any | 0.121 | 0.090 | +0.031 | 0.73 | 1.1e-07 |
| RETFound linear probe | centre | top10_precision | 0.187 | 0.107 | +0.079 | 0.72 | 8.9e-08 |
| RETFound linear probe | centre | energy_od | 0.016 | 0.027 | -0.011 | 0.30 | 3.7e-06 |
| RETFound linear probe | cross_image | lift_any | 1.672 | 1.123 | +0.548 | 0.79 | 2.9e-09 |
| RETFound linear probe | cross_image | auroc_lesion | 0.670 | 0.547 | +0.123 | 0.80 | 2.2e-08 |
| RETFound linear probe | cross_image | energy_any | 0.121 | 0.079 | +0.042 | 0.79 | 3.8e-09 |
| RETFound linear probe | cross_image | top10_precision | 0.187 | 0.089 | +0.098 | 0.80 | 2.7e-09 |
| RETFound linear probe | cross_image | energy_od | 0.016 | 0.031 | -0.015 | 0.30 | 1.8e-06 |
| RETFound linear probe | EfficientNet-B3 | lift_any | 1.672 | 1.372 | +0.299 | 0.72 | 0.00015 |
| RETFound linear probe | EfficientNet-B3 | auroc_lesion | 0.670 | 0.606 | +0.063 | 0.71 | 0.00022 |
| RETFound linear probe | EfficientNet-B3 | energy_any | 0.121 | 0.094 | +0.027 | 0.72 | 2.9e-06 |
| RETFound linear probe | EfficientNet-B3 | top10_precision | 0.187 | 0.121 | +0.066 | 0.73 | 2.2e-06 |
| RETFound linear probe | EfficientNet-B3 | energy_od | 0.016 | 0.026 | -0.010 | 0.33 | 4.9e-05 |
| RETFound full fine-tune | centre | lift_any | 1.788 | 1.283 | +0.505 | 0.69 | 7e-06 |
| RETFound full fine-tune | centre | auroc_lesion | 0.668 | 0.587 | +0.082 | 0.68 | 0.00014 |
| RETFound full fine-tune | centre | energy_any | 0.121 | 0.090 | +0.031 | 0.69 | 1.1e-05 |
| RETFound full fine-tune | centre | top10_precision | 0.186 | 0.107 | +0.079 | 0.72 | 3.3e-06 |
| RETFound full fine-tune | centre | energy_od | 0.019 | 0.027 | -0.008 | 0.36 | 0.00073 |
| RETFound full fine-tune | cross_image | lift_any | 1.788 | 1.022 | +0.766 | 0.85 | 1.1e-10 |
| RETFound full fine-tune | cross_image | auroc_lesion | 0.668 | 0.509 | +0.159 | 0.82 | 9.1e-10 |
| RETFound full fine-tune | cross_image | energy_any | 0.121 | 0.073 | +0.049 | 0.85 | 2.5e-11 |
| RETFound full fine-tune | cross_image | top10_precision | 0.186 | 0.078 | +0.108 | 0.83 | 1.7e-11 |
| RETFound full fine-tune | cross_image | energy_od | 0.019 | 0.036 | -0.017 | 0.26 | 3.3e-06 |
| RETFound full fine-tune | EfficientNet-B3 | lift_any | 1.788 | 1.372 | +0.416 | 0.65 | 0.00012 |
| RETFound full fine-tune | EfficientNet-B3 | auroc_lesion | 0.668 | 0.606 | +0.062 | 0.66 | 0.0008 |
| RETFound full fine-tune | EfficientNet-B3 | energy_any | 0.121 | 0.094 | +0.027 | 0.65 | 5.5e-05 |
| RETFound full fine-tune | EfficientNet-B3 | top10_precision | 0.186 | 0.121 | +0.065 | 0.69 | 2.8e-05 |
| RETFound full fine-tune | EfficientNet-B3 | energy_od | 0.019 | 0.026 | -0.007 | 0.40 | 0.0064 |

## Resolution-matched control (Phase 26b)

All maps degraded to the CNN's native 7x7 grid (average-pool, bilinear back to 224).

| Model | resolution | lift (any) | pixel AUROC | pointing hit | top-10% precision |
|---|---|---|---|---|---|
| EfficientNet-B3 | 7x7 | 1.32 [1.24, 1.41] | 0.61 [0.58, 0.63] | 0.47 [0.36, 0.58] | 0.12 [0.09, 0.15] |
| EfficientNet-B3 | native | 1.37 [1.28, 1.48] | 0.61 [0.58, 0.63] | 0.48 [0.37, 0.59] | 0.12 [0.09, 0.15] |
| ResNet-50 | 7x7 | 1.27 [1.20, 1.34] | 0.61 [0.58, 0.63] | 0.46 [0.36, 0.57] | 0.12 [0.09, 0.15] |
| ResNet-50 | native | 1.32 [1.25, 1.41] | 0.61 [0.59, 0.64] | 0.49 [0.38, 0.60] | 0.12 [0.09, 0.15] |
| RETFound full fine-tune | 7x7 | 1.44 [1.34, 1.54] | 0.66 [0.63, 0.69] | 0.54 [0.43, 0.65] | 0.16 [0.12, 0.20] |
| RETFound full fine-tune | native | 1.79 [1.63, 1.95] | 0.67 [0.63, 0.70] | 0.64 [0.53, 0.74] | 0.19 [0.15, 0.23] |
| RETFound linear probe | 7x7 | 1.41 [1.33, 1.50] | 0.65 [0.62, 0.68] | 0.59 [0.49, 0.70] | 0.16 [0.13, 0.20] |
| RETFound linear probe | native | 1.67 [1.54, 1.82] | 0.67 [0.64, 0.70] | 0.60 [0.51, 0.72] | 0.19 [0.15, 0.23] |

| RETFound (7x7) vs EfficientNet-B3 (native) | metric | RETFound | EffB3 | Δ | images better | p |
|---|---|--:|--:|--:|--:|--:|
| RETFound full fine-tune | lift_any | 1.440 | 1.372 | +0.067 | 0.58 | 0.17 |
| RETFound full fine-tune | auroc_lesion | 0.661 | 0.606 | +0.055 | 0.68 | 0.00041 |
| RETFound full fine-tune | top10_precision | 0.157 | 0.121 | +0.036 | 0.62 | 0.0029 |
| RETFound linear probe | lift_any | 1.409 | 1.372 | +0.037 | 0.58 | 0.13 |
| RETFound linear probe | auroc_lesion | 0.650 | 0.606 | +0.044 | 0.65 | 0.00095 |
| RETFound linear probe | top10_precision | 0.160 | 0.121 | +0.039 | 0.65 | 0.00021 |
