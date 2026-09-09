# Cross-Dataset External Validation of a Retinal Foundation Model for Diabetic Retinopathy Grading: Performance, Explainability, and Uncertainty

_Readable companion. **Authoritative source is now `paper/main_standalone.tex`**
(IEEEtran, 10 figures, 8 tables, Phases 2–16); this render predates the
peer-review-response expansion and is abridged — it omits the calibration /
recalibration, controlled-degradation-ablation, leakage, embedding-dedup and
training-seed sections that the .tex carries, and some table/figure numbers
differ. Where this file and the .tex disagree, the .tex is correct. Numbers here
are from `outputs/phase6/` and `outputs/phase9/`. RETFound full-fine-tune values
marked **[p]** are from an epoch-6 checkpoint at its converged validation QWK
(run stopped early — see Limitations); a completed run refreshes only those
values._

---

## Abstract

**Background.** Retinal foundation models such as RETFound are promoted as
improving generalisation of retinal image analysis, but independent cross-dataset
external validation for diabetic retinopathy (DR) grading remains limited.

**Objective.** To test whether a fine-tuned retinal foundation model generalises
better than an ImageNet-pretrained CNN to independent external datasets for
five-class DR severity grading, and to characterise the explainability and
predictive uncertainty of both models under dataset shift.

**Methods.** An ImageNet-pretrained CNN baseline (EfficientNet-B3 primary,
ResNet-50 secondary) and RETFound (ViT-Large/16, masked-autoencoder pretraining
on ~1.6M colour fundus images) were fine-tuned on APTOS 2019 (n = 3662; stratified
70/10/20 split) with an identical preprocessing, augmentation, loss and
early-stopping protocol, then evaluated **unchanged** on Messidor-2 (n = 1744,
adjudicated reference standard) and IDRiD (n = 455). Primary metric:
quadratic-weighted kappa (QWK). Secondary: accuracy, macro-F1, balanced accuracy,
per-class sensitivity/specificity, one-vs-rest AUC. Explainability: Grad-CAM++
(CNN) / method-matched ViT Grad-CAM against an a-priori plausibility rubric
(automated proxy). Uncertainty: predictive entropy (single pass primary; T = 30
MC-dropout sensitivity), error detection and selective prediction. 95% CIs by
bootstrap (B = 2000); paired AUCs by DeLong's test. TRIPOD+AI reporting.

**Results.** Internal APTOS test QWK: EfficientNet-B3 0.876 (95% CI 0.849–0.901);
RETFound full fine-tune 0.869 **[p]**; RETFound linear probe 0.847. On external
validation **every** model degraded, most on Messidor-2 (EfficientNet-B3 QWK
0.499, drop 0.377 [0.323–0.433]; RETFound full 0.368 **[p]**, drop 0.501 **[p]**).
RETFound did **not** generalise better; on the pre-specified contrasts it
generalised **worse** — its internal→external QWK drop on Messidor-2 was
significantly larger than the baseline's (full fine-tune Δdrop +0.124 **[p]**
[0.074–0.175]; linear probe +0.076 [0.019–0.132]), and the EfficientNet-B3
baseline had a significantly higher referable-DR (grade ≥ 2) AUC on both external
datasets by DeLong's test (all p < 0.001) despite being indistinguishable from
all models internally (p ≥ 0.47). CNN saliency stayed lesion-focused on external
images (75–100% proxy-plausible); RETFound's shifted to the image periphery on
Messidor-2, most for its errors. Predictive entropy was a weak external error
detector for every model (AUROC 0.55–0.68 vs up to 0.84 internally); referring
the most-uncertain 10% of external predictions captured only 10–18% of errors.

**Conclusions.** Fine-tuning a retinal foundation model on APTOS 2019 did not
improve — and by several measures worsened — cross-dataset generalisation for DR
grading relative to a standard ImageNet-pretrained CNN. External validation, weak
uncertainty under shift, and shift-sensitive saliency together argue that
external validation, not internal test performance, should govern claims of
clinical readiness.

**Keywords.** diabetic retinopathy; deep learning; foundation models; RETFound;
external validation; domain shift; explainable AI; uncertainty quantification.

---

## 1. Introduction

DR is a leading cause of preventable vision loss; automated grading of colour
fundus photographs is studied as a route to scalable screening. Reported internal
performance is often high but degrades on images from cameras, populations or
grading protocols unseen in training. Rigorous external validation on independent
datasets is the appropriate evidence standard but is inconsistently reported.

Self-supervised "foundation models" are proposed to improve label efficiency and
robustness. RETFound (Zhou et al., Nature 2023) is a masked-autoencoder ViT
pretrained on ~1.6M retinal images. Whether this pretraining confers better
**cross-dataset** generalisation for DR severity grading — specifically versus a
strong ImageNet CNN under an identical protocol — has not been independently
established.

**Research questions.**
- **RQ1 (Performance).** Does fine-tuned RETFound generalise better than an
  ImageNet CNN to Messidor-2 and IDRiD for five-class DR grading, measured
  primarily by the internal→external QWK drop?
- **RQ2 (Explainability).** Are the image regions driving each model's
  predictions more clinically plausible (visible DR lesions vs optic disc /
  vasculature / borders) for one model than the other, on external images?
- **RQ3 (Uncertainty).** Does predictive uncertainty separate correct from
  incorrect external predictions, and can an uncertainty threshold support
  selective prediction that meaningfully raises accuracy on accepted cases?

**Contribution.** A fully reproducible external validation in which one
preprocessing pipeline, one training protocol and one evaluation protocol are
applied identically to an ImageNet CNN and to RETFound, with a master results
table across one internal and two external datasets, a matched explainability
analysis, an uncertainty analysis with selective-prediction curves, and full code
+ split definitions. The central finding is negative.

---

## 2. Methods (summary — see `main.tex` for full detail)

- **Datasets.** APTOS 2019 (n = 3662, Aravind Eye Hospital, India) —
  **development only**. Messidor-2 (n = 1744, adjudicated reference standard) and
  IDRiD Disease-Grading subset (n = 455 of 516; documented) — **external test
  only**, opened once. Provenance in `DATA_PROVENANCE.md`.
- **Preprocessing.** One script for all three datasets: RGB -> tolerance border
  crop -> pad to square -> 224x224 bilinear; ImageNet normalisation in the loader.
  CLAHE available, off in the primary analysis.
- **Development split.** APTOS 70/10/20 train/val/test, stratified by grade, seed
  42; exact image-ID lists released.
- **Models.** EfficientNet-B3 (primary) and ResNet-50 (secondary),
  ImageNet-pretrained, full fine-tuning, AdamW lr 1e-4, cosine, batch 32, <=30
  epochs, early stop on internal-val QWK (patience 7), inverse-frequency-weighted
  cross-entropy, mild fundus-safe augmentation. RETFound ViT-L/16 MAE checkpoint:
  linear probing (backbone frozen) and full fine-tuning (layer-wise lr decay
  0.65, drop-path 0.2, label smoothing 0.1, lr 1e-4, batch 16). Data,
  normalisation, class weighting, metric set and early-stopping rule identical
  to the baseline.
- **Metrics.** Primary QWK; secondary accuracy, macro-F1, balanced accuracy,
  per-class sensitivity/specificity, one-vs-rest AUC; internal->external drop per
  model per external dataset.
- **External validation.** Each trained model run once, unmodified, no threshold
  tuning, on Messidor-2, IDRiD and (re-scored) APTOS internal test, through the
  identical inference code path. Per-image outputs retained.
- **Explainability.** Grad-CAM++ (CNN, last conv block) / method-matched ViT
  Grad-CAM (last transformer block, patch tokens). Pre-specified plausibility
  rubric scored by a transparent automated proxy (no lesion masks / no second
  rater available): plausible if the saliency peak is inside the fundus, outside
  an estimated optic-disc region and outside the border annulus, with >=50% of
  top-decile mass in that lesion-eligible zone.
- **Uncertainty.** Predictive entropy of the softmax (single pass primary; T = 30
  MC-dropout / drop-path sensitivity). One-sided Mann-Whitney U (incorrect >
  correct), AUROC of entropy as an error detector, selective-prediction curves.
- **Statistics.** Percentile bootstrap 95% CIs (B = 2000, seed 42); internal and
  external sets resampled independently for the drop; paired image-level
  resampling for the between-model difference in drop. DeLong two-sided test for
  paired AUC on the referable-DR (grade >= 2) binarisation and per grade.
  alpha = 0.05, two-sided. Python 3.14, numpy/pandas/scipy/scikit-learn.
  Code: `scripts/phase7_explainability.py`, `phase8_uncertainty.py`,
  `phase8_mc_dropout.py`, `phase9_stats.py`, `phase10_figures.py`.

### Table 1. Dataset characteristics

| Dataset | n | Role | Grade share % (0/1/2/3/4) |
|---|---:|---|---|
| APTOS 2019 | 3662 | Development | 49.3 / 10.1 / 27.3 / 5.3 / 8.1 |
| Messidor-2 | 1744 | External test | 58.3 / 15.5 / 19.9 / 4.3 / 2.0 |
| IDRiD | 455 | External test | 28.4 / 4.8 / 34.3 / 18.5 / 14.1 |

---

## 3. Results

### 3.1 Table 2 — Master results with 95% bootstrap CI

| Model | Dataset | n | Accuracy (95% CI) | QWK (95% CI) | AUC OvR (95% CI) |
|---|---|---:|---|---|---|
| EfficientNet-B3 | APTOS-internal | 733 | 0.804 (0.776–0.832) | 0.876 (0.849–0.901) | 0.928 (0.911–0.943) |
| EfficientNet-B3 | Messidor-2 | 1744 | 0.626 (0.603–0.648) | 0.499 (0.452–0.543) | 0.775 (0.761–0.789) |
| EfficientNet-B3 | IDRiD | 455 | 0.497 (0.451–0.545) | 0.760 (0.715–0.798) | 0.830 (0.805–0.856) |
| ResNet-50 | APTOS-internal | 733 | 0.789 (0.759–0.819) | 0.875 (0.848–0.898) | 0.924 (0.908–0.939) |
| ResNet-50 | Messidor-2 | 1744 | 0.614 (0.591–0.635) | 0.435 (0.381–0.481) | 0.754 (0.738–0.769) |
| ResNet-50 | IDRiD | 455 | 0.448 (0.400–0.497) | 0.707 (0.663–0.751) | 0.841 (0.818–0.862) |
| RETFound linear probe | APTOS-internal | 733 | 0.759 (0.728–0.789) | 0.847 (0.819–0.873) | 0.908 (0.889–0.927) |
| RETFound linear probe | Messidor-2 | 1744 | 0.589 (0.565–0.612) | 0.394 (0.338–0.447) | 0.758 (0.738–0.774) |
| RETFound linear probe | IDRiD | 455 | 0.336 (0.297–0.376) | 0.626 (0.572–0.676) | 0.778 (0.751–0.805) |
| RETFound full fine-tune **[p]** | APTOS-internal | 733 | 0.769 (0.739–0.798) | 0.869 (0.842–0.893) | 0.915 (0.896–0.934) |
| RETFound full fine-tune **[p]** | Messidor-2 | 1744 | 0.602 (0.579–0.625) | 0.368 (0.311–0.422) | 0.758 (0.742–0.772) |
| RETFound full fine-tune **[p]** | IDRiD | 455 | 0.420 (0.376–0.466) | 0.720 (0.673–0.759) | 0.821 (0.796–0.847) |

Full CIs for every metric: `outputs/phase9/bootstrap_ci.csv`.

### 3.2 Table 3 — Internal->external QWK drop and difference vs the baseline

| Model | External set | QWK ext | Drop (95% CI) | dDrop vs EfficientNet-B3 (95% CI) |
|---|---|---:|---|---|
| EfficientNet-B3 | Messidor-2 | 0.499 | 0.377 (0.323–0.433) | reference |
| EfficientNet-B3 | IDRiD | 0.760 | 0.116 (0.068–0.166) | reference |
| ResNet-50 | Messidor-2 | 0.435 | 0.440 (0.388–0.499) | +0.063 (0.017–0.109) |
| ResNet-50 | IDRiD | 0.707 | 0.167 (0.117–0.222) | +0.051 (0.003–0.102) |
| RETFound linear probe | Messidor-2 | 0.394 | 0.453 (0.393–0.516) | +0.076 (0.019–0.132) |
| RETFound linear probe | IDRiD | 0.626 | 0.221 (0.163–0.285) | +0.105 (0.038–0.170) |
| RETFound full fine-tune **[p]** | Messidor-2 | 0.368 | 0.501 (0.441–0.563) | +0.124 (0.074–0.175) |
| RETFound full fine-tune **[p]** | IDRiD | 0.720 | 0.150 (0.101–0.202) | +0.033 (−0.022–0.089) |

CI excludes 0 => statistically distinguishable loss of external ordinal agreement
vs the baseline. All rows except RETFound-full/IDRiD exclude 0.

### 3.3 DeLong paired-AUC, referable DR (grade >= 2), vs EfficientNet-B3

| Dataset | Comparator | AUC (EfficientNet-B3) | AUC (comparator) | p |
|---|---|---:|---:|---:|
| APTOS-internal | ResNet-50 | 0.980 | 0.978 | 0.53 |
| APTOS-internal | RETFound linear probe | 0.980 | 0.976 | 0.47 |
| APTOS-internal | RETFound full fine-tune | 0.980 | 0.978 **[p]** | 0.66 |
| Messidor-2 | ResNet-50 | 0.842 | 0.789 | <0.001 |
| Messidor-2 | RETFound linear probe | 0.842 | 0.781 | <0.001 |
| Messidor-2 | RETFound full fine-tune | 0.842 | 0.789 **[p]** | <0.001 |
| IDRiD | ResNet-50 | 0.962 | 0.950 | 0.076 |
| IDRiD | RETFound linear probe | 0.962 | 0.903 | <0.001 |
| IDRiD | RETFound full fine-tune | 0.962 | 0.873 **[p]** | <0.001 |

Per-grade one-vs-rest DeLong: `outputs/phase9/delong.csv`.

### 3.4 Internal performance

EfficientNet-B3 QWK 0.876 / accuracy 0.804; RETFound full fine-tune QWK
0.869 **[p]** / accuracy 0.769 **[p]**; RETFound linear probe QWK 0.847. RETFound
had flatter per-class sensitivity, with higher recall on the rare proliferative
grade than the CNNs.

### 3.5 External validation and drop (RQ1)

Every model degraded on both external datasets. On Messidor-2 QWK fell from
~0.87 to 0.37–0.50 (non-overlapping CIs) and all models reverted toward grade-0
prediction (grade-1 sensitivity <= 0.02; grade-0 specificity 0.18–0.32). On IDRiD
the QWK drop was smaller (0.12–0.22) but accuracy fell further (0.31–0.35),
consistent with IDRiD's pathology-enriched class mix.

RETFound did not generalise better than EfficientNet-B3 on either external
dataset; on the pre-specified contrasts it generalised worse (Tables 3, 3.3).
RETFound retained isolated per-class advantages under shift (highest severe-grade
sensitivity on IDRiD; highest proliferative-grade sensitivity on Messidor-2) and,
under linear probing, the highest balanced accuracy on Messidor-2 (0.42 vs
0.34–0.35) — a more even but not more accurate error distribution.

### 3.6 Explainability (RQ2)

160 external saliency maps (20 per model per external dataset). CNN Grad-CAM++
maps: 92–100% proxy-plausible on IDRiD, 75–100% on Messidor-2, ~0.92–0.99 of
top-decile mass on retinal tissue. RETFound ViT Grad-CAM: comparably lesion-
focused on IDRiD (83–88% for the full fine-tune) but not on Messidor-2 (12–50%
for full fine-tune, 25–58% for linear probe; 27–54% of top-decile mass on the
border annulus vs ~0 for CNNs), and worse for RETFound's incorrect Messidor-2
predictions than its correct ones. Automated proxy, not blinded expert grading.

### 3.7 Uncertainty (RQ3)

Internal AUROC of entropy as an error detector: EfficientNet-B3 0.84, RETFound
full fine-tune 0.63. External AUROC 0.55–0.68 for every model; RETFound full
fine-tune on IDRiD showed **no** error signal (AUROC 0.50, Mann-Whitney p = 0.52).
Misclassified images carried significantly higher entropy than correct ones on
most model x external combinations (p < 0.001; exceptions ResNet-50/IDRiD
p = 0.04 and RETFound-full/IDRiD p = 0.52). Referring the most-uncertain 10% of
external predictions captured only 9–18% of errors (~ random) vs 21–30%
internally. Unchanged under T = 30 MC-dropout (Supplement).

---

## 4. Discussion

- **RQ1.** Under a matched protocol, fine-tuning RETFound on APTOS did not improve
  cross-dataset generalisation and, on the pre-specified contrasts, worsened it.
  Consistent across linear probing and full fine-tuning, so not an artefact of
  the early-stopped full run. One reading: full fine-tuning a large ViT on ~2.6k
  images fits APTOS-specific acquisition characteristics at least as readily as
  it exploits pretrained retinal features.
- **Nature of the shift.** Messidor-2 = ordinal-structure failure (accuracy
  partly preserved by grade-0 prevalence while QWK collapses). IDRiD = base-rate
  shift toward advanced disease (accuracy falls more than QWK). Reporting
  balanced accuracy alongside QWK separates the two.
- **RQ2.** Neither family more plausible overall; on the harder Messidor-2 shift
  the CNN stayed on-tissue while RETFound's saliency went peripheral, most for
  its errors — consistent with the performance result. Hypothesis-generating
  (automated proxy).
- **RQ3.** Predictive uncertainty was at best a weak safeguard under shift; an
  uncertainty-gated referral step is not, on this evidence, sufficient for
  deployment. Domain-specific calibration / OOD detection would be needed.
- **Literature.** Internal QWK matches typical APTOS reports (0.85–0.93);
  internal->external five-class QWK drops of this magnitude are rarely reported
  because five-class external validation on an adjudicated standard is uncommon.
  Cross-population degradation of DR models is known to be substantial; this
  extends the observation to a retinal foundation model under a matched protocol.
- **Limitations.** (i) RETFound full fine-tune stopped at epoch 6/30 (hardware
  throttling), at its converged validation QWK; values marked **[p]**; conclusion
  robust (linear probe concordant; Messidor-2 deficit large). (ii) retrospective,
  public data; no prospective claim. (iii) IDRiD 455/516 subset (external test —
  no development bias; reduced precision). (iv) rare-grade precision limited
  externally. (v) automated plausibility proxy (40 correct + 25 incorrect per
  model per external set), no blinded expert rater. (vi) one foundation model,
  one task. (vii) calibration assessed (ECE/MCE/Brier/NLL + reliability
  diagrams): RETFound is better-calibrated than the CNNs under shift, but a
  single in-domain temperature does not repair external miscalibration and its
  sign differs between Messidor-2 and IDRiD.
- **Sensitivity analyses.** Ranking stable to: ResNet-50 backbone; linear-probe
  vs full fine-tune; single-pass vs T=20 MC-dropout entropy; independent
  recompute of all 60 cells; temperature-scaling calibration source; inference-
  time CLAHE / colour-transfer / 224-vs-384 resolution (none recover RETFound's
  external QWK — 384 leaves it far below the CNN at any resolution, delta-drop
  small and inconsistent-sign — so the gap is not a preprocessing or working-
  resolution artefact); and four training seeds each for EfficientNet-B3 and
  RETFound linear probing — internal QWK SD 0.003 / 0.008, external QWK SD 0.024
  / 0.015 (EffNet) vs 0.015 / 0.012 (RETFound-LP) on Messidor-2 / IDRiD; the
  per-model seed ranges do not overlap on either external set (Messidor-2
  0.469–0.527 vs 0.374–0.408), Messidor-2 drop 0.379 +/- 0.025 vs 0.461 +/- 0.017,
  so neither direction nor magnitude of the gap is seed noise; RETFound-full's
  +0.124 excess drop is ~5x the baseline seed SD.
- **Implications.** Internal test performance is a poor guide to behaviour on new
  data; external validation and, ideally, a validated uncertainty-aware referral
  mechanism should precede screening deployment — and neither retinal pretraining
  nor an uncertainty threshold can be assumed to deliver that robustness alone.

---

## 5. Conclusions

A fine-tuned retinal foundation model did not outperform — and by the
pre-specified comparisons underperformed — a standard ImageNet-pretrained CNN for
cross-dataset DR grading. All models lost substantial ordinal agreement on
independent data; predictive uncertainty was a weak safeguard under shift; and
the foundation model's saliency became peripheral rather than lesion-centred.
External validation on an independent adjudicated reference standard should be the
primary evidence standard for retinal AI.

---

## Declarations (see `main.tex`; author input required — `paper/AUTHOR_TODO.md`)

Data & Code Availability (repo DOI to be minted), Ethics (public de-identified
secondary use — journal wording to confirm), Funding, Conflicts of Interest,
Author Contributions (CRediT), Use of AI tools (disclosed).
