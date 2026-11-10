# Reviewer round 2: analyses for the four methodological points

Prepared 2026-11-06. **These are analyses only. The manuscript has not been edited.** All numbers below are recomputed from the frozen checkpoints' prediction files (`outputs/phase6/`, `outputs/phase8/mc_lp_deterministic/`). Bootstrap: B = 2000, seed 42, image-level. The selective-prediction gain CIs use B = 1000.

Consistency check: Phase 24 reproduces every Phase 8 entropy AUROC exactly (max difference 1e-16). Phase 25 reproduces all four Phase 23 confirmatory ΔDrop values exactly.

| Point | Status | Where |
|---|---|---|
| 1. Fine-tuning completeness | **Done (2026-10-21)**; verified | `outputs/phase5_retfound/`, `CHECKPOINTS.md` |
| 2. Dataset shift vs site generalisation | **Done** | `scripts/phase25_shift_decomposition.py` → `outputs/phase25/` |
| 3a. Lesion-mask localisation | **Script ready; needs IDRiD Segmentation download + one GPU run** | `scripts/phase26_lesion_localization.py`, `run_phase26_27.bat` |
| 3b. Inter-rater agreement | **Script ready; needs ≥2 human raters** | `scripts/phase27_rater_agreement.py` |
| 4. Uncertainty validity | **Done** | `scripts/phase24_uncertainty_validity.py` → `outputs/phase24/` |

---

## 1. Fine-tuning completeness

The provisional epoch-6 model has been replaced. The full fine-tune ran under the pre-specified protocol (`full_hyperparams.json`):

- ViT-L/16 RETFound MAE (natureCFP), 224 px
- AdamW: lr 1e-4, weight decay 0.05, layer decay 0.65, drop-path 0.2, label smoothing 0.1
- 3-epoch warm-up, then cosine decay; at most 30 epochs
- Early stopping on APTOS-val QWK with patience 7
- Batch size 16, AMP, seed 42

The run stopped early at epoch 28 and used the epoch-21 checkpoint (val QWK 0.939). The full per-epoch history is in `full_history.csv`. The checkpoint is `full_best.pt`, SHA-256 `b1d4a7b2…5fd5933`.

I checked that nothing downstream still uses the epoch-6 model. Phases 6, 7, 8 (MC), 9, 11, 13, 19, 20 and 23 were all regenerated on 2026-10-21 from the completed model. The only remaining mention of epoch-6 numbers in `main_standalone.tex` (around line 1380) is the deliberate account of the 0.368 → 0.487 change. The superseded outputs are archived with their hash.

**Remaining weakness: there is still only one seed.** If GPU time allows, 2 more seeds (43, 44) of `run_retfound_full_clean.bat` would turn "single seed" into a seed spread. The linear probe and EfficientNet-B3 already have 4 seeds each (Phase 16).

---

## 2. Dataset shift vs independent clinical-site generalisation

### What kind of external validation each set is

| | Messidor-2 | IDRiD |
|---|---|---|
| Site | Different country (France), different programme | Same country (India), different institution (Nanded vs Aravind) |
| Camera | Topcon TRC NW6, non-mydriatic | Kowa VX-10α, mydriatic, 50° |
| Reference standard | **Adjudicated 3-specialist consensus** (Krause 2018) | "Medical experts", adjudication not reported |
| Development reference (APTOS) | Single clinician per image | |
| Case mix | Gradable-only, curated; referable 26.2% | Pathology-enriched; referable 66.8% |
| What it tests | Geographic, camera *and* reference-standard transfer | Institution and camera transfer within one population, under a very different case mix |

Neither set is a consecutive screening population, and neither was collected prospectively. Both are curated research releases. So neither tests clinical-site generalisation in the TRIPOD-AI sense. Both test transfer to a different curated dataset. That should be stated as a limitation of scope, not left implicit. Phases 12 and 14 found no duplicate or near-duplicate images between the sets.

### Decomposing the QWK drop (`outputs/phase25/phase25_summary.md`)

Three corrections were applied, alone and together:

- **Reweighted:** external images are importance-weighted to the APTOS grade mix, which removes the prevalence shift.
- **4-level:** grades 0 and 1 are merged, which removes disagreement at the No-DR/mild boundary.
- **Both:** the two corrections combined.

QWK drop (internal − external) under each:

| Model | External | Raw | Reweighted | 4-level | Both |
|---|---|---|---|---|---|
| EfficientNet-B3 | Messidor-2 | 0.377 | 0.293 | 0.200 | **0.141** [0.052, 0.234] |
| EfficientNet-B3 | IDRiD | 0.116 | 0.115 | 0.062 | **0.006** [−0.068, 0.085] |
| RETFound full FT | Messidor-2 | 0.395 | 0.284 | 0.202 | **0.117** [0.044, 0.194] |
| RETFound full FT | IDRiD | 0.139 | 0.161 | 0.097 | **0.084** [0.011, 0.157] |

**Messidor-2: the drop is mostly not image shift.**
- Reweighting plus merging grades 0/1 removes 52–73% of the drop across the four models.
- Specificity stays at 0.97–0.98. Every model sends 96–99% of reference-standard *mild* images to No DR, and about 75% of *moderate* images to grade 0. For example, EfficientNet-B3 predicts grade 0 for 249/347 moderate images.
- So referable sensitivity falls from about 0.90 to 0.36–0.43.
- This pattern (normal eyes recognised, subtle disease under-called) is what you would expect if adjudicated specialists grade subtle lesions more readily than APTOS's single grader. Part of it could also be the lower visibility of microaneurysms on this camera. The predictions alone cannot separate those two explanations.
- A residual drop of 0.12–0.21 remains significant after both corrections. That residual is the best estimate of the genuine camera/population effect.

**IDRiD: the error goes in the opposite direction.**
- About half of grade-0 images are called *mild*: 61/129 for EfficientNet-B3 and 54/129 for RETFound. The models over-call.
- For EfficientNet-B3, case mix plus the 0/1 boundary explain essentially all of the drop (residual 0.006, CI includes 0).
- RETFound keeps a small residual (0.084, CI excludes 0). Its moderate cases are split between grades 1 and 3 (62 and 42 of 156).

**The headline comparison is robust to every correction.** Difference in drop for the full fine-tune vs EfficientNet-B3, with both corrections applied:
- Messidor-2: −0.024 [−0.096, +0.042]
- IDRiD: +0.079 [−0.009, +0.166]

Both CIs still include 0, so "neither better nor measurably worse" holds.

**One finding changes.** The linear probe's Messidor-2 deficit disappears once prevalence is reweighted: +0.076 [0.020, 0.131] becomes +0.004 [−0.060, 0.069]. So that deficit is a case-mix interaction, not a transfer failure. The IDRiD deficit grows instead (+0.196 [0.117, 0.275]), driven by low specificity (0.64). Claims about the linear probe should be worded per dataset.

Prevalence-invariant referable sensitivity and specificity, and PPV re-expressed at APTOS prevalence, are in `referable_invariant.csv`.

---

## 3. Explainability validity

### 3a. Quantitative localisation against expert lesion masks

**Data needed (cannot be fetched from here):** the IDRiD **"A. Segmentation"** subset from IEEE DataPort (free account, CC BY 4.0). It contains 81 images with expert pixel masks for MA, HE, EX, SE and the optic disc. None of these images were used in training or model selection.

**To run:** unzip the subset, set `SEG=` in `run_phase26_27.bat` to that folder, then double-click the file. It runs one GPU pass, about 81 images × 4 models.

**What the script computes (`scripts/phase26_lesion_localization.py`).** It uses the same Grad-CAM++ and ViT Grad-CAM maps as the paper, for the predicted class. Masks go through the exact same crop/pad/resize as the images, with max-pool downsampling so microaneurysms survive. For each image:

- **Lesion energy:** share of saliency mass that falls on lesions.
- **Lift:** lesion energy ÷ lesion area share. 1 means no better than a uniform map.
- **Pointing game:** whether the saliency peak lands on a lesion.
- **Pixel AUROC** of saliency for lesion vs non-lesion pixels.
- **Top-10% precision:** fraction of the most salient pixels that are lesion.
- **Lift per lesion type.**
- **Energy on the expert optic-disc mask.** This replaces the estimated disc in the current proxy.

**Controls:** a uniform map, a centre-Gaussian prior, and a *cross-image* control (the same model's map for a different image). The cross-image control tests whether the map tracks *this* image's lesions or only a habitual location. Results are reported as per-model means with bootstrap CIs, plus paired Wilcoxon tests vs each control and vs EfficientNet-B3.

**Testing done so far:** I checked the metric code end-to-end on synthetic images. A map placed on the lesion scored lift ≈ 100, AUROC 1.0 and pointing hit 1.0. A centre map scored lift ≈ 1.6 and pointing hit 0. **It has not been run on real data.**

### 3b. Inter-rater agreement

The blinded 160-map packet already exists (`outputs/phase7/rating_packet/`). Two or more raters with fundus-grading experience need to complete `rating_sheet_BLANK.csv` independently.

`scripts/phase27_rater_agreement.py` then reports, with bootstrap CIs:
- Cohen's or Fleiss' κ, Gwet's AC1 (robust when most maps get the same rating) and Krippendorff's α
- the same statistics by model family, by dataset and by correct/incorrect prediction
- quadratic-weighted κ for confidence
- intra-rater κ, if a rater re-rates a reshuffled copy at least 2 weeks later (`--rerate`)
- agreement of the rater consensus with the automated proxy

The statistics were checked against sklearn's Cohen κ (exact match) and against simulated raters. **No real ratings exist.** The packet's DEMO files are synthetic and must not be reported.

---

## 4. Uncertainty validity

`outputs/phase24/phase24_summary.md` covers error detection, selective prediction and AURC. It includes risk–coverage curves in `fig_risk_coverage.png`.

Six scores were compared:
- **Simple baselines:** 1 − max probability (MSP), top-2 margin, random ranking
- **Scores from the paper:** single-pass entropy, T = 20 entropy
- **Ordinal score:** grade SD (spread of the predicted grade)

Two error types were used: any grade error, and a wrong referable/non-referable call. Referral rates were 0, 5, 10, 20, 30, 40 and 50%.

### Does uncertainty identify errors? (grade-error AUROC)

| Model | Internal | Messidor-2 | IDRiD |
|---|---|---|---|
| EfficientNet-B3, MSP | 0.840 | 0.682 | 0.587 |
| RETFound full FT, MSP | 0.635 | 0.653 | 0.518 |
| RETFound full FT, entropy | 0.516 | 0.633 | 0.491 |

- **For the CNNs, entropy offers nothing beyond simple confidence.** Every ΔAUROC vs MSP is within ±0.01.
  - The one exception: T = 20 entropy beats MSP for EfficientNet-B3 on IDRiD (+0.039 [+0.013, +0.066]).
- **For RETFound, entropy is significantly *worse* than MSP.**
  - Full fine-tune: −0.118 internally and −0.020 on Messidor-2, both CIs excluding 0.
  - Linear probe: −0.067 internally.
  - Margin is the best score for RETFound.
  - Label smoothing plus class weighting flatten RETFound's softmax, so its entropy is dominated by tail mass and stays around 1.0–1.3 nats for every image.
  - **This changes the paper's claim.** "RETFound full FT has no uncertainty signal internally" was a property of entropy as the score, not of the model.
- **On IDRiD, no score detects RETFound full-FT errors.** All scores fall in 0.46–0.53. This null holds across scores.
- **Grade SD should not be used for RETFound.** Its AUROC is 0.26–0.45, below chance, because its value depends on the predicted grade.

### Does selective prediction improve performance? (MSP ranking)

Accuracy on the retained cases rises on every set: about +0.03 per 10% referred on Messidor-2, and less on IDRiD. On IDRiD, EfficientNet-B3 only gains +0.042 [0.009, 0.071] at 30%.

**On Messidor-2, the accuracy gain is misleading.** The CNNs refer out the diseased cases. Retained QWK *falls* (EfficientNet-B3: 0.499 → 0.188 at 30% referral) and retained referable sensitivity falls to 0.06.

The correct framing is human-in-the-loop: referred cases go to a grader. Under that framing:
- On Messidor-2, the share of all referable cases the system misses falls from 0.61 to 0.28 at 30% referral (EfficientNet-B3), and from 0.61 to 0.26 for RETFound full FT.
- On IDRiD, the same share falls only from 0.14 to 0.10 (EfficientNet-B3) and from 0.24 to 0.17 (RETFound full FT).

Suggested framing: uncertainty-based referral helps under the Messidor-2 type of shift. Under the IDRiD type of shift it does little more than random referral.

Entropy vs MSP at the same referral rate makes no significant difference for the CNNs. For RETFound full FT internally, entropy is significantly worse (−0.049 accuracy at 30%).

---

## Still needs you

1. Download the IDRiD Segmentation subset and run `run_phase26_27.bat` on the GPU laptop. Send me `outputs/phase26/phase26_summary.md` and I'll interpret it.
2. Recruit 2 blinded raters for the 160-map packet. A 2-week reshuffled re-rate by one rater gives intra-rater reliability.
3. Optional: 2 more RETFound full fine-tune seeds.
4. Tell me how you want this written up.
