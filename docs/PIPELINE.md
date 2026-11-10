# Analysis pipeline

The study ran as numbered phases. Each phase is one script under `scripts/` that writes to
`outputs/phase<N>/`. Run each with `--help` for its exact options. Commands assume you
are at the repository root with the environment from the main README active.

## Phases

| Phase | What | Script(s) | Outputs | Status |
|---|---|---|---|---|
| 2 | Data audit, provenance log, class balance | `phase2_data_audit.py` | `outputs/phase2_*`, `logs/phase2_audit_report.md` | done |
| 3 | Single preprocessing pipeline; stratified APTOS 70/10/20 split (seed 42); sanity contact sheets | `phase3_split_aptos.py`, `phase3_preprocess.py`, `phase3_sanity_check.py`, `preprocessing.py`, `dataset.py` | `data_splits/`, `outputs/phase3_sanity_*.png` | done |
| 4 | ImageNet CNN baselines (EfficientNet-B3, ResNet-50) | `train_baseline.py --arch {efficientnet_b3,resnet50}` | `outputs/phase4_baseline/` | done |
| 5 | RETFound linear probe and full fine-tune | `train_retfound.py --mode {linprobe,full}` | `outputs/phase5_retfound/` | done (full run stopped at epoch 6; see Limitations) |
| 6 | External validation on Messidor-2 and IDRiD; master results table | `phase6_external_eval.py` | `outputs/phase6/` | done |
| 7 | Grad-CAM++ / ViT Grad-CAM saliency and automated plausibility proxy; blinded-rating packet | `phase7_explainability.py`, `phase7_rating_packet.py` | `outputs/phase7/` | done |
| 8 | Predictive-entropy uncertainty, selective prediction, MC-dropout sensitivity | `phase8_uncertainty.py`, `phase8_mc_dropout.py` | `outputs/phase8/` | done |
| 9 | Bootstrap 95% CIs, drop CIs, paired difference-in-drop CIs, DeLong tests | `phase9_stats.py --boot 2000 --seed 42` | `outputs/phase9/` | done |
| 10 | Figures | `phase10_figures.py`, `phase10b_figures.py` | `outputs/figures/` | done |
| 11 | Independent recompute of every headline number; per-class CIs; calibration; ROC/PR; Holm–Bonferroni; domain-shift table | `phase11_review_response.py`, `phase11_figures.py` | `outputs/phase11/` | done |
| 12 | Exact (MD5) and perceptual duplicate / leakage check | `phase12_leakage.py` | `outputs/phase12/` | done |
| 13 | Post-hoc temperature scaling | `phase13_temperature_scaling.py` | `outputs/phase13/` | done |
| 14 | Embedding-based cross-dataset near-duplicate check | `phase14_embedding_dedup.py` | `outputs/phase14/` | done |
| 15 | Inference-time ablations: CLAHE, colour transfer, 224 vs 384 px | `phase15_ablation.py`, `phase15c_resolution.py` | `outputs/phase15/`, `outputs/phase15c/` | done |
| 16 | Training-seed sensitivity (seeds 43–45) | `phase16_seed_sweep.py` | `outputs/phase16/` | done |
| 17 | Loss / class-weighting sensitivity (plain CE, focal) | `phase17_loss_sensitivity.py` | `outputs/phase17/` | done for CNN and RETFound linear probe |
| 18 | Data prep for training-time retraining ablations | `phase18_retrain_variants_prep.py` | manifests in `data_splits/` | data prep done; training needs a cloud GPU |
| 19 | Referable DR (ICDR ≥ 2) as a standalone binary task | `phase19_referable_dr.py` | `outputs/phase19/` | done |
| 20 | Mahalanobis out-of-distribution detection vs. entropy as an error detector | `phase20_ood_detection.py` | `outputs/phase20/` | done |
| 21 | Training-set-size sensitivity | `phase21_training_size_sensitivity.py` | `outputs/phase21/` | scorer done; training needs a cloud GPU |

## Shared modules

| Module | Purpose |
|---|---|
| `scripts/preprocessing.py` | The single preprocessing pipeline (crop black frame, pad to square, resize to 224) and ImageNet normalisation constants |
| `scripts/dataset.py` | Dataset / DataLoader over the split manifests |
| `scripts/models.py` | `build_model(arch)` for the CNN baselines |
| `scripts/retfound.py` | ViT-L/16 construction and RETFound weight loading |
| `scripts/losses.py` | Weighted cross-entropy and focal loss |
| `scripts/metrics.py` | QWK, per-class sensitivity/specificity, one-vs-rest AUC, confusion matrix |
| `scripts/predict.py` | Inference over a manifest, writing per-image prediction CSVs |
| `scripts/colour_ops.py` | Colour-transfer and CLAHE operations for the Phase 15/18 ablations |

## End-to-end order

```bash
python scripts/phase2_data_audit.py
python scripts/phase3_split_aptos.py
python scripts/phase3_preprocess.py
python scripts/train_baseline.py --arch efficientnet_b3
python scripts/train_baseline.py --arch resnet50
python scripts/train_retfound.py --mode linprobe --weights weights/RETFound_mae_natureCFP.pth
python scripts/train_retfound.py --mode full     --weights weights/RETFound_mae_natureCFP.pth
python scripts/phase6_external_eval.py
python scripts/phase7_explainability.py
python scripts/phase8_uncertainty.py
python scripts/phase9_stats.py --boot 2000 --seed 42
python scripts/phase10_figures.py
# Phases 11-21: run the corresponding scripts in order; see --help for each.
```

Phases 6 onward that only re-score predictions can be run without a GPU from the committed
files in `outputs/phase6/`. Training (Phases 4–5, 16–18, 21) needs a GPU and the raw data.

## Revision phases (manuscript version of November 2026)

| Phase | Script | Purpose | Outputs |
|---|---|---|---|
| 24 | `phase24_uncertainty_validity.py` | Error detection (MSP, margin, entropy) and selective referral, including the fraction of referable DR missed | `outputs/phase24/` |
| 25 | `phase25_shift_decomposition.py` | Case-mix reweighting and no-DR/mild merging of the external drop | `outputs/phase25/` |
| 26, 26b | `phase26_lesion_localization.py`, `phase26b_resolution_matched.py` | Saliency against IDRiD expert lesion masks, with centre and cross-image controls and a 7×7 resolution-matched control | `outputs/phase26/` |
| 27 | `phase27_rater_agreement.py` | Agreement statistics for a blinded rating packet | — |
| 28 | `phase28_idrid_official_prep.py` | Official IDRiD manifest (516 images) and audit of the third-party copy | `outputs/phase28/` |
| 29 | `phase29_calibration_ls.py`, `phase29_delong.py`, `phase29_master_table.py`, `phase29_figures.py` | Label-smoothing calibration control, referable-DR AUC tests, figure style | `outputs/phase29/` |
| 30, 30b | `phase30_messidor_pairs.py`, `phase30b_official_pairs.py` | Messidor-2 examinations from the official ADCIS file (needs `messidor-2.csv`, not redistributed); validation of an earlier reconstruction | `outputs/phase30/` |
| 31 | `phase31_paper_numbers.py` | All manuscript metrics; examination-level bootstrap; seed-averaged primary contrast with two-level bootstrap; crossed-recipe contrasts | `outputs/phase31/` |
| 32 | `phase32_tex_numbers.py` | Every manuscript number as a LaTeX macro (`paper/submission_io/generated/`) | `numbers.tex`, `numbers.json` |
| 33 | `phase33_figures.py` | Manuscript Figures 1–5 | `outputs/phase33/figures/` |
| 34 | `phase34_code_archive.py` | Online Resources 3 (predictions) and 4 (code) | `paper/submission_io/` |
| 35 | `phase35_esm1.py` | Data-driven sections of Online Resource 1 | `outputs/phase35/` |

Additional training runs (seeds 43–45, label-smoothing control, crossed recipes) are launched by
`run_round3_gpu.bat`, `run_round4_gpu.bat` and `run_seedpreds_cpu.bat`.

```bash
for m in baseline_efficientnet_b3 baseline_resnet50 retfound_linprobe retfound_full; do
  python scripts/phase31_paper_numbers.py --model $m; done   # and every other run in ALL_MODELS
python scripts/phase31_paper_numbers.py --merge
python scripts/phase32_tex_numbers.py
python scripts/phase33_figures.py
```
