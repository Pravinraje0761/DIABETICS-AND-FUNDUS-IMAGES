# Cross-Dataset External Validation of a Retinal Foundation Model for DR Grading

Reproducibility repository for the study described in
`option_A_methodology_1.txt` (the pre-registered analysis plan). Compares an
ImageNet-pretrained CNN baseline against a fine-tuned retinal foundation model
(RETFound) on 5-class diabetic retinopathy grading, with the emphasis on
**external generalisation**, explainability, and predictive uncertainty.

**Manuscript:** `paper/main_standalone.tex` (IEEEtran). Headline finding: under
a matched preprocessing/fine-tuning/evaluation protocol, RETFound did not
generalise better than a standard ImageNet-pretrained CNN for cross-dataset DR
grading — on the pre-specified contrasts it generalised *worse* — and this
holds under a five-class ordinal metric, a binary referable-DR secondary task,
several training-choice sensitivity analyses (seed, loss function, resolution,
colour normalisation), and a domain-shift out-of-distribution check. See the
manuscript's Results/Discussion for the full picture, including a partial
counterpoint on calibration.

## Design in one line

Train + tune on **APTOS 2019** only, then evaluate **once** on **Messidor-2** and
**IDRiD** as untouched external test sets. Keeping those two sets out of every
training/tuning decision is the whole point of the study.

## Environment

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.14.7 on Windows. See the note in `requirements.txt` about PyTorch /
Python 3.14 compatibility before starting Phase 4.

## Data

Not stored here (licence -- see `DATA_PROVENANCE.md`). Expected layout, relative
to this repo's parent folder:

```
../aptos2019-blindness-detection/   train.csv, train_images/, test.csv, test_images/
../archive/                          messidor_data.csv, messidor-2/messidor-2/preprocess/
../archive (1)/                      idrid_labels.csv, Imagenes/Imagenes/
```

## Progress by phase

| Phase | What | Status |
|---|---|---|
| 2 | Data acquisition, provenance log, class-balance audit | **done** -- `PHASE2_REPORT.md`, `DATA_PROVENANCE.md` |
| 3 | Single preprocessing pipeline; stratified APTOS 70/10/20 split; DataLoader; sanity-viz | **done** -- `PHASE3_REPORT.md` |
| 4 | Baseline CNN (ResNet-50 + EfficientNet-B3), internal-test metrics | **done** -- `PHASE4_REPORT.md`. EfficientNet-B3 selected (test QWK 0.876, acc 0.804) |
| 5 | RETFound fine-tuning (linear probe + full), internal-test metrics | **done** -- `PHASE5_REPORT.md`. linprobe test QWK 0.847; full (ep6, truncated) 0.869 |
| 6 | External validation on Messidor-2 + IDRiD; master results table | **done** -- `PHASE6_REPORT.md`. RETFound shows NO generalisation advantage over the CNN baseline |
| 7 | Grad-CAM++ / attention-rollout maps + automated plausibility proxy | **done** -- `scripts/phase7_explainability.py`, `outputs/phase7/` |
| 8 | Predictive-entropy uncertainty; correct-vs-incorrect (Mann-Whitney U); selective prediction; MC-dropout sensitivity | **done** -- `scripts/phase8_uncertainty.py`, `scripts/phase8_mc_dropout.py`, `outputs/phase8/` |
| 9 | Bootstrap 95% CIs, internal->external drop CIs, difference-in-drop CIs, DeLong AUC test, stats methods paragraph | **done** -- `scripts/phase9_stats.py`, `outputs/phase9/` |
| 10 | Figures (study design, forest plot; Phase 7/8 emit their own) | **done** -- `scripts/phase10_figures.py`, `outputs/figures/` |
| 11 | Independent recompute/verification of every headline number, per-class CIs, calibration, ROC/PR, multiplicity correction, domain-shift table, split counts | **done** -- `scripts/phase11_review_response.py`, `scripts/phase11_figures.py`, `outputs/phase11/` |
| 12 | Duplicate / leakage check (MD5 exact + embedding near-duplicate) | **done** -- `scripts/phase12_leakage.py`, `outputs/phase12/` |
| 13 | Post-hoc temperature-scaling recalibration + robustness (bootstrap CI, disjoint calibration-source check) | **done** -- `scripts/phase13_temperature_scaling.py`, `outputs/phase13/` |
| 14 | Embedding-based cross-dataset near-duplicate check (frozen ImageNet features, cosine) | **done** -- `scripts/phase14_embedding_dedup.py`, `outputs/phase14/` |
| 15 | Controlled degradation ablation: CLAHE / colour-transfer (Variants A/B) + input resolution 224 vs 384 (Variant C), inference-time only | **done** -- `scripts/phase15_ablation.py`, `scripts/phase15c_resolution.py`, `outputs/phase15{,c}/` |
| 16 | Training-seed sensitivity (EfficientNet-B3 + RETFound linear probe, seeds 43-45) | **done** -- `scripts/phase16_seed_sweep.py`, `outputs/phase16/` |
| 17 | Loss-function / class-weighting sensitivity (plain CE, focal loss) | **done** -- `scripts/phase17_loss_sensitivity.py`, `outputs/phase17/` |
| 18 | Training-time retraining-ablation data prep (CLAHE/colour-transfer/384px baked into APTOS, all splits, from source) | **done** (data prep; training itself needs a cloud GPU) -- `scripts/phase18_retrain_variants_prep.py`, `COLAB_RETRAIN_GUIDE.md` |
| 19 | Referable-DR (ICDR grade >=2) as a standalone binary secondary task, with its own internal->external AUROC-drop table | **done** -- `scripts/phase19_referable_dr.py`, `outputs/phase19/` |
| 20 | Domain-shift out-of-distribution detection (Mahalanobis distance) vs. predictive entropy as an external error detector | **done** -- `scripts/phase20_ood_detection.py`, `outputs/phase20/` |
| 21 | Training-set-size sensitivity scorer (pairs with Phase 18 prep; training needs a cloud GPU) | scorer **done**, training pending -- `scripts/phase21_training_size_sensitivity.py`, `outputs/phase21/` |

Manuscript, TRIPOD+AI checklist, cover letter, and submission package are
drafted in `paper/`; remaining author-only items (identity, ORCID, ethics
wording, a clean 30-epoch RETFound full fine-tune, etc.) are tracked in
`paper/AUTHOR_TODO.md` and `TODO.md`.

## Scripts

Phases 2-10 (initial pipeline through first figures) below; every script for
Phases 11-21 is listed in the "Progress by phase" table above and documented
in its own module docstring (`--help` for the exact CLI).

| Script | Phase | Run |
|---|---|---|
| `scripts/phase2_data_audit.py` | 2 | `.venv\Scripts\python.exe scripts\phase2_data_audit.py` |
| `scripts/preprocessing.py` | 3 | importable — the single pipeline + constants |
| `scripts/phase3_split_aptos.py` | 3 | `.venv\Scripts\python.exe scripts\phase3_split_aptos.py` |
| `scripts/phase3_preprocess.py` | 3 | `.venv\Scripts\python.exe scripts\phase3_preprocess.py` |
| `scripts/phase3_sanity_check.py` | 3 | `.venv\Scripts\python.exe scripts\phase3_sanity_check.py` |
| `scripts/dataset.py` | 3+ | `.venv\Scripts\python.exe scripts\dataset.py` (smoke test) |
| `scripts/metrics.py` | 4+ | importable — QWK, per-class sens/spec, OvR AUC, confusion |
| `scripts/models.py` | 4+ | importable — `build_model(arch)` |
| `scripts/train_baseline.py` | 4 | `.venv\Scripts\python.exe scripts\train_baseline.py --arch efficientnet_b3` |
| `scripts/predict.py` | 4/6/8 | `.venv\Scripts\python.exe scripts\predict.py --checkpoint ... --manifest ... --split ... --out-prefix ...` |
| `scripts/retfound.py` | 5+ | importable — ViT-L build + RETFound weight loader |
| `scripts/train_retfound.py` | 5 | `.venv\Scripts\python.exe scripts\train_retfound.py --mode {linprobe,full} --weights weights\...pth` (`--save-every N` throttles the 1.2 GB checkpoint write; `DR_DATA_ROOT` env relocates `data_processed/` off OneDrive) |
| `scripts/phase6_external_eval.py` | 6 | `.venv\Scripts\python.exe scripts\phase6_external_eval.py` — master results table |
| `scripts/phase7_explainability.py` | 7 | `.venv\Scripts\python.exe scripts\phase7_explainability.py [--device cpu]` — Grad-CAM++/attention rollout + plausibility proxy |
| `scripts/phase8_mc_dropout.py` | 8 | `.venv\Scripts\python.exe scripts\phase8_mc_dropout.py --passes 30` — T=30 MC-dropout preds -> `outputs/phase8/mc/` |
| `scripts/phase8_uncertainty.py` | 8 | `.venv\Scripts\python.exe scripts\phase8_uncertainty.py [--preds-dir outputs/phase8/mc --tag mc]` |
| `scripts/phase9_stats.py` | 9 | `.venv\Scripts\python.exe scripts\phase9_stats.py --boot 2000 --seed 42` — bootstrap CIs, drop CIs, DeLong |
| `scripts/phase10_figures.py` | 10 | `.venv\Scripts\python.exe scripts\phase10_figures.py` — Fig 1 (design) + Fig 2 (forest) |

## Reproducing Phases 2–3

```
.venv\Scripts\python.exe scripts\phase2_data_audit.py       # audit + class balance
.venv\Scripts\python.exe scripts\phase3_split_aptos.py      # stratified APTOS split (seed 42)
.venv\Scripts\python.exe scripts\phase3_preprocess.py       # preprocess all 3 datasets -> data_processed/
.venv\Scripts\python.exe scripts\phase3_sanity_check.py     # label-alignment contact sheets
.venv\Scripts\python.exe scripts\dataset.py                 # DataLoader smoke test
```

Headline numbers are summarised in `PHASE2_REPORT.md` and `PHASE3_REPORT.md`.
`data_processed/` (5,861 × 224² PNGs) is gitignored and regenerable from the raw
datasets via `phase3_preprocess.py`.

## Key finding from Phase 2 that shapes later phases

All three datasets are class-imbalanced, and **differently** so (APTOS/Messidor-2
are ~50-58% grade 0; IDRiD's modal class is grade 2 at 34%). Primary metrics are
**quadratic-weighted kappa** and **per-class sensitivity/specificity**, not raw
accuracy. APTOS split must be **stratified by grade** with a fixed seed and saved
to CSV.
