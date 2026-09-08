# Cross-Dataset External Validation of a Retinal Foundation Model for DR Grading

Reproducibility repository for the study described in
`../option_A_methodology_1.txt`. Compares an ImageNet-pretrained CNN baseline
against a fine-tuned retinal foundation model (RETFound) on 5-class diabetic
retinopathy grading, with the emphasis on **external generalisation**,
explainability, and predictive uncertainty.

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
| 5 | RETFound fine-tuning (linear probe + full), internal-test metrics | **code ready** -- blocked on gated HF weights, see `PHASE5_SETUP.md` |
| 6 | External validation on Messidor-2 + IDRiD; master results table | not started (needs Phase 5) |
| 7 | Grad-CAM / attention maps + plausibility rubric | not started |
| 8 | MC-dropout uncertainty; correct-vs-incorrect test; selective prediction | not started |
| 9 | Bootstrap 95% CIs, DeLong AUC test, stats methods paragraph | not started |
| 10-13 | Figures/tables, manuscript, TRIPOD-AI checklist, submission | not started |

## Scripts

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
| `scripts/train_retfound.py` | 5 | `.venv\Scripts\python.exe scripts\train_retfound.py --mode {linprobe,full} --weights weights\...pth` |

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
