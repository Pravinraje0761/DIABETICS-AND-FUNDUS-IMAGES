# Online Resource 4 - analysis code

Code for: *Retinal foundation-model versus ImageNet pretraining for deep learning grading of
diabetic retinopathy: an external validation study* (Pravin Raj E).

The archive contains every script used to produce the reported results, the exact software
environment, and the data split lists. Images and
trained weights are not included: the images are available from their original providers (see
the manuscript's Data availability statement) and the weights from the author on reasonable
request. Per-image predictions for every model, run and dataset are in Online Resource 3; with
those files alone, every number, table and figure in the manuscript can be regenerated without
a GPU (steps 6-8 below).

## Environment

Python 3.14.7, PyTorch 2.11 (CUDA 12.8), timm 1.0.29, Windows 11, one NVIDIA RTX 5060 (8 GB).

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128

`requirements.txt` lists the direct dependencies; `requirements-lock.txt` pins every package.

## Expected data layout (relative to the folder that contains this one)

    APTOS 2019:  aptos2019-blindness-detection/train_images/*.png, train.csv
    Messidor-2:  messidor-2/IMAGES/*, the adjudicated grades (Krause et al. 2018) as messidor_data.csv,
                 and the ADCIS examination file messidor-2.csv (left;right), copied to data_splits/messidor2_official_pairs_adcis.csv
    IDRiD:       B. Disease Grading/ (official release)  and  A. Segmentation/ (lesion masks)
    RETFound:    weights/RETFound_cfp_weights.pth

## Pipeline (in order)

| Step | Script(s) | Purpose |
|---|---|---|
| 1 | `phase2_data_audit.py`, `phase3_split_aptos.py`, `phase3_preprocess.py`, `phase28_idrid_official_prep.py` | audit, stratified APTOS split (seed 42), identical preprocessing for all images, official IDRiD manifest and label audit |
| 2 | `phase12_leakage.py`, `phase14_embedding_dedup.py` | byte-level and embedding duplicate checks |
| 3 | `train_baseline.py`, `train_retfound.py` (+ `models.py`, `retfound.py`, `dataset.py`, `losses.py`) | training; launchers `run_round3_gpu.bat`, `run_round4_gpu.bat`, `run_retfound_linprobe_seeds.sh` give the exact arguments of every run (seeds, label-smoothing control, crossed recipes) |
| 4 | `predict.py`, `phase6_external_eval.py` | per-image predictions on the internal and external sets |
| 5 | `phase30b_official_pairs.py` (`phase30_messidor_pairs.py`) | Messidor-2 examinations from the official ADCIS file (and validation of an earlier reconstruction from file names and optic-disc position) |
| 6 | `phase31_paper_numbers.py` | all metrics and bootstrap intervals (examination-level for Messidor-2; two-level over training runs for the primary contrast) |
| 7 | `phase32_tex_numbers.py` | writes every manuscript number as a LaTeX macro |
| 8 | `phase33_figures.py` (+ `phase29_figures.py`), `phase26_lesion_localization.py`, `phase26b_resolution_matched.py` | figures and lesion-mask localisation |

Scripts named `phase7`-`phase25` and `phase29_*` implement the supplementary and sensitivity
analyses reported in Online Resource 1. External datasets were never used for training, model
selection, thresholds or calibration.

## Licence

Code released under the MIT licence for research use. Not a medical device; not for clinical use.
