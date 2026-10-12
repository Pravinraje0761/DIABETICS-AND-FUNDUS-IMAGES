# Phase 3 — Preprocessing Pipeline: Completion Report

**Status:** complete. **Date:** 2026-09-08.
Methodology reference: `../analysis_plan.txt`, Phase 3.

Reproduce end to end:
```
.venv/Scripts/python.exe scripts/phase3_split_aptos.py      # -> data_splits/aptos_*.csv
.venv/Scripts/python.exe scripts/phase3_preprocess.py       # -> data_processed/, *_manifest.csv
.venv/Scripts/python.exe scripts/phase3_sanity_check.py     # -> outputs/phase3_sanity_*.png
.venv/Scripts/python.exe scripts/dataset.py                 # DataLoader smoke test
```

---

## Checklist against the methodology

| Phase 3 task | Status | Where |
|---|---|---|
| One preprocessing script, identical across all three datasets | done | `scripts/preprocessing.py` + `scripts/phase3_preprocess.py` |
| Crop to fundus / remove black borders | done | `crop_borders()` — tolerance-based bounding box (tol=7) |
| Resize to 224x224 (RETFound native) | done | pad-to-square then bilinear resize (no aspect distortion) |
| Normalise with RETFound's expected mean/std | done | ImageNet stats, applied in `dataset.py` transform (not baked into pixels) |
| Optional CLAHE on green channel | available, **off by default** | `--clahe` flag (LAB-L channel; needs opencv-python). Ablation-ready. |
| Split APTOS 70/10/20 stratified by grade, fixed seed, saved to CSV | done | `scripts/phase3_split_aptos.py`, seed 42, `data_splits/aptos_{train,val,test}.csv` |
| PyTorch Dataset/DataLoader reading the split CSVs | done | `scripts/dataset.py` — `FundusDataset`, `build_dataloader` |
| Sanity-check ~20 preprocessed images/dataset with labels | done | `scripts/phase3_sanity_check.py` -> 3 contact sheets, reviewed (see section 4) |
| Commit script + split CSVs | done | local git repo initialised, Phase 2+3 committed |

**Deliverable produced:** preprocessing script; fixed split files; verified/visualised
samples; code committed.

---

## 1. Preprocessing pipeline (`preprocessing.py`)

Deterministic, no randomness. Same six steps for APTOS, Messidor-2, IDRiD:

1. Convert to RGB.
2. **Border crop** - grey = channel mean; keep the tightest bounding box of
   pixels > 7/255. Robust to all-dark images and stray bright pixels.
3. **Pad to centred square** - zero-pad the short side. Preserves aspect ratio;
   no fundus distortion.
4. **Resize to 224x224** - bilinear.
5. *(optional)* CLAHE on the LAB L-channel - **off**. Provided for the
   with/without ablation the methodology suggests.
6. *(optional)* circular corner mask - **off**.

Saved images are uint8 RGB PNG [0, 255] so they stay viewable and reusable.
Channel normalisation (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`)
happens in the `FundusDataset` transform, identically for the baseline CNN and
RETFound.

Config recorded to `logs/phase3_preprocess_config.json`.

### Cross-dataset consistency note (for Methods)

Messidor-2's mirror mixes two source forms in one folder: **1,057** vendor-
preprocessed `*_PP.png` and **687** original `IM*.JPG`. The single Phase 3
pipeline was applied to **all** images of **all three** datasets regardless of
prior processing, so every image the models see has passed through the identical
crop -> square -> 224 -> ImageNet-normalise path. (Open item from Phase 2 -
"re-preprocess Messidor from raw?" - resolved this way: one pipeline on top of
whatever the source form was, rather than trying to source a uniform raw
release.)

---

## 2. APTOS 2019 split (`phase3_split_aptos.py`)

- **Method:** two-stage `sklearn.train_test_split`, `stratify=diagnosis`,
  `random_state=42`. Carve 20% test first, then 12.5% of the remainder as val
  (= 10% overall).
- **Integrity asserted in code:** the three partitions are pairwise disjoint and
  their union is exactly the 3,662 labelled ids.
- Files: `data_splits/aptos_train.csv`, `aptos_val.csv`, `aptos_test.csv`
  (`id_code,diagnosis`), plus `aptos_split_all.csv` (`+split`). Config +
  per-grade counts in `logs/phase3_split_config.json`.

| Grade | Total | Train (70%) | Val (10%) | Test (20%) |
|---|--:|--:|--:|--:|
| 0 No DR | 1,805 | 1,263 | 181 | 361 |
| 1 Mild | 370 | 259 | 37 | 74 |
| 2 Moderate | 999 | 699 | 100 | 200 |
| 3 Severe | 193 | 135 | 19 | 39 |
| 4 Proliferative | 295 | 206 | 30 | 59 |
| **All** | **3,662** | **2,562** | **367** | **733** |

Stratification held to +/-1 image of the exact proportion in every cell. The
internal-validation set has only 19 grade-3 and 30 grade-4 images - expect wide
per-class CIs there; this is inherent to APTOS's size and imbalance, not the
split.

---

## 3. Preprocessing run

| Dataset | Rows | Processed | Failures | Split |
|---|--:|--:|--:|---|
| APTOS 2019 | 3,662 | 3,662 | 0 | train 2,562 / val 367 / test 733 |
| Messidor-2 | 1,744 | 1,744 | 0 | external 1,744 |
| IDRiD | 455 | 455 | 0 | external 455 |

**5,861 images, 0 failures.** Every output is 224x224 RGB. Manifests
(`data_splits/{aptos,messidor2,idrid}_manifest.csv`) carry portable
project-relative paths so they are committable and machine-independent.

Manifest columns:
- all: `id_code, diagnosis, split, source_path, processed_path`
- Messidor-2 also: `adjudicated_dme, adjudicated_gradable`
- IDRiD also: `dme_risk, orig_split` (`idrid_train` / `idrid_test` - both used as
  external test; the official split is irrelevant here)

---

## 4. Sanity check - label alignment

Stratified contact sheets, 4 images per grade, seed 42:
`outputs/phase3_sanity_{aptos,messidor2,idrid}.png`. All three reviewed.

**Result: labels align with visible pathology in all three datasets.**
- Grade 0 panels: clean retinas, no lesions.
- Grade 2: dot/blot haemorrhages, hard exudates appearing.
- Grade 3-4: unmistakable - extensive haemorrhages, cotton-wool spots,
  neovascularisation, and panretinal photocoagulation scars in several
  proliferative panels.

No sign of a shifted/off-by-one label column. Border crop centres the fundus
correctly, including IDRiD's wide letterboxed originals (4288x2848 -> tight
square) and APTOS's varied native sizes.

---

## 5. `FundusDataset` (`dataset.py`)

- `FundusDataset(manifest_csv, split=None, train=False)` ->
  `(FloatTensor[3,224,224], label:int, id_code:str)`.
- `build_transform(train)` - eval: resize+normalise; train: +H/V flip,
  +/-15 deg rotation, mild brightness/contrast jitter. **Deliberately mild** - DR
  lesions are small; aggressive crops would remove the graded signal.
- Helpers: `.class_counts()`, `.class_weights()` (inverse-frequency, mean-
  normalised) for a weighted loss in Phase 4.
- `build_dataloader(...)` convenience wrapper.
- Smoke test (`python scripts/dataset.py`) passes for all five split views;
  tensor ranges approx [-2.12, +2.25] as expected for ImageNet normalisation.

---

## 6. Carried into Phase 4

- **Compute:** torch is installed as **2.14.0+cpu** - no CUDA on this machine.
  Fine-tuning ResNet-50 / EfficientNet-B3 and a ViT-Large RETFound on CPU is
  impractical. Decide on a GPU environment (Colab, Kaggle, cloud) before Phase 4;
  the `data_processed/` folder + manifests + split CSVs are all that needs to move.
- **Class imbalance:** use `FundusDataset.class_weights()` for a weighted
  cross-entropy, and keep quadratic-weighted kappa as the early-stopping metric
  on internal validation (not accuracy).
- **Untouched:** Messidor-2 and IDRiD manifests exist but must not be loaded
  until Phase 6.
- **Optional ablation:** `phase3_preprocess.py --clahe` regenerates a
  CLAHE variant if time allows in Phase 5.

---

## Artefacts produced in Phase 3

| Path | Contents |
|---|---|
| `scripts/preprocessing.py` | The single pipeline (importable functions + constants) |
| `scripts/phase3_split_aptos.py` | Stratified APTOS split, seeded |
| `scripts/phase3_preprocess.py` | Runs the pipeline over all 3 datasets, writes manifests |
| `scripts/dataset.py` | `FundusDataset` / `build_dataloader` |
| `scripts/phase3_sanity_check.py` | Label-alignment contact sheets |
| `data_splits/aptos_{train,val,test,split_all}.csv` | Fixed split (committed) |
| `data_splits/{aptos,messidor2,idrid}_manifest.csv` | id -> label -> processed path |
| `data_processed/{aptos,messidor2,idrid}/*.png` | 5,861 x 224^2 RGB (gitignored, regenerable) |
| `logs/phase3_split_config.json` | Seed, ratios, per-grade counts |
| `logs/phase3_preprocess_config.json` | Pipeline params, counts, failures |
| `outputs/phase3_sanity_{aptos,messidor2,idrid}.png` | Reviewed sanity sheets |
