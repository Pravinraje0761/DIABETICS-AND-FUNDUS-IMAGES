# Data Provenance Log

Prepared for the Data Availability statement (Phase 12). One row of truth per
dataset: where it came from, when it was obtained, under what licence, and how
to cite it. **Verify every "TODO / confirm" item before submission** — the
copies on disk are third-party mirrors, not the canonical releases.

Machine that holds the data: local workstation, `C:\Users\pravi\OneDrive\Desktop\DIABETIC_FUNDUS_REASEARCH PAPER\`
Date this log was written: 2026-09-08

---

## 1. APTOS 2019 Blindness Detection

| Field | Value |
|---|---|
| Role in study | **Internal only** — train / internal-val / internal-test (split in Phase 3) |
| On-disk path | `aptos2019-blindness-detection/` (`train.csv`, `train_images/`, `test.csv`, `test_images/`) |
| Images present | 3,662 labelled (`train_images/`) + 1,928 unlabelled Kaggle holdout (`test_images/`) |
| Labels present | `train.csv` — `id_code`, `diagnosis` (0–4). Holdout `test.csv` has **no** `diagnosis` column. |
| Grading scale | International Clinical DR severity: 0 none, 1 mild NPDR, 2 moderate NPDR, 3 severe NPDR, 4 PDR |
| Source URL | https://www.kaggle.com/competitions/aptos2019-blindness-detection/data |
| Origin of images | Aravind Eye Hospital, India; released by the Asia Pacific Tele-Ophthalmology Society (APTOS) |
| Version / date accessed | Kaggle competition data, 2019 release. **TODO: record the exact download date** (before 2026-09-08). |
| Licence | Kaggle competition rules — permitted for **non-commercial academic research**; redistribution of the raw images is not permitted. Link to source, do not upload images to the study repo. **TODO: paste the exact competition-rules clause used.** |
| Citation | Karthik, Maggie, Sohier Dane. *APTOS 2019 Blindness Detection.* Kaggle, 2019. https://kaggle.com/competitions/aptos2019-blindness-detection |
| Also cite (context) | Asia Pacific Tele-Ophthalmology Society. APTOS 2019 Blindness Detection dataset (Aravind Eye Hospital). |

Notes
- The Kaggle **holdout `test_images/` has no public labels** and is **not** used
  anywhere in this study. All internal train/val/test partitions come from the
  3,662 labelled `train.csv` images.

---

## 2. Messidor-2

| Field | Value |
|---|---|
| Role in study | **External test only** — never used for training or hyperparameter tuning |
| On-disk path | Images: `archive/messidor-2/messidor-2/preprocess/` · Labels: `archive/messidor_data.csv` |
| Images present | 1,744 `*_PP.png` (preprocessed) |
| Labels present | `messidor_data.csv` — `id_code` (filename), `diagnosis` (0–4), `adjudicated_dme` (0/1), `adjudicated_gradable` (0/1). All 1,744 rows `adjudicated_gradable == 1`. |
| Grading scale | ICDR 0–4 (adjudicated reference standard) |
| Image source URL | https://www.adcis.net/en/third-party/messidor2/ (Messidor-2 image release, ADCIS / Messidor programme partners) |
| Label source URL | Adjudicated grades — Google Health / Verily, distributed via Kaggle mirror. **TODO: record the exact Kaggle dataset slug + URL + download date this `messidor_data.csv` came from.** The `adjudicated_dme` / `adjudicated_gradable` columns indicate these are the Krause et al. 2018 adjudicated consensus grades, not the original Messidor grades. |
| Version / date accessed | **TODO** |
| Licence (images) | Messidor-2 is released by ADCIS for research use under the Messidor licence agreement; **on-site registration/agreement required**. Redistribution not permitted — link only. **TODO: confirm current wording at adcis.net.** |
| Licence (labels) | Adjudicated grades released by Google for research use. **TODO: confirm licence text of the mirror used.** |
| Citation (database) | Decencière E., Zhang X., Cazuguel G., et al. *Feedback on a publicly distributed image database: the Messidor database.* Image Analysis & Stereology, 2014;33(3):231–234. |
| Citation (Messidor-2 extension) | Abràmoff M.D., Folk J.C., Han D.P., et al. *Automated analysis of retinal images for detection of referable diabetic retinopathy.* JAMA Ophthalmology, 2013;131(3):351–357. |
| Citation (adjudicated grades) | Krause J., Gulshan V., Rahimy E., et al. *Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy.* Ophthalmology, 2018;125(8):1264–1272. |

Notes
- Methodology draft anticipated ~874 images; this mirror is the **full
  Messidor-2 (1,744 preprocessed images)**. Not a problem for an external test
  set — just report the actual n. Confirm whether the `preprocess/` images are
  acceptable as-is or whether the raw release should be re-preprocessed with the
  Phase 3 pipeline for cross-dataset consistency. **Decision pending Phase 3.**

---

## 3. IDRiD — Indian Diabetic Retinopathy Image Dataset (Disease Grading subset)

| Field | Value |
|---|---|
| Role in study | **External test only** — never used for training or hyperparameter tuning |
| On-disk path | Images: `archive (1)/Imagenes/Imagenes/` · Labels: `archive (1)/idrid_labels.csv` |
| Images present | 455 `*.jpg` — 375 named `IDRiD_XXX.jpg` (official train split) + 80 named `IDRiD_XXXtest.jpg` (official test split, renamed to avoid collision) |
| Labels present | `idrid_labels.csv` — positional columns: `id_code`, DR grade (0–4), risk of macular oedema (0–2). Header row is malformed (trailing commas, stray column name); test-split rows carry only 3 fields. Parsed positionally in `scripts/phase2_data_audit.py`. |
| Grading scale | ICDR 0–4 |
| Source URL | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid — project page: https://idrid.grand-challenge.org/ |
| Origin of images | Kasturba Medical College, Manipal, India (Nidek AFC-210, 50° FOV, 4288×2848) |
| Version / date accessed | IEEE Dataport, 2018 release. **TODO: record the exact download date and which mirror `archive (1)` came from.** |
| Licence | **CC BY 4.0** (IEEE Dataport open-access). Redistribution permitted with attribution — but check the study repo policy anyway; default is link-only. |
| Citation (dataset) | Porwal P., Pachade S., Kamble R., et al. *Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research.* Data, 2018;3(3):25. |
| Citation (challenge) | Porwal P., Pachade S., Kokare M., et al. *IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge.* Medical Image Analysis, 2020;59:101561. |
| Citation (Dataport DOI) | Porwal P., et al. *Indian Diabetic Retinopathy Image Dataset (IDRiD).* IEEE Dataport, 2018. doi:10.21227/H25W98 |

Notes / **open issue**
- Official IDRiD Disease Grading subset = **516 images (413 train + 103 test)**.
  This mirror has **455 (375 + 80)** — i.e. **~61 images missing** (38 from the
  train split, 23 from the test split). **Action: either re-download the
  complete Disease Grading subset from IEEE Dataport, or state explicitly in the
  Methods that a 455-image subset of IDRiD was used and why.** Do not leave this
  undocumented.
- IDRiD class mix is very different from APTOS/Messidor-2: majority grade is
  **2 (Moderate), 34%**, grade 0 only 28%. It is an enriched-pathology set. This
  matters for interpreting the external-validation performance drop.

---

## Cross-dataset separation (the core design constraint)

| Dataset | Train | HP tuning | Internal test | External test |
|---|:--:|:--:|:--:|:--:|
| APTOS 2019 | yes | yes | yes | — |
| Messidor-2 | no | no | no | yes |
| IDRiD | no | no | no | yes |

Messidor-2 and IDRiD are opened **once**, in Phase 6, for both models under
identical conditions. Any earlier peeking (including for preprocessing-parameter
choices) invalidates the study.

---

## Outstanding TODOs before the Data Availability statement

- [ ] Record exact download date for each of the three datasets.
- [ ] Identify and record the exact Kaggle URL/slug the `messidor_data.csv`
      adjudicated grades came from; confirm its licence.
- [ ] Confirm current Messidor-2 image licence wording at adcis.net.
- [ ] Resolve the IDRiD 455-vs-516 image-count gap (re-download or document).
- [ ] Paste the exact APTOS/Kaggle competition-rules clause permitting research use.
- [ ] Decide (Phase 3) whether Messidor-2 `preprocess/` images are used as-is or
      re-preprocessed from a raw release for pipeline consistency.
