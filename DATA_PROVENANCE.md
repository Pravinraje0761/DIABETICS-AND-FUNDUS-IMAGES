# Data Provenance Log

Where each dataset came from, when it was obtained, under what licence, how its
reference grades were produced, and what is known about patient/eye linkage.
The copies on disk are **third-party mirrors, not the canonical releases**;
every difference from the canonical release that could be checked is recorded
below.

Machine that holds the data: local workstation,
`C:\Users\pravi\OneDrive\Desktop\DIABETIC_FUNDUS_REASEARCH PAPER\`
First written: 2026-09-08. Revised for peer review: 2026-10-21.

**Access dates.** All three raw folders (`aptos2019-blindness-detection/`,
`archive/` = Messidor-2, `archive (1)/` = IDRiD) and every image and label file
in them carry a file-system date of **2026-09-08** (UTC 06:01-06:03), the day they
were downloaded and extracted on this machine. 2026-09-08 is therefore recorded
as the access date for all three. Original download URLs of the two Kaggle
mirrors are listed below; their exact Kaggle dataset version numbers were not
recorded at download time (item still open, see end of file).

---

## 1. APTOS 2019 Blindness Detection (development set)

| Field | Value |
|---|---|
| Role in study | **Development only** — train / internal-validation / internal-test (stratified 70/10/20, seed 42) |
| On-disk path | `aptos2019-blindness-detection/` (`train.csv`, `train_images/`, `test.csv`, `test_images/`) |
| Images used | 3,662 labelled images (`train_images/`). The 1,928-image Kaggle holdout (`test_images/`) has no public labels and is not used anywhere. |
| Source | Kaggle competition data, https://www.kaggle.com/competitions/aptos2019-blindness-detection/data (2019 release; competition data are not versioned) |
| Access date | 2026-09-08 |
| Origin | Aravind Eye Hospital, India, released by the Asia Pacific Tele-Ophthalmology Society (APTOS). The competition data page states that the images were gathered from multiple clinics using a variety of cameras over an extended period, and warns of noise in both images and labels. |
| How the reference grade was produced | The competition data page states that **a clinician rated each image** on a 0-4 scale (0 none, 1 mild, 2 moderate, 3 severe, 4 proliferative DR). No second grader, adjudication or grading protocol is described. Treated here as a **single-grader, non-adjudicated** label. |
| Patient / eye linkage | Each examination contributes **two images (left and right eye)**. The official ADCIS examination file (`messidor-2.csv`, columns `left;right`; 874 examinations, 870 with both images gradable and 4 with one) was obtained and is used for all Messidor-2 intervals in the manuscript, which resample whole examinations (`scripts/phase30b_official_pairs.py`, `scripts/phase31_paper_numbers.py`). The file belongs to ADCIS and is not redistributed here (see `.gitignore`); download it with the Messidor-2 release. An earlier reconstruction of the pairs from file order and optic-disc position (`scripts/phase30_messidor_pairs.py`) matched 91% of the official pairs and is kept only as a validation. |
| Licence | Kaggle competition rules: data for non-commercial research and competition use; **redistribution of the images is not permitted**. The study links to the source and does not redistribute images. *Open: paste the operative competition-rules clause.* |
| Citation | Karthik, Maggie, Sohier Dane. *APTOS 2019 Blindness Detection.* Kaggle, 2019. |

---

## 2. Messidor-2 (external test set)

| Field | Value |
|---|---|
| Role in study | **External test only** — never used for training, model selection, early stopping or hyperparameter choice |
| On-disk path | Images: `archive/messidor-2/messidor-2/preprocess/` · Labels: `archive/messidor_data.csv` |
| Canonical release | ADCIS, https://www.adcis.net/en/third-party/messidor2/ — **874 examinations, 1,748 images** (Messidor-Original 529 exams / 1,058 PNG; Messidor-Extension 345 exams / 690 JPG). Each examination = **two macula-centred images, one per eye**; the image-pair list is a spreadsheet in the ADCIS release. |
| Local mirror | **1,744 images** = 1,057 PNG (`*_PP.png`) + 687 JPG (`IM*.JPG`), i.e. 4 fewer than the canonical 1,748 (1 PNG, 3 JPG). `messidor_data.csv` (columns `id_code, diagnosis, adjudicated_dme, adjudicated_gradable`) lists exactly these 1,744 images, all `adjudicated_gradable = 1`. The 4 absent images are therefore the images excluded as **ungradable** under the adjudicated reference standard; this is the standard 1,744-image gradable Messidor-2 evaluation set. |
| Labels source | Krause et al. (2018) adjudicated grades, released by Google via Kaggle (the `adjudicated_*` columns identify them). *Open: record the exact Kaggle slug / version of the mirror.* |
| Access date | 2026-09-08 |
| Camera / acquisition | Messidor-Extension: Topcon TRC NW6 non-mydriatic camera, 45° field, **no pharmacological dilation**, Brest University Hospital, Oct 2009 – Sep 2010 (ADCIS page). Messidor-Original: provided by the Messidor programme partners (three French ophthalmology departments); camera and dilation status not stated on the ADCIS release page. |
| How the reference grade was produced | **Adjudicated consensus.** Three fellowship-trained retina specialists graded independently on the 5-point ICDR scale, then resolved every disagreement in asynchronous and live adjudication sessions (Krause et al. 2018). This is a different, stricter protocol than APTOS's single-clinician label. |
| Patient / eye linkage | Each patient examination contributes **two images (left and right eye)**, so the 1,744 images come from at most 872–874 examinations. The pair spreadsheet is part of the ADCIS distribution but is **not present in the local mirror**, so images could not be grouped by examination here. Consequences: (i) image-level bootstrap CIs on Messidor-2 treat fellow eyes as independent and may be too narrow — a worst-case m-out-of-n (m = n/2) bootstrap is reported in `outputs/phase23/cluster_sensitivity.csv`; (ii) no train/test leakage is possible because Messidor-2 is never used for training. |
| Licence (images) | ADCIS terms: free for research and educational use; copying, redistribution and commercial use prohibited; publications must acknowledge the Messidor programme partners and LaTIM and cite the specified papers. Not redistributed. |
| Licence (labels) | Released by Google for research use with Krause et al. (2018). *Open: confirm the mirror's licence text.* |
| Citations | Decencière et al., Image Anal Stereol 2014;33:231–234. Abràmoff et al., JAMA Ophthalmol 2013;131:351–357. Krause et al., Ophthalmology 2018;125:1264–1272. |

Preprocessing note: the mirror mixes vendor-preprocessed `*_PP.png` and
original `IM*.JPG` frames; the single preprocessing pipeline was applied to all
1,744 images regardless of source form (`docs/phase_reports/PHASE3_REPORT.md`).

---

## 3. IDRiD — Indian Diabetic Retinopathy Image Dataset, Disease-Grading subset (external test set)

| Field | Value |
|---|---|
| Role in study | **External test only** — never used for training, model selection, early stopping or hyperparameter choice |
| On-disk path | Images: `archive (1)/Imagenes/Imagenes/` · Labels: `archive (1)/idrid_labels.csv` |
| Canonical release | IEEE DataPort, doi:10.21227/H25W98 (created 2018-04-24); **516 images = 413 train + 103 test**, CC BY 4.0 (Porwal et al., *Data* 2018). Official per-grade counts (0/1/2/3/4): train 134/20/136/74/49, test 34/5/32/19/13, total **168/25/168/93/62** (label statistics of the 516-image release as mirrored at huggingface.co/datasets/amin-nejad/idrid-disease-grading, retrieved 2026-10-21). |
| Local mirror | **455 images** = 375 `IDRiD_XXX.jpg` (official train IDs) + 80 `IDRiD_XXXtest.jpg` (official test IDs, renamed). The mirror is a third-party Kaggle re-upload (Spanish folder name `Imagenes`, relabelled CSV header `id_code, diagnosis, Risk of macular edema`). *Open: record the exact Kaggle slug.* |
| Access date | 2026-09-08 |
| Origin / acquisition | Eye Clinic, Nanded, Maharashtra, India; Kowa VX-10α camera, 50° field, 4288×2848 px; **all subjects dilated** (tropicamide 0.5%) (Porwal et al. 2018). *(An earlier version of this log and of the manuscript wrongly gave Kasturba Medical College, Manipal and a Nidek camera; corrected.)* |
| How the reference grade was produced | "The medical experts graded the full set of 516 images" on a 0–4 scale based on the International Clinical DR scale (Porwal et al. 2018). The **number of graders and any adjudication procedure are not reported.** |
| Patient / eye linkage | **None available.** File names are sequential image numbers; no patient or eye identifier is released. Never used for training, so no leakage is possible; image-level CIs may be too narrow if a patient contributes both eyes (worst-case check as for Messidor-2). |
| Licence | CC BY 4.0 (data descriptor). Images are nevertheless not redistributed by this study (link only). |
| Citations | Porwal et al., *Data* 2018;3(3):25. Porwal et al., *Med Image Anal* 2020;59:101561. IEEE DataPort doi:10.21227/H25W98. |

### 3a. What is missing from the IDRiD mirror, and could it bias the result?

- **Which images.** 61 of the 516 official images are absent: 38 train IDs and
  23 test IDs, listed in `data_splits/idrid_missing_vs_official.csv`. They are
  clustered in contiguous ID runs (train 138–173 contains 22 of the 38; test
  39–56 and 92–99 contain 18 of the 23), which looks more like an incomplete
  upload than selection by grade, but the cause cannot be established from the
  mirror.
- **Class representation.** Mirror grade counts are 129/22/156/84/64
  (28.4/4.8/34.3/18.5/14.1 %; referable 66.8 %) versus the official
  168/25/168/93/62 (32.6/4.8/32.6/18.0/12.0 %; referable 62.6 %). Grade 0 is
  under-represented by ~4 percentage points.
- **The mirror's labels are not identical to the official labels.** The mirror
  has *more* grade-4 images in the official-train ID range (55) than the official
  train set has in total (49), and more grade-3 images in the test range (20 vs
  19). Removing images cannot create this, so **at least 7 of the 455 mirror
  labels (≥1.5 %) differ from the official grades**; the true number cannot be
  determined without the official per-image label file.
- **Sensitivity analysis.** Re-weighting the 455 images by grade to the official
  516-image mix changes IDRiD QWK by ≤0.015 for every model and leaves every
  IDRiD difference-in-drop conclusion unchanged
  (`outputs/phase23/idrid_reweighted.csv`). This addresses class-mix
  missingness only, not the label discordance.
- **Resolution path (recommended before submission if feasible).** Download the
  516-image release from IEEE DataPort (doi:10.21227/H25W98), re-run
  `phase3_preprocess.py` (IDRiD only), `phase6_external_eval.py`,
  `phase9_stats.py`, `phase11_review_response.py`, `phase19_referable_dr.py`,
  `phase23_primary_inference.py`, and report the official-label result.

---

## Grading-protocol comparability (feeds the Methods / Limitations)

| | APTOS 2019 | Messidor-2 | IDRiD |
|---|---|---|---|
| Scale | 0–4 (ICDR wording) | ICDR 0–4 | 0–4 "based on" ICDR |
| Graders | one clinician per image | 3 retina specialists | "medical experts", number not reported |
| Adjudication | none described | full adjudication of every disagreement | not reported |
| Ungradable images | retained (no gradability label) | excluded (4 of 1,748) | not reported |

Because the three reference standards differ, an internal-to-external QWK drop
mixes **model degradation** with **label-protocol difference**: a model trained
on single-clinician APTOS labels is scored against adjudicated Messidor-2 labels,
and Krause et al. showed that adjudicated and non-adjudicated grades of the
same images can differ materially. This affects every model equally
(same labels, same images), so it does not bias the **between-model** contrasts
that are the primary inference, but it does mean the absolute drop is not a pure
measure of image-domain shift, and that QWK values are not directly comparable
*across* the two external datasets.

---

## Cross-dataset separation (the core design constraint)

| Dataset | Train | Early stopping / HP choice | Internal test | External test |
|---|:--:|:--:|:--:|:--:|
| APTOS 2019 | yes | yes (validation split) | yes | — |
| Messidor-2 | no | no | no | yes |
| IDRiD | no | no | no | yes |

What *did* touch the external images before external scoring: the Phase 2 data
audit (counts, label parsing) and the Phase 3 preprocessing / visual
label-image alignment audit, which applied the already-fixed pipeline. No model,
hyperparameter, checkpoint or threshold decision used external images or labels.
See `CHECKPOINTS.md` for the git-history evidence and the frozen checkpoint list.

---

## Still open (author's records needed)

- [ ] Kaggle slug + version of the Messidor-2 adjudicated-grade mirror, and its licence text.
- [ ] Kaggle slug of the IDRiD mirror (`Imagenes/`, `idrid_labels.csv`).
- [ ] Operative APTOS competition-rules clause on research use / redistribution.
- [ ] (Recommended) official 516-image IDRiD release and official labels — see 3a.
