# Phase 2 — Data Acquisition: Completion Report

**Status:** complete, with two documented follow-ups (IDRiD image count, mirror
metadata). **Date:** 2026-09-08.

Methodology reference: `../option_A_methodology_1.txt`, Phase 2.
Regenerate the numbers below with:
`.venv/Scripts/python.exe scripts/phase2_data_audit.py`

---

## Checklist against the methodology

| Phase 2 task | Status | Where |
|---|---|---|
| Download APTOS 2019 (~3,662 labelled, 5-class) | done, on disk | `../aptos2019-blindness-detection/` — 3,662 labelled + 1,928 unlabelled holdout |
| Download Messidor-2 + confirm **grade labels** present | done, on disk | `../archive/` — 1,744 images + `messidor_data.csv` with adjudicated DR grades |
| Download IDRiD Disease Grading subset | partial | `../archive (1)/` — 455 images + labels (official subset is 516; see follow-up 1) |
| Record source URL / version / licence / citation per dataset | done | `DATA_PROVENANCE.md` (some fields flagged TODO — mirror metadata) |
| Check class balance per dataset | done | this report §2, `outputs/phase2_class_distribution.csv`, `outputs/phase2_class_distribution.png` |
| Fix APTOS = internal only; Messidor-2 + IDRiD = external test only | documented | `DATA_PROVENANCE.md` separation table; enforced by Phase 3 split script |

**Deliverable produced:** three raw datasets on disk (done); data provenance log
(`DATA_PROVENANCE.md`); confirmed class distributions (below).

---

## 1. File-level integrity (labels vs images)

All three datasets are internally consistent — every label row has its image and
every image has a label. No duplicate IDs, no unexpected label values.

| Dataset | Label rows | Image files | Labels w/o image | Images w/o label | Dup IDs | Bad label values |
|---|--:|--:|--:|--:|--:|--:|
| APTOS 2019 (`train.csv`) | 3,662 | 3,662 | 0 | 0 | 0 | 0 |
| Messidor-2 | 1,744 | 1,744 | 0 | 0 | 0 | 0 |
| IDRiD (Disease Grading) | 455 | 455 | 0 | 0 | 0 | 0 |

- **APTOS `test_images/` (1,928 images)** has no public labels — Kaggle
  competition holdout. **Not used anywhere in this study.** The internal
  train/val/test split (Phase 3) is carved entirely from the 3,662 labelled
  `train.csv` images.
- **Messidor-2**: all 1,744 rows are `adjudicated_gradable == 1`, so no
  gradability filtering is needed. 151 rows are `adjudicated_dme == 1` (macular
  oedema) — recorded but not an endpoint of this study.
- **IDRiD**: the on-disk folder merges the official train split
  (`IDRiD_001..`, 375 images) and test split (`IDRiD_001test..`, 80 images).
  Both are used purely as held-out external evaluation data, so the original
  split is irrelevant here — but the total (455) is short of the official 516.

---

## 2. DR-grade class distribution

Grades: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative.

### APTOS 2019 — internal (all 3,662 labelled)
| 0 | 1 | 2 | 3 | 4 | majority | max/min |
|--:|--:|--:|--:|--:|:--:|:--:|
| 1,805 (49.3%) | 370 (10.1%) | 999 (27.3%) | 193 (5.3%) | 295 (8.1%) | 0 | 9.35 |

### Messidor-2 — external (all 1,744, all gradable)
| 0 | 1 | 2 | 3 | 4 | majority | max/min |
|--:|--:|--:|--:|--:|:--:|:--:|
| 1,017 (58.3%) | 270 (15.5%) | 347 (19.9%) | 75 (4.3%) | 35 (2.0%) | 0 | 29.06 |

### IDRiD — external (all 455)
| 0 | 1 | 2 | 3 | 4 | majority | max/min |
|--:|--:|--:|--:|--:|:--:|:--:|
| 129 (28.4%) | 22 (4.8%) | 156 (34.3%) | 84 (18.5%) | 64 (14.1%) | **2** | 7.09 |

### What this means for the study

1. **All three sets are imbalanced, and imbalanced differently.**
   - APTOS and Messidor-2 are dominated by grade 0 (~49% / ~58%).
   - IDRiD is enriched for pathology — grade 0 is only 28% and the modal class
     is grade 2. IDRiD also has the sparsest grade 1 (n = 22).
   - Messidor-2 grade 4 is extremely rare (n = 35, 2%); IDRiD grade 1 is rare
     (n = 22). Per-class external metrics for those cells will have wide CIs —
     plan to report them with the Phase 9 bootstrap intervals and not
     over-interpret point estimates.

2. **Metric choice is now locked in** (as the methodology anticipated):
   primary = **quadratic-weighted kappa** + **per-class sensitivity/specificity**;
   secondary = macro-F1, one-vs-rest AUC. **Raw accuracy is reported but not
   headline** — a grade-0-only predictor scores 49% / 58% / 28% "accuracy" on the
   three sets with zero clinical value.

3. **The APTOS to IDRiD shift is not just a domain shift, it is a prevalence
   shift.** When reporting the internal-to-external performance drop, separate
   "the model sees different images" from "the model sees a different class
   mix". Consider reporting balanced accuracy / per-class recall alongside the
   raw drop so the headline finding is not confounded by IDRiD's base rates.

4. **Stratified splitting is mandatory in Phase 3** — with grade 3 at 5.3% of
   APTOS (193 images), an unstratified 70/10/20 split could leave the internal
   validation set with ~19 grade-3 images. Stratify by grade, fix the seed,
   save the split CSVs.

---

## 3. Follow-ups (not blockers for Phase 3, must close before submission)

1. **IDRiD is 455 images, not the official 516.** 38 missing from the train
   split, 23 from the test split. Either re-download the complete Disease
   Grading subset from IEEE Dataport, or explicitly document in Methods that a
   455-image subset was used. Tracked in `DATA_PROVENANCE.md`.
2. **Mirror metadata for the Data Availability statement.** Exact download
   dates, the Kaggle slug behind `messidor_data.csv`, and current licence
   wording for Messidor-2 images all still need to be recorded. Tracked as
   TODOs in `DATA_PROVENANCE.md`.
3. **Preprocessing consistency decision (Phase 3).** Messidor-2 images on disk
   are already preprocessed (`*_PP.png`). Decide whether to accept them as-is
   or re-run the single Phase 3 pipeline from a raw release, so all three
   datasets pass through identical preprocessing (a stated methods point).

---

## Artefacts produced in Phase 2

| Path | Contents |
|---|---|
| `scripts/phase2_data_audit.py` | Re-runnable audit: integrity checks + class distribution + plot |
| `DATA_PROVENANCE.md` | Source, licence, citation, access notes per dataset + separation table |
| `PHASE2_REPORT.md` | This report |
| `logs/phase2_audit_report.md` | Auto-generated raw audit tables |
| `outputs/phase2_audit.json` | Machine-readable full audit result |
| `outputs/phase2_class_distribution.csv` | Long-format grade counts (dataset, subset, grade, count, pct) |
| `outputs/phase2_class_distribution.png` | Grouped bar chart of grade share by dataset |
| `requirements.txt` | Pinned Python environment |
