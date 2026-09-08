# Phase 6 — External Validation: Completion Report

**Status:** complete. **Date:** 2026-09-08.
Methodology reference: `../option_A_methodology_1.txt`, Phase 6 — "the core experiment".

Every model was run **unmodified, with no further training**, on APTOS internal
test, Messidor-2 (all 1,744 gradable), and IDRiD (all 455). Identical code path
(`scripts/predict.py`) for all. Raw per-image predictions saved for every
model x dataset for Phases 8-9.

Reproduce: `.venv/Scripts/python.exe scripts/phase6_external_eval.py`

---

## 1. Master results table (the spine of the Results section)

| Model | Dataset | n | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |
|---|---|--:|--:|--:|--:|--:|--:|
| EfficientNet-B3 (baseline) | APTOS-internal | 733 | 0.804 | 0.876 | 0.644 | 0.641 | 0.928 |
| EfficientNet-B3 (baseline) | Messidor-2 | 1744 | 0.626 | 0.499 | 0.360 | 0.353 | 0.775 |
| EfficientNet-B3 (baseline) | IDRiD | 455 | 0.497 | 0.760 | 0.441 | 0.497 | 0.830 |
| ResNet-50 (baseline) | APTOS-internal | 733 | 0.789 | 0.875 | 0.616 | 0.630 | 0.924 |
| ResNet-50 (baseline) | Messidor-2 | 1744 | 0.614 | 0.435 | 0.354 | 0.339 | 0.754 |
| ResNet-50 (baseline) | IDRiD | 455 | 0.448 | 0.707 | 0.403 | 0.477 | 0.841 |
| RETFound linprobe | APTOS-internal | 733 | 0.759 | 0.847 | 0.607 | 0.633 | 0.908 |
| RETFound linprobe | Messidor-2 | 1744 | 0.589 | 0.394 | 0.347 | 0.423 | 0.758 |
| RETFound linprobe | IDRiD | 455 | 0.336 | 0.626 | 0.325 | 0.395 | 0.778 |
| RETFound full (ep6) | APTOS-internal | 733 | 0.769 | 0.869 | 0.616 | 0.629 | 0.915 |
| RETFound full (ep6) | Messidor-2 | 1744 | 0.602 | 0.368 | 0.362 | 0.346 | 0.758 |
| RETFound full (ep6) | IDRiD | 455 | 0.420 | 0.720 | 0.423 | 0.480 | 0.821 |

The 6-row master table the methodology specifies (baseline + RETFound x internal
+ 2 externals) is the EfficientNet-B3 and RETFound-full rows above; the
ResNet-50 and RETFound-linprobe rows are the supporting comparison / ablation.

Full CSVs: `outputs/phase6/master_results.csv`, `master_results_per_class.csv`,
`internal_vs_external_drop.csv`.

---

## 2. Internal -> external performance drop (QWK) — the headline comparison

| Model | Internal | Messidor-2 | drop | IDRiD | drop |
|---|--:|--:|--:|--:|--:|
| EfficientNet-B3 (baseline) | 0.876 | 0.499 | **-0.377** | 0.760 | **-0.116** |
| ResNet-50 (baseline) | 0.875 | 0.435 | -0.440 | 0.707 | -0.167 |
| RETFound linprobe | 0.847 | 0.394 | -0.453 | 0.626 | -0.221 |
| RETFound full (ep6) | 0.869 | 0.368 | **-0.501** | 0.720 | -0.150 |

Accuracy drop, for reference: Messidor-2 -0.16 to -0.18 for all models; IDRiD
-0.31 to -0.35 for all models.

---

## 3. Headline finding

**On these two external datasets, fine-tuning RETFound on APTOS did not produce
better cross-dataset generalisation than an ImageNet-pretrained CNN.**

- On **Messidor-2**, every model's QWK collapses from ~0.87 to 0.37-0.50.
  RETFound-full has the **largest** drop (-0.501); the best external QWK is the
  EfficientNet-B3 baseline (0.499). AUC is essentially tied (~0.75-0.78 for all).
- On **IDRiD**, the QWK drop is smaller for all (-0.12 to -0.22). Again the
  baseline EfficientNet-B3 is best (0.760); RETFound-full (0.720) edges out
  ResNet-50 (0.707) but not the CNN baseline.
- The direction is **consistent across both RETFound variants** — and the
  linprobe run was *not* truncated — so it is not an artefact of the stopped
  full run (though see section 5).

### Why Messidor-2 collapses (all models)

Every model reverts toward majority-class (grade 0) prediction:

| Grade | Eff-B3 sens | ResNet-50 sens | RETFound-full sens | RETFound-linprobe sens |
|---|--:|--:|--:|--:|
| 0 No DR | 0.97 | 0.95 | 0.97 | 0.90 |
| 1 Mild | **0.02** | **0.01** | **0.02** | **0.00** |
| 2 Moderate | 0.22 | 0.20 | 0.08 | 0.18 |
| 3 Severe | 0.13 | 0.28 | 0.24 | 0.41 |
| 4 Proliferative | 0.43 | 0.26 | 0.43 | 0.63 |

Grade-0 specificity on Messidor-2 falls to 0.18-0.32 for all models — they
label almost everything grade 0 or grade 2. This is a domain-shift failure
(different cameras, grading protocol, and the adjudicated reference standard),
not something the foundation-model pretraining corrects.

### Where RETFound does show something

- **IDRiD grade 3 (Severe):** RETFound-full sensitivity 0.56 — the best of any
  model (CNNs 0.25-0.38).
- **Messidor-2 grade 4 (Proliferative):** RETFound-linprobe sensitivity 0.63 —
  best of any model.
- RETFound-linprobe has the highest **balanced accuracy** on Messidor-2 (0.42
  vs 0.34-0.35) despite the lowest QWK — it spreads its errors more evenly
  across grades instead of collapsing to grade 0.

So the foundation model is not useless on shift — it is modestly better at
*not ignoring* the rare severe grades — but on the ordinal agreement metric
(QWK) and overall it does not beat the CNN baseline here.

---

## 4. Interpretation notes for the Discussion

1. **Prevalence shift vs domain shift.** IDRiD's smaller QWK drop but larger
   *accuracy* drop reflects its enriched-pathology class mix (modal grade = 2,
   Phase 2). Messidor-2's QWK collapse is a genuine ordinal-structure failure.
   Report balanced accuracy alongside QWK so the two shifts are not conflated.
2. **A negative result is still a result.** The methodology explicitly frames
   "RETFound generalises better" as a *measured claim to be tested*. Here it is
   not supported. Frame the paper around the measured cross-dataset drop and the
   (lack of) foundation-model advantage, not around a presumed win.
3. **Single-annotator plausibility / no prospective claim** — standard
   limitations still apply (Phase 11).

---

## 5. Limitations that bear on this table

- **RETFound-full was truncated at epoch 7/30** (OneDrive I/O slowdown, see
  `PHASE5_REPORT.md` section 4). Its internal QWK (0.869) was still climbing; a
  completed run could raise both its internal and external numbers. The
  conclusion should be stated as "did not show an advantage **in this
  experiment**" and the caveat repeated. **Re-running full FT to completion
  outside OneDrive is the single most important thing to do before submission.**
- External sets are the full public releases (Messidor-2 1,744; IDRiD 455, a
  455/516 subset — Phase 2 follow-up still open).
- No test-time augmentation or ensembling for any model (deliberate — keeps the
  comparison clean).
- Grade-1 (Mild) sensitivity ~ 0 on Messidor-2 for all models means per-class
  Mild numbers there are not meaningfully estimable; Phase 9 CIs will be wide.

---

## 6. Deliverables produced

| Path | Contents |
|---|---|
| `outputs/phase6/master_results.csv` / `.md` | master table (12 rows: 4 models x 3 datasets) |
| `outputs/phase6/master_results_per_class.csv` | per-class sensitivity/specificity/AUC, every model x dataset |
| `outputs/phase6/internal_vs_external_drop.csv` | internal vs external, per metric, per model, with drop |
| `outputs/phase6/<model>__<dataset>_preds.csv` | raw per-image: id_code, y_true, y_pred, prob_0..4, entropy — **inputs to Phase 8 (uncertainty) and Phase 9 (bootstrap CIs, DeLong)** |
| `outputs/phase6/<model>__<dataset>_metrics.json` | full metric set incl. confusion matrix, per model x dataset |

---

## 7. Next: Phase 9 (stats) can start now

The raw prediction CSVs are everything Phase 9 needs:
- bootstrap 95% CIs on QWK / accuracy / AUC for every cell of the master table
- **DeLong** test comparing baseline vs RETFound AUC on each dataset (paired,
  same test set)
- CI on the internal->external drop difference between the two models

Phase 7 (Grad-CAM / attention) and Phase 8 (MC-dropout) also draw their image
samples from these prediction CSVs (correct vs misclassified).
