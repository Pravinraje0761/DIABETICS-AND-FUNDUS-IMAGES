# TRIPOD+AI Reporting Checklist

Study: *Cross-Dataset External Validation of a Retinal Foundation Model for
Diabetic Retinopathy Grading: Performance, Explainability, and Uncertainty*

Guideline: Collins GS, Moons KGM, Dhiman P, et al. **TRIPOD+AI statement.** *BMJ*
2024;385:e078378. This study is a **prediction-model development study with
external validation** (development plus geographically distinct external
validation, no model updating).

Manuscript: `paper/main_standalone.tex` (IEEEtran; `paper/main.tex`, an earlier
`article`-class draft, is superseded and not the submission target). Locations
below give the section and paragraph so they remain valid after edits, and use
a line-numbered compile (e.g. `\usepackage{lineno}` + `\linenumbers`, or the
target journal's own line-numbering) for the final submitted PDF. "SM" =
Supplementary Material.

| # | TRIPOD+AI item (abridged) | Reported? | Where |
|---|---|:--:|---|
| **Title & abstract** | | | |
| 1 | Title identifies the study as developing/validating a multivariable prediction model, the target population, and the outcome | Yes | Title |
| 2 | Structured summary: objectives, data, methods, results, conclusions | Yes | Abstract (single unstructured paragraph covering objective, data/methods, primary results with effect sizes and CIs, and conclusion, in that order; not divided into headed subsections) |
| **Introduction** | | | |
| 3a | Background and rationale, including clinical context and why AI is appropriate | Yes | Introduction para 1-2 |
| 3b | Objectives / research questions; generalisation specified as the target | Yes | Introduction, "Research questions" (RQ1-RQ3) |
| **Methods - data** | | | |
| 4a | Data sources, separately for development and evaluation; study design (retrospective, public) | Yes | Methods, Datasets; `DATA_PROVENANCE.md` (SM) |
| 4b | Dates of data collection / access, split into development and evaluation | Yes | Methods, Datasets (all three accessed 2026-09-08, file-system evidence); original collection periods as published (Messidor-Extension Oct 2009 - Sep 2010; others not published) in `DATA_PROVENANCE.md` |
| 5a | Study setting (countries, sites, imaging devices) | Yes | Methods, Datasets; Table 1; `DATA_PROVENANCE.md` |
| 5b | Eligibility criteria for images; gradability handling | Yes | Methods, Datasets (Messidor-2: 1744 of 1748, the 4 adjudicated-ungradable excluded); Methods, "Audit of the IDRiD copy" (455 of 516; missing IDs listed in `data_splits/idrid_missing_vs_official.csv`) |
| 5c | Whether images form part of a previously reported dataset | Yes | Methods, Datasets (all three public, previously published) |
| **Methods - outcome** | | | |
| 6a | Outcome definition: 5-class ICDR DR severity grade; how/when measured | Yes | Methods, Datasets |
| 6b | Reference standard per dataset; adjudication where applicable | Yes | Methods, "Reference standards and their comparability" (APTOS one clinician, no adjudication; Messidor-2 3 retina specialists, full adjudication; IDRiD "medical experts", number/adjudication not reported); Table 2 (domain shift); effect on cross-dataset comparisons discussed |
| 6c | Blinding of outcome assessors to predictors | N/A (secondary use of pre-graded public data) | Methods, Datasets; Limitations |
| **Methods - predictors / input data** | | | |
| 7a | Definition of model input: preprocessed colour fundus image; all preprocessing steps | Yes | Methods, Preprocessing |
| 7b | Preprocessing applied identically to development and external data, without access to external labels | Yes | Methods, Preprocessing; External validation; `DATA_PROVENANCE.md` "Cross-dataset separation" |
| 8 | Sample size rationale / available sample | Yes | Methods, Datasets; Table 1; Limitations |
| 9 | Missing data handling (ungradable images, incomplete IDRiD mirror) | Yes | Methods, "Audit of the IDRiD copy" (which images, class-mix effect, >=7 discordant labels); grade-reweighting sensitivity (`outputs/phase23/idrid_reweighted.csv`); Limitations |
| **Methods - model development** | | | |
| 10a | Model type and architecture: EfficientNet-B3, ResNet-50 (ImageNet CNN); RETFound ViT-Large/16 (MAE, colour-fundus pretraining); 5-way head | Yes | Methods, Models and fine-tuning |
| 10b | Predictor handling / feature extraction (end-to-end; RETFound encoder loaded, `fc_norm` reused, fresh head) | Yes | Methods, Models and fine-tuning; `outputs/phase5_retfound/*_weight_load_report.json` (SM) |
| 10c | Training procedure: loss (inverse-frequency-weighted cross-entropy; label smoothing for RETFound), optimiser (AdamW), schedule (warmup + cosine), layer-wise LR decay, drop-path, augmentation, mixed precision, epochs, early stopping on internal-validation QWK | Yes | Methods, Models and fine-tuning; hyperparameter JSONs (SM); Table 3 |
| 10d | Hyperparameter tuning on internal APTOS validation only; external sets untouched | Yes | Methods, "Locking the external evaluation" (git-history evidence); `CHECKPOINTS.md` (frozen checkpoints + SHA-256) |
| 11 | How predictions are generated (softmax over 5 grades; argmax label; predictive entropy retained) | Yes | Methods, Evaluation metrics; Uncertainty; `scripts/predict.py` |
| 12 | Differences between development and evaluation (none: identical code path; models unchanged; single evaluation run) | Yes | Methods, External validation |
| **Methods - evaluation / analysis** | | | |
| 13a | Performance measures and rationale (imbalance -> primary = quadratic-weighted kappa; secondary = accuracy, macro-F1, balanced accuracy, per-class sensitivity/specificity, one-vs-rest AUC; internal->external drop) | Yes | Methods, Evaluation metrics |
| 13b | Model comparison: RETFound vs. ImageNet CNN under a matched protocol; primary contrast = difference in internal->external QWK drop | Yes | Methods, Evaluation metrics; Statistical analysis; Results, RQ1 |
| 14 | Model calibration assessment | Yes | Methods/Results, Calibration (ECE with 95% CI, MCE, multi-class Brier, NLL, reliability diagrams; internal and both external sets); post-hoc temperature-scaling recalibration check, including a calibration-source robustness check |
| 15 | Model uncertainty / instability quantification | Yes | Methods, Uncertainty (primary: single-pass softmax entropy; sensitivity: T=20 stochastic passes - head dropout for CNNs, stochastic depth for RETFound full FT, none for linear probe); bootstrap CIs; training-seed sweep |
| 16 | Explainability method and its evaluation | Partial | Methods, Explainability: Grad-CAM++ / ViT patch-token Grad-CAM evaluated only by a geometric location proxy; no lesion masks or blinded expert grading (packet released, not yet completed); no clinical-plausibility claim made |
| 17 | Statistical methods | Yes | Methods, Statistical analysis: QWK primary; 4-contrast confirmatory family (difference in QWK drop) with Holm and Bonferroni CIs; DeLong secondary (own Holm); per-grade exploratory; worst-case non-independence and IDRiD reweighting sensitivity; `scripts/phase23_primary_inference.py` |
| 18 | Fairness / subgroup analysis | Partial | Per-severity-grade and per-dataset (population/device) results reported; demographic subgroup labels absent from the public datasets - stated in Limitations |
| **Methods - other** | | | |
| 19 | Reporting-guideline adherence stated | Yes | Methods, opening sentence (TRIPOD+AI); this checklist (SM) |
| 20 | Ethical approval / consent / data governance | Yes | Declarations, Ethics |
| **Open science** | | | |
| 21a | Funding | Yes | Declarations, Funding |
| 21b | Conflicts of interest | Yes | Declarations, Conflicts of Interest |
| 21c | Protocol / pre-registration availability | Partial | Analysis plan fixed a priori in `../docs/analysis_plan.txt` (released with the code); not registered on a public registry |
| 21d | Data availability | Yes | Declarations, Data and Code Availability; `DATA_PROVENANCE.md` |
| 21e | Code availability | Yes | Section Reproducibility (release package; what is not redistributed and why); `requirements-lock.txt`; `CHECKPOINTS.md`; repository DOI to be inserted at release |
| **Results** | | | |
| 22 | Image flow: counts available, gradable, analysed per dataset | Yes | Table 1; Results, Internal performance; External validation |
| 23 | Image characteristics incl. outcome distribution | Yes | Table 1 (per-grade share per dataset) |
| 24 | Model specification: final architecture(s), key hyperparameters, where weights/config are archived | Yes | Methods, Models; hyperparameter JSONs; frozen checkpoint per comparison with SHA-256 in `CHECKPOINTS.md` (weights on request; not redistributed for licence reasons) |
| 25a | Overall performance, development (internal APTOS test) with 95% CI | Yes | Table 2; Results, Internal performance; `outputs/phase9/bootstrap_ci.csv` |
| 25b | Overall performance, external (Messidor-2, IDRiD) with 95% CI | Yes | Table 2; Results, External validation; Figure 2 |
| 25c | Model comparison and internal->external drop with CI on the drop and on the between-model difference in drop | Yes | Table (QWK drop) incl. Bonferroni CIs and Holm p; paired QWK differences; `outputs/phase23/` |
| 26 | Explainability results (location proxy only); uncertainty results (error detection, selective prediction) | Yes | Results, Explainability; Uncertainty |
| 27 | Post-hoc / additional analyses, flagged as such | Yes | Results / Discussion (per-class advantages; secondary per-grade DeLong tests marked exploratory; a standalone referable-DR binary secondary task with its own internal-to-external drop; a domain-shift out-of-distribution detector versus entropy) |
| **Discussion** | | | |
| 28 | Interpretation in context, overall and vs. prior evidence | Yes | Discussion: "Performance", "Nature of the shift", "Comparison with the literature" |
| 29 | Limitations (retrospective; public data; IDRiD subset and label discordance; differing reference standards; single foundation model; geometric location proxy instead of expert rating; rare-grade precision; no patient identifiers, Messidor-2 fellow-eye pairing) | Yes | Limitations |
| 30 | Usability / clinical implications | Yes | Discussion, "Implications" (no screening or clinical-decision claim; prospective evaluation required); Conclusions |
| 31 | Future work | Yes | Limitations, "Highest-value next experiments" (a second foundation model; blinded expert rating against IDRiD lesion masks; the full 516-image IDRiD set; prospective validation). Training-time colour/resolution/training-size retraining ablations and RETFound-full loss sensitivity are prepared (data + code released) but not yet executed pending cloud-GPU access (`COLAB_RETRAIN_GUIDE.md`) |

## Items requiring author input before submission

- **21a Funding**, **21b COI**, **21c** registry status, **Author Contributions (CRediT)**, **Use of AI tools** - complete in `main_standalone.tex` Declarations.
- **20 Ethics** - replace the placeholder with the target journal's exact required wording for secondary use of fully public, de-identified data.
