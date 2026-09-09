# TRIPOD+AI Reporting Checklist

Study: *Cross-Dataset External Validation of a Retinal Foundation Model for
Diabetic Retinopathy Grading: Performance, Explainability, and Uncertainty*

Guideline: Collins GS, Moons KGM, Dhiman P, et al. **TRIPOD+AI statement.** *BMJ*
2024;385:e078378. This study is a **prediction-model development study with
external validation** (development plus geographically distinct external
validation, no model updating).

`main.tex` is compiled with `\linenumbers`; locations below give the section and
paragraph so they remain valid after edits. "SM" = Supplementary Material.

| # | TRIPOD+AI item (abridged) | Reported? | Where |
|---|---|:--:|---|
| **Title & abstract** | | | |
| 1 | Title identifies the study as developing/validating a multivariable prediction model, the target population, and the outcome | Yes | Title |
| 2 | Structured summary: objectives, data, methods, results, conclusions | Yes | Abstract (Background/Objective/Methods/Results/Conclusions) |
| **Introduction** | | | |
| 3a | Background and rationale, including clinical context and why AI is appropriate | Yes | Introduction para 1-2 |
| 3b | Objectives / research questions; generalisation specified as the target | Yes | Introduction, "Research questions" (RQ1-RQ3) |
| **Methods - data** | | | |
| 4a | Data sources, separately for development and evaluation; study design (retrospective, public) | Yes | Methods, Datasets; `DATA_PROVENANCE.md` (SM) |
| 4b | Dates of data collection / access, split into development and evaluation | Partial | Methods, Datasets; **exact access dates to be inserted from `DATA_PROVENANCE.md`** |
| 5a | Study setting (countries, sites, imaging devices) | Yes | Methods, Datasets; Table 1; `DATA_PROVENANCE.md` |
| 5b | Eligibility criteria for images; gradability handling | Yes | Methods, Datasets (Messidor-2 adjudicated-gradable filter; IDRiD 455/516 subset stated) |
| 5c | Whether images form part of a previously reported dataset | Yes | Methods, Datasets (all three public, previously published) |
| **Methods - outcome** | | | |
| 6a | Outcome definition: 5-class ICDR DR severity grade; how/when measured | Yes | Methods, Datasets |
| 6b | Reference standard per dataset; adjudication where applicable | Yes | Methods, Datasets (Messidor-2 = Krause et al. adjudicated consensus; APTOS/IDRiD = release grades) |
| 6c | Blinding of outcome assessors to predictors | N/A (secondary use of pre-graded public data) | Methods, Datasets; Limitations |
| **Methods - predictors / input data** | | | |
| 7a | Definition of model input: preprocessed colour fundus image; all preprocessing steps | Yes | Methods, Preprocessing |
| 7b | Preprocessing applied identically to development and external data, without access to external labels | Yes | Methods, Preprocessing; External validation; `DATA_PROVENANCE.md` "Cross-dataset separation" |
| 8 | Sample size rationale / available sample | Yes | Methods, Datasets; Table 1; Limitations |
| 9 | Missing data handling (ungradable images, incomplete IDRiD mirror) | Yes | Methods, Datasets; Limitations |
| **Methods - model development** | | | |
| 10a | Model type and architecture: EfficientNet-B3, ResNet-50 (ImageNet CNN); RETFound ViT-Large/16 (MAE, colour-fundus pretraining); 5-way head | Yes | Methods, Models and fine-tuning |
| 10b | Predictor handling / feature extraction (end-to-end; RETFound encoder loaded, `fc_norm` reused, fresh head) | Yes | Methods, Models and fine-tuning; `outputs/phase5_retfound/*_weight_load_report.json` (SM) |
| 10c | Training procedure: loss (inverse-frequency-weighted cross-entropy; label smoothing for RETFound), optimiser (AdamW), schedule (warmup + cosine), layer-wise LR decay, drop-path, augmentation, mixed precision, epochs, early stopping on internal-validation QWK | Yes | Methods, Models and fine-tuning; hyperparameter JSONs (SM); Table 3 |
| 10d | Hyperparameter tuning on internal APTOS validation only; external sets untouched | Yes | Methods, Development split; Models; External validation |
| 11 | How predictions are generated (softmax over 5 grades; argmax label; predictive entropy retained) | Yes | Methods, Evaluation metrics; Uncertainty; `scripts/predict.py` |
| 12 | Differences between development and evaluation (none: identical code path; models unchanged; single evaluation run) | Yes | Methods, External validation |
| **Methods - evaluation / analysis** | | | |
| 13a | Performance measures and rationale (imbalance -> primary = quadratic-weighted kappa; secondary = accuracy, macro-F1, balanced accuracy, per-class sensitivity/specificity, one-vs-rest AUC; internal->external drop) | Yes | Methods, Evaluation metrics |
| 13b | Model comparison: RETFound vs. ImageNet CNN under a matched protocol; primary contrast = difference in internal->external QWK drop | Yes | Methods, Evaluation metrics; Statistical analysis; Results, RQ1 |
| 14 | Model calibration assessment | Partial | Predictive-entropy / selective-prediction analysis (Methods, Uncertainty); formal reliability diagrams / ECE flagged as future work in Limitations |
| 15 | Model uncertainty / instability quantification (bootstrap CIs; MC-dropout predictive entropy) | Yes | Methods, Uncertainty; Statistical analysis |
| 16 | Explainability method and its pre-specified evaluation rubric | Yes | Methods, Explainability; `scripts/phase7_explainability.py` |
| 17 | Statistical methods: bootstrap (B, seed, percentile), DeLong paired-AUC test, alpha, software | Yes | Methods, Statistical analysis; `outputs/phase9/stats_methods_paragraph.md` (SM) |
| 18 | Fairness / subgroup analysis | Partial | Per-severity-grade and per-dataset (population/device) results reported; demographic subgroup labels absent from the public datasets - stated in Limitations |
| **Methods - other** | | | |
| 19 | Reporting-guideline adherence stated | Yes | Methods, opening sentence (TRIPOD+AI); this checklist (SM) |
| 20 | Ethical approval / consent / data governance | Yes | Declarations, Ethics |
| **Open science** | | | |
| 21a | Funding | Yes | Declarations, Funding |
| 21b | Conflicts of interest | Yes | Declarations, Conflicts of Interest |
| 21c | Protocol / pre-registration availability | Partial | Analysis plan fixed a priori in `../option_A_methodology_1.txt` (released with the code); not registered on a public registry |
| 21d | Data availability | Yes | Declarations, Data and Code Availability; `DATA_PROVENANCE.md` |
| 21e | Code availability | Yes | Declarations, Data and Code Availability (repository DOI); all `scripts/` released |
| **Results** | | | |
| 22 | Image flow: counts available, gradable, analysed per dataset | Yes | Table 1; Results, Internal performance; External validation |
| 23 | Image characteristics incl. outcome distribution | Yes | Table 1 (per-grade share per dataset) |
| 24 | Model specification: final architecture(s), key hyperparameters, where weights/config are archived | Yes | Methods, Models; Table 3; hyperparameter JSONs + checkpoints (SM / repo) |
| 25a | Overall performance, development (internal APTOS test) with 95% CI | Yes | Table 2; Results, Internal performance; `outputs/phase9/bootstrap_ci.csv` |
| 25b | Overall performance, external (Messidor-2, IDRiD) with 95% CI | Yes | Table 2; Results, External validation; Figure 2 |
| 25c | Model comparison and internal->external drop with CI on the drop and on the between-model difference in drop | Yes | Table 3; Results, RQ1; `outputs/phase9/drop_ci.csv`, `drop_difference_ci.csv` |
| 26 | Explainability results against the rubric; uncertainty results (error detection, selective prediction) | Yes | Results, Explainability; Uncertainty; Figures 3-4 |
| 27 | Post-hoc / additional analyses, flagged as such | Yes | Results / Discussion (per-class advantages; secondary per-grade DeLong tests marked exploratory) |
| **Discussion** | | | |
| 28 | Interpretation in context, overall and vs. prior evidence | Yes | Discussion: "Performance", "Nature of the shift", "Comparison with the literature" |
| 29 | Limitations (retrospective; public data; IDRiD subset; single foundation model / single task; automated plausibility proxy; external precision for rare grades; calibration not formally assessed) | Yes | Discussion, Limitations |
| 30 | Usability / clinical implications; recommended safeguards | Yes | Discussion, "Clinical and practical implications"; Conclusions |
| 31 | Future work | Yes | Discussion, Limitations / implications (second foundation model, prospective validation, formal calibration, blinded expert rating) |

## Items requiring author input before submission

- **4b / access dates** - insert the exact dataset download dates into `DATA_PROVENANCE.md` and cross-reference here.
- **21a Funding**, **21b COI**, **21c** registry status, **Author Contributions (CRediT)**, **Use of AI tools** - complete in `main.tex` Declarations.
- **20 Ethics** - replace the placeholder with the target journal's exact required wording for secondary use of fully public, de-identified data.
- Decide whether to add a formal **calibration** analysis (reliability diagram + ECE) now or cite as future work (item 14).
