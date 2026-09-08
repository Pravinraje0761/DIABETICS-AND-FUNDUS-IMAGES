# Q1 Journal Submission Runbook

Covers methodology Phases 10–13. Work top to bottom. Nothing here can be
finished until Phases 7–9 are complete and the RETFound full fine-tune has been
re-run to completion (see `../PHASE5_REPORT.md` section 4).

---

## 0. Blocking items before writing is "done"

| Item | Owner | Status |
|---|---|---|
| RETFound full fine-tune re-run to completion in a clean env (not OneDrive) | you | **open** — see note A |
| Phase 7 — Grad-CAM / attention maps + plausibility rating | pipeline | not started |
| Phase 8 — MC-dropout uncertainty + selective prediction | pipeline | not started |
| Phase 9 — bootstrap 95% CIs + DeLong tests | pipeline | not started |
| Replace every `\todo{}` and `\prov{}` in `main.tex` | you + pipeline | open |
| Resolve IDRiD 455/516 image count (re-download or document) | you | open (from Phase 2) |
| Fill provenance TODOs in `../DATA_PROVENANCE.md` (access dates, licences) | you | open |
| Revoke the Hugging Face token pasted earlier in the session | you | **do now** |

**Note A — completing the RETFound run.** Two independent full-FT runs on this
machine slowed from ~90 s/epoch to 400–900 s/epoch after epoch 2, even with
checkpoints moved off OneDrive. Root cause not fully isolated (OneDrive
background scan of `.venv`/`outputs`, Windows Defender real-time scan, or laptop
GPU thermal throttling under sustained ViT-L load are the candidates). Options,
best first:
1. Run on **Google Colab / Kaggle** (free T4/L4). Upload `data_processed/`,
   `data_splits/`, and the RETFound `.pth`; `pip install -r requirements.txt`;
   run `scripts/train_retfound.py --mode full --ckpt-dir /content/ckpts`.
2. **Move the whole project** to `C:\dr_foundation_study\` (outside OneDrive),
   add a Defender exclusion for that folder, ensure good cooling, re-run.
3. Cap it: `--epochs 12` still exceeds the epoch-6 checkpoint currently used and
   completes before the slowdown compounds badly.

After the run: `python scripts/predict.py --checkpoint <new full_best.pt>
--manifest data_splits/aptos_manifest.csv --split test --out-prefix
outputs/phase5_retfound/full_internal_test`, then re-run
`scripts/phase6_external_eval.py`, then update Tables 1–3 and every `\prov{}` in
`main.tex`.

---

## 1. Target journal (Phase 13)

Primary: **Diagnostics** (MDPI) — Q1 in *Clinical Medicine* / *Medicine
(miscellaneous)* in recent JCR/Scopus; ~2–3 week first decision; open access.
Backup: **Journal of Imaging** (MDPI). Higher-impact alternatives if you want to
aim up: *Ophthalmology Science*, *npj Digital Medicine*, *Translational Vision
Science & Technology (TVST)*, *Medical Image Analysis* (methods-heavy). A clean
negative external-validation result with code + provenance is a reasonable fit
for TVST or Ophthalmology Science.

Before committing:
- [ ] Download the **current** author guidelines + LaTeX template from the
  journal site. For MDPI: the `mdpi.cls` package + `Definitions/mdpi.bst`.
- [ ] Confirm word limit, structured-vs-unstructured abstract, reference style,
  figure count/format (DPI, TIFF/EPS/PDF, RGB vs CMYK), and whether a graphical
  abstract is required.
- [ ] Confirm the **APC** on the journal's own page and arrange funding/waiver
  **before** submission.
- [ ] Check for an open **Special Issue** on AI/ML in medical/ophthalmic imaging
  (mention it in the cover letter if so).

## 2. Port `main.tex` to the journal class

- [ ] Swap `\documentclass{article}` for the journal class; move author/affil
  block to its macros; move keywords to its field.
- [ ] Convert `\citep`/`natbib` to the journal's citation command if different.
- [ ] Compile clean (no undefined refs, no overfull boxes into the margin).
- [ ] Figures as separate files at required DPI; check they are legible in
  greyscale.

## 3. Reporting checklist (Phase 12)

- [ ] Complete **TRIPOD+AI** (template below). MDPI/Diagnostics accepts it;
  some reviewers prefer **CLAIM** for imaging-AI — fill whichever the journal
  names, or both.
- [ ] Cite the checklist in Methods (already done in `main.tex`) and upload the
  filled checklist as a supplementary file with page/line references.

## 4. Reproducibility package (Phase 12)

- [ ] Public repo (GitHub + Zenodo DOI). Include: all `scripts/`, `data_splits/`
  (the id-list CSVs), `outputs/phase6/*_preds.csv`, `requirements.txt`, a README
  with exact run order, `DATA_PROVENANCE.md`. **Do not** commit raw images,
  `*.pt`, `weights/`, or `data_processed/` (already gitignored).
- [ ] Tag a release; mint the DOI; put the DOI in the Data/Code Availability
  statement (`main.tex` availability section).
- [ ] `environment` note: Python 3.14, torch 2.11.0+cu128, exact package
  versions in `requirements.txt`.

## 5. Statements to finalise in `main.tex`

- [ ] **Ethics** — public, de-identified, previously consented; no new
  human-subjects data. Use the journal's exact required wording for secondary
  use of public data.
- [ ] **Data & Code Availability** — repo DOI + the three dataset source links
  with licences.
- [ ] **Funding**, **Conflicts of Interest**, **Author Contributions** (CRediT),
  **Use of AI tools** disclosure (this manuscript's code and drafting were
  assisted by an AI coding agent — disclose per journal policy).

## 6. Submission package

- [ ] Manuscript (journal class, line-numbered).
- [ ] Figures (separate, correct format/DPI).
- [ ] Supplementary: TRIPOD+AI checklist, `DATA_PROVENANCE.md`, per-class
  results table, hyperparameter tables.
- [ ] Cover letter (template below).
- [ ] Suggested reviewers (3–5, no COI): pick from authors of RETFound,
  DR-AI external-validation, and reporting-guideline literature.
- [ ] ORCIDs for all authors; corresponding author details.
- [ ] Submit; record manuscript ID + expected first-decision date.

## 7. After submission (Phase 14)

- [ ] If desk-rejected: read the editor's stated reason, fix, submit to the
  backup journal within days — the package is already built.
- [ ] If reviewed: use the wait to run any analysis you're unsure of and draft
  the response template (below).
- [ ] On acceptance: check the proof numbers against `outputs/phase6/` and the
  submitted tables — transcription errors in typesetting are common.

---

## Cover letter — template

> Dear Editors,
>
> Please consider our manuscript, *"Cross-Dataset External Validation of a
> Retinal Foundation Model for Diabetic Retinopathy Grading: Performance,
> Explainability, and Uncertainty"*, for publication in *[Journal]*.
>
> Retinal foundation models are increasingly proposed for clinical retinal image
> analysis, but independent external validation for diabetic retinopathy grading
> is limited. Under a single, fully matched preprocessing / fine-tuning /
> evaluation protocol, we compared a fine-tuned retinal foundation model
> (RETFound) with an ImageNet-pretrained CNN on two external datasets
> (Messidor-2, IDRiD). Our principal finding is that RETFound did **not**
> generalise better than the CNN baseline; all models lost substantial ordinal
> agreement on external data. We add matched explainability and predictive-
> uncertainty analyses and release full code, split definitions, and per-image
> predictions.
>
> We believe this is relevant to *[Journal]*'s readership because [it directly
> tests a widely promoted claim / it fits the Special Issue on AI in ophthalmic
> imaging / it exemplifies TRIPOD+AI-compliant external validation]. The work is
> original, not under consideration elsewhere, and all authors approve
> submission. We declare [no conflicts of interest]. Suggested reviewers:
> [names + affiliations + emails].
>
> Sincerely, [Corresponding author, on behalf of all authors]

---

## TRIPOD+AI checklist — fill-in stub

For each item give the manuscript **section and line/para**. `main.tex` is
line-numbered (`\linenumbers`).

| # | Item (abbrev.) | Where addressed |
|---|---|---|
| 1 | Title identifies study as prediction-model development + external validation | Title |
| 2 | Structured abstract | Abstract |
| 3 | Background & rationale; objectives | Introduction |
| 4 | Data sources; eligibility; setting; dates | Methods, Datasets; `DATA_PROVENANCE.md` |
| 5 | Sample size / n per dataset | Table 1 |
| 6 | Outcome: 5-class ICDR grade; reference standard per dataset | Methods, Datasets |
| 7 | Predictors: preprocessed fundus image | Methods, Preprocessing |
| 8 | Missing data / gradability handling | Methods, Datasets (Messidor-2 gradable filter; IDRiD subset) |
| 9 | Model: EfficientNet-B3 / ResNet-50 / RETFound ViT-L; architecture, pretraining | Methods, Models |
| 10 | Training: split, seed, loss, optimiser, augmentation, early stopping | Methods, Split + Models; hyperparam JSONs |
| 11 | Hyperparameter selection (internal validation only) | Methods, Models |
| 12 | Evaluation metrics + rationale (imbalance -> QWK) | Methods, Metrics |
| 13 | External validation: unmodified models, single evaluation | Methods, External validation |
| 14 | Explainability method + plausibility rubric | Methods, Explainability |
| 15 | Uncertainty method | Methods, Uncertainty |
| 16 | Statistical methods: bootstrap CIs, DeLong, alpha | Methods, Statistical analysis |
| 17 | Participants flow / dataset counts | Table 1; Results |
| 18 | Model performance internal + external, with CIs | Tables 2–3; Fig 2 |
| 19 | Model comparison (baseline vs RETFound), drop analysis | Results, External; Table 3 |
| 20 | Explainability + uncertainty results | Results, Explainability + Uncertainty; Figs 3–4 |
| 21 | Limitations (retrospective, public, subset, truncated run, single rater) | Discussion, Limitations |
| 22 | Interpretation vs prior evidence | Discussion |
| 23 | Data availability | Declarations |
| 24 | Code availability | Declarations |
| 25 | Funding | Declarations |
| 26 | Conflicts of interest | Declarations |
| 27 | Ethics / consent | Declarations |
| 28 | AI-assistance disclosure | Declarations |

---

## Response-to-reviewers — template

> We thank the reviewers for their careful reading. Our point-by-point responses
> follow; reviewer comments are in *italics*, our responses in plain text, and
> manuscript changes are quoted with their new line numbers.
>
> **Reviewer 1**
>
> *Comment 1.1 — [paste].*
> Response: [what we did]. Changes: *"[new text]"* (p. X, ll. Y–Z).
>
> **Anticipated major comments and prepared responses**
> - *"The full fine-tune was truncated."* — We re-ran RETFound full fine-tuning
>   to completion in [environment]; Tables 2–3 and Figure 2 now reflect the
>   completed run (previously provisional). The conclusion is unchanged and is
>   corroborated by the linear-probe result, which always ran to completion.
> - *"Only one foundation model / one task."* — Acknowledged in Limitations;
>   [either add a second checkpoint / task, or scope the claim explicitly].
> - *"Single-rater explainability."* — [added a second blinded rater with
>   Cohen's kappa = ... / retained as a stated limitation].
> - *"IDRiD 455 vs 516."* — [re-ran on the full 516 / documented the subset and
>   showed results are stable].
> - *"No calibration analysis."* — [added reliability diagrams + ECE, or state
>   as future work].
