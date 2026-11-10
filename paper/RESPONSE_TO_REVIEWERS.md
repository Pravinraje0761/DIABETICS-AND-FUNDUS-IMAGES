# Response to reviewer

Manuscript: *Cross-Dataset External Validation of a Retinal Foundation Model for
Diabetic Retinopathy Grading: Performance, Explainability, and Uncertainty*
Revision date: 2026-10-21. Section names refer to `paper/main_standalone.tex`.

We thank the reviewer. The reviewer's central concern was right: the completed
RETFound run **changed the main conclusion** for the full fine-tune. We report
that change openly instead of keeping the earlier framing.

---

## Must fix before submission

### 1. Complete the RETFound full-fine-tuning experiment

**Done.** The pre-specified protocol (≤30 epochs, AdamW, layer-wise LR decay
0.65, drop-path 0.2, label smoothing 0.1, early stopping on APTOS-validation QWK
with patience 7, seed 42) ran to completion on 2026-10-21 without interruption.
Early stopping triggered at epoch 28; the best epoch was 21 (validation QWK
0.939). The training code is unchanged except for an added resume-from-epoch
option, which was **not used** (`full_summary.json`: `resumed_from_epoch: []`).

Every downstream analysis that uses this model was then re-run by one launcher
(`run_retfound_full_clean.bat`). That covers internal and external evaluation,
bootstrap CIs, drop and difference-in-drop, DeLong, calibration and
reliability, temperature scaling (both calibration sources), referable-DR,
OOD detection, single-pass and stochastic-pass uncertainty, saliency and the
rating packet, the CLAHE / colour-transfer / 384-px ablations, and every
figure. All tables, CIs, comparisons and abstract statements were then
regenerated.

The provisional marker macro and every superscript-p marker are removed. The
superseded epoch-6 outputs are kept in `outputs/_archive_provisional_epoch6/`,
and the checkpoint is kept and hashed in `CHECKPOINTS.md`.

**Effect on conclusions (Results, RQ1, Table V; Discussion, "What changed from the
interrupted run").**

| RETFound full fine-tune | epoch 6 (provisional) | completed run |
|---|---|---|
| QWK internal / Messidor-2 / IDRiD | 0.869 / 0.368 / 0.720 | 0.882 / 0.487 / 0.744 |
| ΔDrop vs EfficientNet-B3, Messidor-2 | +0.124 [0.074, 0.175] | **+0.018 [−0.024, 0.058]**, Holm p = 0.79 |
| ΔDrop vs EfficientNet-B3, IDRiD | +0.033 [−0.022, 0.089] | +0.022 [−0.033, 0.083], Holm p = 0.79 |

The paper's conclusion is therefore revised. Before, it said RETFound
generalised **worse**. Now it says fully fine-tuned RETFound generalised
**neither better nor measurably worse** than the CNN, and that RETFound used as
a frozen linear probe generalised worse (+0.076, Holm p = 0.030; +0.105, Holm
p = 0.016). Title, abstract, results, discussion, limitations, conclusion and
the cover letter were rewritten to match.

### 2. Replace the author placeholders

**Done.** Author: Pravin Raj E, independent researcher (no institutional
affiliation), with correspondence e-mail. The "to be supplied" footnote is gone,
and the author-contributions initials are updated. *Optional, pending the
author:* city/country and an ORCID iD, if the target journal requires them.

### 3. Recheck every headline number after the completed run

**Done.**
- **Garbled abstract.** The garbled notation in the PDF abstract came from the
  `\pv{}` superscript-p markers attached to numbers. They are removed. The
  abstract was rewritten and checked in a text extraction of the recompiled PDF:
  it has no stray superscripts.
- **Automated check of the tables against the analysis outputs.**
  - 73 table cells (master table, drop table, ΔDrop, DeLong, calibration) were
    checked by script against `outputs/phase9`, `phase11` and `phase23`.
  - Mismatches are limited to third-decimal rounding of 4-decimal CSV values.
  - The independent recomputation in `phase11` matches 60 of 60 cells
    (max |Δ| = 5×10⁻⁷).
- **Two pre-existing errors found and fixed during this check:**
  - The referable-DR prevalences were hand-entered and wrong (APTOS 40.7 → 40.6 %;
    IDRiD 67.6 → 66.8 %). They are now computed from the manifests in
    `phase11_review_response.py`.
  - The external-evaluation script overwrote the master tables when run for one
    model. This truncated the per-class heatmap. The tables were rebuilt and the
    figure regenerated, and the script now merges rather than overwrites.
- **Compilation.** The manuscript compiles with no undefined references or
  citations and no overfull boxes (21 pages, IEEEtran).

---

## Scientific revisions

### Document the external datasets completely
New subsections in Methods: "Datasets", "Audit of the IDRiD copy" and
"Reference standards". `DATA_PROVENANCE.md` is rewritten.

- **Access date.** 2026-09-08 for all three datasets, from file-system
  evidence.
- **Licences.** APTOS: Kaggle competition rules. Messidor-2: ADCIS terms (no
  redistribution). IDRiD: CC BY 4.0, doi:10.21227/H25W98.
- **Versions.** The canonical release is identified for each dataset. The exact
  slug/version of each Kaggle mirror is flagged as still to be recorded.
- **Messidor-2.** 1,744 of 1,748 images. The 4 absent images are exactly those
  marked ungradable under the adjudicated standard.
- **IDRiD, 455 vs 516 images.**
  - **Which images are missing:** the 61 absent official images (38 train IDs,
    23 test IDs) are listed in `data_splits/idrid_missing_vs_official.csv`.
    Most fall in contiguous ID runs.
  - **Bias in class representation:** grade 0 is under-represented (28.4 % vs
    32.6 %), and referable prevalence is 66.8 % vs 62.6 %.
  - **New finding:** the mirror's labels are not identical to the official
    grades. At least 7 images (≥1.5 %) must carry a different grade, because
    the mirror holds more grade-4 train-ID images than the official set
    contains.
  - **Sensitivity analysis:** re-weighting by grade to the official mix changes
    IDRiD QWK by ≤0.015 and leaves every IDRiD contrast unchanged
    (`outputs/phase23/idrid_reweighted.csv`).
  - Confirming on the official release is listed as the top next experiment.
- **Correction.** An earlier draft wrongly gave IDRiD's origin as Kasturba
  Medical College, Manipal, with a Nidek camera. It is the Eye Clinic in Nanded,
  using a Kowa VX-10α camera, with all subjects dilated.

### Clarify grade comparability
New Methods subsection "Reference standards and their comparability", plus
rows in Table II.
- **APTOS:** one clinician per image; no adjudication described.
- **Messidor-2:** three retina specialists, with full adjudication of every
  disagreement (Krause et al.).
- **IDRiD:** "medical experts"; the number of graders and any adjudication are
  not reported.
- **Consequences, stated in Methods, Results and Limitations:**
  - absolute drops combine image shift with a change of labelling protocol;
  - QWK is not comparable across the two external sets;
  - between-model contrasts hold labels fixed, which is why they are the
    primary inference.

### Address patient-level leakage
New paragraph "Patient- and eye-level linkage" (Methods, Development split),
plus Limitations.
- **What linkage exists:** none of the releases has patient IDs. APTOS IDs are
  opaque hashes, so the APTOS split is image-level, and fellow-eye or repeat
  images may cross it.
- **What was checked:** byte-identical duplicates were checked (41/733), and
  deduplicated re-scoring changes internal QWK by ≤0.006. Non-identical
  fellow-eye images cannot be detected, and this is stated.
- **Messidor-2 eye pairing:** every examination has two eyes, but the pair list
  is not in the local copy.
- **New worst-case analysis:** an m-out-of-n bootstrap with m = n/2 on every
  test set. The full-fine-tune null and the linear probe's IDRiD deficit are
  unaffected. The linear probe's Messidor-2 interval reaches zero
  ([−0.004, 0.157]), and this is reported in the Results.

### Lock the external evaluation
New Methods subsection "Locking the external evaluation", plus a new
`CHECKPOINTS.md`.
- **Git timestamps:** both trainers, and every hyperparameter default, were
  committed (12:28 / 12:44 IST) before the first external scoring (13:16 IST).
- **No later changes:** `git diff` of both trainers between those commits and
  the commits that produced the frozen checkpoints is empty. All defaults are
  byte-identical today.
- **Selection and calibration:** checkpoints were selected by APTOS validation
  QWK only, and no threshold or temperature was fitted on external data.
- **Disclosed external contact:** the data audit and the preprocessing QA.
- **Post-hoc sweeps:** these replaced no primary checkpoint.
- **Checkpoint list:** `CHECKPOINTS.md` lists every model version and
  checkpoint with its SHA-256 hash, and identifies the frozen checkpoint for
  each comparison:
  - EfficientNet-B3 `a5951332…`
  - ResNet-50 `225610ff…`
  - RETFound linear probe `e7795a1b…`
  - RETFound full fine-tune `b1d4a7b2…`
  - superseded epoch-6 checkpoint `4344dd30…`

### Make QWK the clear primary outcome
Methods, Statistical analysis, is rewritten; Table V is now the primary-outcome
table; new script `scripts/phase23_primary_inference.py`.
- **Primary outcome:** QWK is the single primary outcome.
- **Confirmatory family:** four contrasts, ΔDrop for {RETFound full,
  RETFound linear probe} vs EfficientNet-B3 on {Messidor-2, IDRiD}. The
  rationale is given: a claim about RQ1 could rest on any one of the four.
- **Multiplicity adjustment:** the earlier draft used none on the primary
  contrasts. Rather than justify that, we now apply it — Holm-adjusted p
  (Table V) and Bonferroni 98.75 % CIs (text and `outputs/phase23`).
- **Paired CIs:**
  - paired QWK differences (model − baseline) on every dataset;
  - internal-to-external drops, with CIs for every model.
- **Secondary and exploratory analyses:**
  - ResNet-50 is secondary;
  - DeLong referable-DR is a separate, Holm-adjusted secondary family;
  - per-grade tests, per-class sensitivities, calibration, OOD, saliency,
    uncertainty and all ablations are labelled exploratory.
- **Internal comparison corrected:** the paired comparison shows the linear
  probe is slightly but significantly lower than the CNN internally (−0.029
  [−0.054, −0.005]). The earlier claim of internal parity for the linear probe
  is corrected.

### Downgrade the saliency conclusion
- **RQ2 reframed:** it is now an exploratory location question.
- **Methods state what is measured:** only a geometric location proxy ("in-zone" =
  peak and top-decile mass inside the fundus, away from disc and border). The
  zone also contains vessels, the macula and normal retina.
- **Claims removed:** all "lesion-focused" and "clinically plausible" claims,
  from the abstract, results, discussion and conclusion.
- **Figure labels:** panel titles now read "in-zone / out-of-zone (location
  proxy)".
- **Stated finding:** a larger share of RETFound's attribution mass lay on the
  image border on Messidor-2. This may partly reflect the different attribution
  methods.
- **Expert review:** the blinded expert packet remains released and is
  explicitly reported as not completed. No clinician raters were available, so
  the reviewer's first option (blinded expert review) could not be done.

### Use precise uncertainty terminology
Methods, Uncertainty quantification, is rewritten.
- **Primary measure:** single-pass softmax entropy, a point-estimate score.
- **Sensitivity analysis:** called "stochastic-pass" throughout, with what is
  stochastic stated per model — head dropout for the CNNs (Monte-Carlo dropout
  in the sense of Gal & Ghahramani), stochastic depth for the RETFound full
  fine-tune, nothing for the linear probe.
- **Terminology:** "Monte Carlo dropout" no longer labels the whole analysis.
- **Bug found and fixed:** while making this precise we found that the
  stochastic-pass script had rebuilt the linear probe's frozen backbone with
  drop-path 0.2, which it was never trained with.
  - The script is fixed.
  - The affected files are superseded
    (`outputs/phase8/mc_lp_deterministic/README.md`).
  - The corrected sensitivity analysis leaves the linear-probe contrasts
    significant (Holm p = 0.039, 0.016).

### Keep clinical-readiness language restrained
- **Discussion:** "Clinical significance" is now "Implications". It states that
  no result supports screening patients or informing clinical decisions, and
  that clinical use would require prospective evaluation.
- **Removed:** "deployment-ready", "screening tool" and safeguard language,
  from the abstract, discussion and conclusion.

### Complete the reporting and reproducibility package
Reproducibility section: new "Release package" and "Not redistributed, and why"
paragraphs.
- **Code:** all scripts, plus the launcher that produced the completed run and
  refresh.
- **Exact split lists:** APTOS train/val/test, both external manifests, and the
  missing-IDRiD list.
- **Checkpoint identifiers:** `CHECKPOINTS.md` and
  `outputs/checkpoint_sha256.txt`.
- **Preprocessing:** the script plus its logged config.
- **Environment:** new `requirements-lock.txt`, the exact 47-package lock
  (Python 3.14.7, torch 2.11.0+cu128).
- **Reporting checklist:** the TRIPOD+AI checklist is updated (items 4b, 5b,
  6b, 9, 10d, 15, 16, 17, 21e, 24, 25c, 26, 29, 30).
- **Not redistributed, and why:**
  - **Raw images:** the APTOS Kaggle rules and the ADCIS Messidor-2 terms
    prohibit redistribution, and for IDRiD we point to the canonical
    516-image release.
  - **RETFound weights:** gated.
  - **Fine-tuned checkpoints:** identified by hash, available on request for
    non-commercial research.
- *Pending the author:* the public repository URL / Zenodo DOI, which can only
  be minted when the repository is published.

---

## Structural revision

To keep the paper's argument clear after the conclusion changed, the manuscript
was reorganised around its three questions:
- **Introduction:** rewritten to move from problem to known evidence to the
  unresolved gap. It now cites the closest prior work (Hou et al., Ophthalmol
  Sci 2025, the same three datasets) and two recent external-validation studies.
- **Research questions:** RQ2 is now uncertainty and calibration, and RQ3 is the
  exploratory saliency question.
- **Results:** follow the question order, with a roadmap at the start and
  question-led subsection titles.
- **Appendix:** the head-depth sweep detail moved to Appendix A, summarised in
  one Results subsection.
- **Discussion:** opens by answering the questions directly.

## Reviewer concern to watch

The concern proved warranted, and the abstract was not circulated in its old
form. Everything downstream of the RETFound full fine-tune was regenerated
before any statement was rewritten, and the revised conclusion is drawn from
the completed run only.
