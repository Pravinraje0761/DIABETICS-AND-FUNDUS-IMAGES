# Journal Submission Runbook

_Last refreshed 2026-09-10 (after Phase 7/8/9 extensions + Phase 17 loss
ablation, commit `ced8492`)._

Analysis is complete (Phases 2–17). Work top to bottom. The only non-author
compute item still open is the completed RETFound full fine-tune
(`AUTHOR_TODO.md` §B item 7); everything else here is an author decision or a
mechanical step.

---

## 0. Status of the blocking items

| Item | Owner | Status |
|---|---|---|
| Phase 7 — saliency maps + plausibility proxy + blinded-rating packet/scorer | pipeline | **done** (`outputs/phase7/`) |
| Phase 8 — MC-dropout uncertainty, selective prediction, per-grade breakdown | pipeline | **done** (`outputs/phase8/`) |
| Phase 9 — bootstrap 95% CIs + DeLong + MC-dropout re-run | pipeline | **done** (`outputs/phase9/`, incl. `*_mc.csv`) |
| Phases 10–16 — figures, review-response recompute, leakage, calibration, temperature scaling, ablations, seed sweep | pipeline | **done** (`outputs/phase10..16/`) |
| Phase 17 — class-weighting / loss-function robustness | pipeline | **done** (`outputs/phase17/`) |
| RETFound full fine-tune to completion in a clean env (removes 17 `\pv{}`) | you | **open** — `AUTHOR_TODO.md` §B7 |
| RETFound full fine-tune × 3 extra seeds | you | open — `AUTHOR_TODO.md` §B8 |
| Fill author identity + Declarations brackets in `main_standalone.tex` | you | open — `AUTHOR_TODO.md` §A1–2 |
| Fill provenance facts in `DATA_PROVENANCE.md` | you | open — `AUTHOR_TODO.md` §A3 |
| Public repo + Zenodo DOI | you | open — `AUTHOR_TODO.md` §A4 |
| Revoke the Hugging Face token from Phase 5 | you | **do now** — `AUTHOR_TODO.md` §A5 |
| `git checkout HEAD --` the two v3-clobber files | you | open — `AUTHOR_TODO.md` §A6 |
| Resolve IDRiD 455 vs 516 (documented as a subset; optional re-download) | you | documented in Limitations; optional (`AUTHOR_TODO.md` §D17) |

---

## 1. Target journal

**Primary recommendation: _Translational Vision Science & Technology_ (TVST).**

Rationale — this is the Q1 venue where the paper is most likely to be *accepted*
on its merits rather than desk-rejected for "impact":

- **Q1, ophthalmology.** ARVO's methods/translational journal (2023 JIF ≈ 2.6–3.3,
  Q1 in Ophthalmology by CiteScore); PubMed/MEDLINE, Scopus, Web of Science
  indexed; fully open access.
- **Scope fit.** TVST explicitly welcomes validation studies, algorithm
  assessment, and methodological work, and has published multiple
  DR-AI external-validation and RETFound-adjacent papers. External validation of
  a widely reused model is squarely in scope.
- **No novelty gate.** Unlike *npj Digital Medicine* or *Ophthalmology Science*,
  TVST does not screen on perceived impact/novelty, so a rigorous
  confirmatory/negative result with a full reproducibility package is a normal
  acceptance, not an uphill fight.
- **Reviewer pool.** Retina + imaging-AI researchers who care about exactly this
  question and will engage with the DeLong / drop-CI / calibration machinery.
- **Format.** Flexible (unstructured abstract allowed); the current IEEEtran
  manuscript ports with light effort — see §2. ~4–8 week first decision. APC
  ≈ USD 2.6k with waivers available for demonstrated need.
- **Single-author, consumer-GPU, epoch-6-checkpoint** situation: acceptable at
  TVST with the caveats already stated in Limitations; higher desk-reject risk
  at the pickier venues below.

**Backups, in order:**

1. **_Scientific Reports_** — Q1 (Multidisciplinary Sciences), "technically sound"
   acceptance criterion, explicitly publishes rigorous negative results; highest
   raw acceptance probability. Use if TVST desk-rejects or if you want the
   fastest route to *a* Q1 acceptance. APC ≈ USD 2.7k.
2. **_Ophthalmology Science_** (AAO, open access, Q1) — better profile than TVST;
   accepts if the editors judge the message important, but a null methods result
   carries a real "not impactful enough" desk-reject risk.
3. **_Diagnostics_ / _Journal of Imaging_ (MDPI)** — fast (~2–3 week first
   decision), Q1 by some category metrics, high acceptance; weigh the
   reputational trade-off.
4. **_PLOS Digital Health_** — newer, welcomes reproducibility-focused and null
   results; indexing/JIF still maturing.

**Poor fits (do not target):** IEEE J-BHI / TMI, *Medical Image Analysis* — all
want methodological novelty, which this validation study deliberately does not
claim; *npj Digital Medicine* — impact bar too high for a single-author null
result.

Before committing to any target:
- [ ] Download the **current** author guidelines + template from the journal
  site.
- [ ] Confirm word limit, structured-vs-unstructured abstract, reference style,
  figure count/format (DPI, TIFF/EPS/PDF, RGB vs CMYK), graphical-abstract
  requirement.
- [ ] Confirm the **APC** on the journal's own page; arrange funding/waiver
  **before** submission.
- [ ] Check for an open **Special Issue** on AI/ML in ophthalmic/medical imaging;
  mention it in the cover letter if so.

## 2. Port `main_standalone.tex` to the journal class (if required)

- [ ] TVST / Scientific Reports: a clean standard-LaTeX or Word/PDF submission is
  accepted — minimal porting; just satisfy the length/abstract rules. For MDPI:
  swap `\documentclass[...]{IEEEtran}` for `mdpi.cls`, move
  `\author`/`\thanks`/keywords to the class macros, and switch the bibliography
  style to `Definitions/mdpi.bst`.
- [ ] Keep numbered `\cite`; `references.bib` has 24 entries and all resolve.
- [ ] Compile clean on Overleaf (no undefined refs, no overfull boxes into the
  margin). No local LaTeX toolchain — see `AUTHOR_TODO.md` §E.
- [ ] Figures as separate files at the required DPI; confirm greyscale legibility
  (they are 300-dpi PDF + PNG in `paper/figures/`, 10 files).

## 3. Reporting checklist

- [ ] Complete **TRIPOD+AI** (`paper/TRIPOD-AI_checklist.md`) with section +
  line/paragraph references from the line-numbered compiled PDF. Some ophthalmic
  reviewers also like **CLAIM** — fill whichever the journal names, or both.
- [ ] Methods already cites the checklist; upload the filled checklist as a
  supplementary file.

## 4. Reproducibility package

- [ ] Public repo (GitHub + Zenodo DOI). Include all `scripts/`, `data_splits/`
  (the ID-list CSVs), `outputs/phase6/*_preds.csv` (and the other committed
  `outputs/phaseN/` CSVs/summaries/figures), `requirements.txt`, `README.md` with
  the exact run order, `DATA_PROVENANCE.md`. **Do not** commit raw images,
  `*.pt`, `weights/`, or `data_processed*/` (already gitignored), nor the
  regenerable Phase-7 rating-packet images (`outputs/phase7/rating_packet/img/`,
  also gitignored).
- [ ] Tag a release; mint the DOI; put it in the Code-availability statement.
- [ ] `environment` note: Python 3.14, `torch` 2.11.0+cu128, `timm` 1.0.29,
  `scikit-learn` 1.9.0; full list in `requirements.txt`.

## 5. Statements to finalise in `main_standalone.tex` (Declarations, ~line 1405)

- [ ] **Ethics** — replace the bracketed sentence with the journal's exact
  required wording for secondary use of fully public, de-identified data.
- [ ] **Data & Code Availability** — repo DOI + the three dataset source links
  with licences (facts from `DATA_PROVENANCE.md`).
- [ ] **Funding**, **Competing interests**, **Author contributions** (CRediT),
  **Use of AI tools** — drafted for one author; confirm or replace, and match the
  AI-use sentence to journal policy.
- [ ] Author identity block (`% >>> BEFORE SUBMISSION`, `\thanks{}`): name,
  affiliation, city/country, corresponding email, ORCID.

## 6. Submission package

- [ ] Manuscript (journal class, line-numbered).
- [ ] Figures (separate, correct format/DPI).
- [ ] Supplementary: TRIPOD+AI checklist; `DATA_PROVENANCE.md`; per-class results
  table; hyperparameter JSONs; the extended `outputs/` CSVs listed in the
  manuscript's "Supplementary material" paragraph (bootstrap CIs, per-grade
  DeLong + Holm, calibration/reliability, MC-dropout uncertainty + the MC re-run
  of the stats pipeline, per-grade entropy breakdown, temperature-scaling table,
  dedup-internal check, seed sweep, resolution ablation, loss ablation,
  attention-rollout maps, blinded rating packet).
- [ ] Cover letter (`paper/COVER_LETTER.md`) — fill journal name + reviewer list.
- [ ] Suggested reviewers (3–5, no COI): authors of RETFound, DR-AI external
  validation, and TRIPOD+AI.
- [ ] ORCIDs; corresponding-author details.
- [ ] Submit; record manuscript ID + expected first-decision date.

## 7. After submission

- [ ] If desk-rejected: read the editor's stated reason, fix, submit to the next
  journal in §1 within days — the package is already built.
- [ ] If reviewed: use the wait to run any analysis you are unsure of; draft the
  point-by-point response. Anticipated major comments and the prepared answers
  are in the response template below.
- [ ] On acceptance: check the proof numbers against `outputs/phase6/` and the
  submitted tables — typesetting transcription errors are common.

---

## Cover letter — template

> Dear Editors,
>
> Please consider our manuscript, *"Cross-Dataset External Validation of a
> Retinal Foundation Model for Diabetic Retinopathy Grading: Performance,
> Explainability, and Uncertainty"*, for publication in *[Journal]*.
>
> Retinal foundation models are increasingly reused for clinical retinal image
> analysis and are promoted as improving generalisation, yet independent,
> protocol-matched external validation for diabetic retinopathy grading is
> scarce. Under a single, fully matched preprocessing / fine-tuning / evaluation
> protocol, we compared a fine-tuned retinal foundation model (RETFound) with an
> ImageNet-pretrained CNN on two external datasets (Messidor-2, IDRiD). Our
> principal finding is negative: RETFound did **not** generalise better than the
> CNN baseline and, on the pre-specified contrasts, generalised worse; all models
> lost substantial ordinal agreement on external data. We add matched,
> method-matched explainability and predictive-uncertainty analyses (including a
> per-grade uncertainty breakdown and a Monte-Carlo-dropout re-run of the full
> statistical pipeline), a calibration and temperature-scaling analysis, and an
> extensive sensitivity battery (secondary backbone, adaptation regime, training
> seeds, inference-time CLAHE / colour-transfer / resolution, and loss-function /
> class-weighting choice). We release all code, split image-ID lists, per-image
> predictions, and a blinded expert-rating packet, so every number can be
> regenerated from the public data.
>
> We believe this is relevant to *[Journal]*'s readership because it directly
> tests a widely promoted claim with a TRIPOD+AI-compliant external-validation
> design [/ it fits the Special Issue on AI in ophthalmic imaging]. The work is
> original, not under consideration elsewhere, and all authors approve
> submission. We declare [no conflicts of interest]. Suggested reviewers:
> [names + affiliations + emails].
>
> Sincerely, [Corresponding author, on behalf of all authors]

---

## Response-to-reviewers — template

> We thank the reviewers for their careful reading. Reviewer comments are in
> *italics*, our responses in plain text, manuscript changes quoted with new
> line numbers.
>
> **Anticipated major comments and prepared responses**
> - *"The full fine-tune was truncated at epoch 6."* — We re-ran RETFound full
>   fine-tuning to its pre-specified early-stopping convergence in [environment];
>   Tables 2–3 and Figures 2/… now reflect the completed run (previously
>   superscript-p). The conclusion is unchanged and is corroborated by the
>   linear-probe row (which always ran to completion), the four-seed sweep, the
>   CLAHE / colour-transfer / 224-vs-384 resolution ablations, and the
>   loss-function / class-weighting ablation.
> - *"Only one foundation model / one task."* — Acknowledged and scoped
>   explicitly in Limitations; [add a second checkpoint or task, or retain the
>   scoped claim].
> - *"Single-rater / automated explainability proxy."* — The proxy is a
>   transparent location screen, stated as such. We release a complete blinded
>   two-rater packet, rubric, and scorer (`scripts/phase7_rating_packet.py`);
>   [we have now added ratings from N ophthalmologists, Cohen's κ = … / we retain
>   the proxy with the stated limitation].
> - *"IDRiD 455 vs 516."* — The mirror used ships 455/516; it is an external test
>   set, so the subset cannot bias development, and it spans all five grades with
>   the full release's class mix. [We re-ran on the full 516 and the drop is
>   unchanged / we document the subset and show stability.]
> - *"Class weighting / loss choice drives the result."* — Addressed in the
>   Phase 17 ablation (Limitations, "Sensitivity analyses"): re-training the CNN
>   baseline and the RETFound linear probe with plain cross-entropy and with
>   focal loss leaves the internal→external drop ranking and the RETFound–CNN gap
>   direction unchanged.
> - *"No calibration analysis."* — Full ECE/MCE/Brier/NLL + reliability diagrams
>   per model per dataset, plus post-hoc temperature scaling on a dedicated
>   in-distribution split (Results, Calibration).
