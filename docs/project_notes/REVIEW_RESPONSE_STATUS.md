# Peer-review checklist -- what was done, what cannot be done here, what needs you

Primary manuscript: **`paper/main_standalone.tex`** (IEEEtran, 10 figures, 8 tables).
Integrity: every `\ref` resolves, every `\cite` key is in `references.bib`.

## DONE this session

| Checklist item | What was done | Artefact |
|---|---|---|
| Verify all numerical results | Every headline metric recomputed from prediction files by an independent script; **all 60 cells match to <5e-4** | `outputs/phase11/verification.csv` |
| Multiplicity / testing hierarchy | Holm-Bonferroni over the 9-comparison DeLong family added; **all external significance claims survive** | `outputs/phase11/multiplicity.csv`; Methods + stats para |
| ECE / reliability / Brier, internal vs external | Full calibration table + reliability diagrams; **new finding: RETFound better-calibrated under shift (ECE 0.10-0.16 vs CNN 0.26-0.29)** | `outputs/phase11/calibration.csv`, `reliability_*.csv`; new Results subsec + Table + Fig |
| Domain-shift analysis (camera/FOV/res/population/prevalence/reference std/quality) | 11-axis comparison table | `outputs/phase11/domain_shift.md`; Table in Methods |
| Duplicate / leakage check | MD5 byte-identical check of all 5861 images: **0 cross-dataset duplicates** (external sets clean); 41/733 internal-test images have a dup in APTOS train/val (known APTOS property) -- dedup shifts internal QWK by <=0.006, no comparison changes | `outputs/phase12/{leakage_summary.md,exact_duplicates.csv,dedup_internal_metrics.csv}` |
| Training-data accounting | Exact per-split per-grade counts | `outputs/phase11/training_accounting.csv`; Table in Methods |
| Per-class metrics with 95% CI | Sensitivity/specificity/precision/F1/support + CIs, all models x datasets | `outputs/phase11/per_class_ci.csv`; forest figure |
| Confusion matrices, both externals | fig4; all 4 models in `master_results_per_class.csv` | `paper/figures/fig_confusion.pdf` |
| ROC/PR for referable DR | Curves, all models x datasets | `outputs/phase11/referable_roc_pr.csv`; `fig_roc_pr.pdf` |
| Explainability: more images | Re-run at 40 correct + 25 incorrect / model / dataset (was 12+8) | `scripts/phase7_explainability.py --n-correct 40 --n-incorrect 25` (re-running; overwrites `outputs/phase7/`) |
| Explainability wording | "lesion-focused" -> "concentrated in the lesion-eligible retinal region (a location proxy, not lesion-mask overlap)" | main_standalone.tex Results + Limitations |
| "Why RETFound fails" | Mechanistic paragraph: domain-shift table + border-shifted saliency + calibration/label-smoothing -> "large ViT fitted APTOS acquisition statistics"; controlled ablation = follow-up #1 | Discussion |
| Novelty statement | "Gap and novelty" paragraph in Intro | main_standalone.tex |
| Clinical-significance discussion | New paragraph: what a QWK 0.87->0.37-0.50 collapse means operationally | Discussion |
| Avoid overclaiming | Scope + Limitations restrict to one checkpoint / one task; conclusion says "did not outperform" not "fails" | main_standalone.tex |
| Retrospective / non-clinical | Stated in Scope, Methods, Limitations, Ethics | main_standalone.tex |
| Sample-size / power | No a-priori power calc (external n fixed by releases); rare-grade precision limits stated with actual counts; primary-quantity CI half-widths given | Limitations |
| Sensitivity analyses | Documented: 2nd CNN backbone, 2 RETFound regimes, single-pass vs MC entropy, dedup-internal, independent recompute. Not-yet-run (CLAHE, seeds, class-weighting) listed | Limitations |
| Stronger baseline symmetry | 4-model framework already symmetric; per-class + calibration + ROC/PR now for all 4 | tables/figs |
| Figures | Now 10: pipeline (TikZ), grade dist, forest, confusion, per-class heatmap, per-class CI forest, reliability, ROC/PR, saliency, uncertainty | `paper/figures/` |
| Supplementary material | Itemised paragraph before Declarations | main_standalone.tex |
| Reproducibility | Result->script table extended with phase11/phase12; verification statement added | Reproducibility section |

## CANNOT be done in this environment

- **Complete the RETFound full fine-tune.** Attempted 4x this session; the machine
  throttles (~90s -> 400-900s/epoch) then exhausts host memory. Two attempts DID
  reach converged validation QWK (~0.94, epoch 16-19) before dying, so the
  epoch-6 checkpoint is representative -- now stated as fact in Limitation (i);
  a clean run on other hardware is follow-up #2. `\pv{}` markers stay until that
  run exists (honest, not placeholders). A 5th attempt runs in the background
  (`--num-workers 2 --save-every 2`); if it finishes, run the refresh block in
  `docs/project_notes/SESSION_STATUS.md` and delete `\pv{}`.
- **Blinded ophthalmologist assessment + inter-rater kappa.** Needs a human
  expert. Follow-up #3.
- **IDRiD lesion-mask overlap + full 516 set.** Lesion masks are in the IDRiD
  *Segmentation* subset (not downloaded; needs IEEE DataPort access); the 61
  missing Disease-Grading images need a re-download. Current proxy is a location
  check, now labelled as such everywhere.
- **Controlled colour/resolution/training-size/multi-dataset ablation.** Hours of
  GPU work on hardware that cannot sustain it. Specified precisely as follow-up #1.

## NEEDS YOU (cannot be fabricated)

- Author name, affiliation, city/country, ORCID (`% >>> BEFORE SUBMISSION`).
- Repository URL + Zenodo DOI (push repo; `.gitignore` excludes images/weights).
- Ethics wording for the target journal's secondary-use clause.
- Funding + competing-interests statements (drafts present as `[bracketed]` text).
- Dataset download dates; Messidor-2 adjudicated-grades Kaggle slug/licence; APTOS
  competition-rules research-use clause (`DATA_PROVENANCE.md`).
- No LaTeX toolchain here: compile `paper/main_standalone.tex` + `references.bib`
  + `paper/figures/` on Overleaf.
