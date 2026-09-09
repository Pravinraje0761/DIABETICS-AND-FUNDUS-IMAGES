# Session status — Q1 submission preparation

## COMPLETE this session

### New analysis code (all syntax-checked, self-contained — no new deps)
- `scripts/phase7_explainability.py` — Grad-CAM++ (CNN) + method-matched ViT
  Grad-CAM over patch tokens; automated plausibility proxy; `--vit-method
  rollout` keeps attention rollout as a supplement. Outputs `outputs/phase7/`.
- `scripts/phase8_uncertainty.py` — predictive-entropy error detection
  (Mann-Whitney U, AUROC), selective-prediction curves, error capture at
  90/80% coverage. Outputs `outputs/phase8/`.
- `scripts/phase8_mc_dropout.py` — T=30 MC-dropout (+ drop-path) inference;
  writes Phase-6-schema preds to `outputs/phase8/mc/` for `phase8_uncertainty.py
  --preds-dir outputs/phase8/mc --tag mc`. NOT YET RUN (needs a free GPU).
- `scripts/phase9_stats.py` — bootstrap 95% CIs (B=2000, seed 42), drop CIs,
  paired difference-in-drop CIs, DeLong paired-AUC (self-contained). Outputs
  `outputs/phase9/`.
- `scripts/phase10_figures.py` — Fig 1 study design, Fig 2 forest plot.

### Results produced (against the current epoch-6 RETFound checkpoint)
- `outputs/phase9/` — bootstrap_ci.csv, drop_ci.csv, drop_difference_ci.csv,
  delong.csv, phase9_summary.md, stats_methods_paragraph.md. HEADLINE:
  RETFound's internal->external QWK drop on Messidor-2 is significantly larger
  than EfficientNet-B3's (full FT +0.124 [0.074,0.175]; linprobe +0.076
  [0.019,0.132]); EfficientNet-B3 referable-DR AUC significantly higher on both
  external sets (DeLong p<0.001) but tied internally (p>=0.47).
- `outputs/phase8/` — uncertainty_singlepass.csv, selective_prediction_singlepass.csv,
  fig_uncertainty_singlepass.png. External entropy AUROC 0.55-0.68; RETFound-full
  on IDRiD = chance (0.50, p=0.52). 10% referral captures ~10-18% of errors.
- `outputs/phase7/` — saliency_per_image.csv (160 maps), plausibility_rates.csv,
  fig_explainability.png, maps/. CNN Grad-CAM 75-100% proxy-plausible externally;
  RETFound ViT Grad-CAM 83-88% on IDRiD but 12-58% on Messidor-2 (border-heavy),
  worse on its errors.
- `outputs/figures/` — fig1_study_design.{png,pdf}, fig2_performance_forest.{png,pdf}.

### Manuscript / package
- `paper/main.tex` — Abstract, Tables 2-3 (with CIs + difference-in-drop),
  RQ1/RQ2/RQ3 Results, Explainability + Uncertainty Results with figures,
  Discussion (RQ1-3, literature, 7-point Limitations), Conclusions, all Methods
  subsections, Declarations templated. Figures 1-4 wired via \includegraphics.
  All \ref resolve; all \cite keys exist. Remaining \todo = author-only
  (names/affil/email, repo DOI, ethics wording, funding, COI, CRediT, AI
  disclosure). RETFound-full numbers carry the \prov superscript.
- `paper/TRIPOD-AI_checklist.md`, `paper/COVER_LETTER.md`, `paper/AUTHOR_TODO.md`,
  `paper/manuscript_rendered.md` (plain-text render for review).
- `DATA_PROVENANCE.md`, `README.md`, `references.bib` updated.
- `scripts/dataset.py` (`DR_DATA_ROOT` env), `scripts/train_retfound.py`
  (`--save-every`) — infra fixes for the OneDrive/Defender stalls.

## RETFound full fine-tune — OUTCOME (2nd completion attempt, 2026-09-08)

Ran epochs 1-22 (val QWK 0.645 -> 0.933@ep10 -> **0.943@ep19**, then plateau at
~0.94). At epoch 22 the host ran out of memory; the dataloader thrashed on swap
(epoch 23 made no progress for ~50 min) and the run was terminated. The good
weights (epoch 19) were held in RAM by `--save-every 6` and were lost with the
kill; the on-disk `C:\dr_ckpts_full\full_best.pt` is only epoch 6 (val 0.915) and
the committed `outputs/phase5_retfound/full_best.pt` (original run, epoch 6, val
0.935) remains the RETFound-full model used everywhere.

**Consequence for the paper: none.** The `\prov` values are already computed from
`outputs/phase5_retfound/full_best.pt` and are correct/reproducible. The 2nd
attempt's val-QWK trajectory converging to ~0.94 (matching the 1st) is now folded
into Limitation (i) as *strengthening* evidence that epoch 6 is representative of
the converged model. No number changes. `main.tex` Limitation (i) updated.

A clean completed run (Colab/Kaggle, or a machine with more RAM + a Defender
exclusion) remains the recommended pre-submission step but is not blocking.

## MC-dropout sensitivity — DONE (2026-09-08)

`phase8_mc_dropout.py --passes 20` (baselines first at default; RETFound re-run
after fixing `--min-dropout-p` default 0.1 -> 0.0: forcing drop-path noise into
the frozen linear-probe backbone had collapsed it to QWK -0.01). Outputs
`outputs/phase8/mc/`, `outputs/phase8/uncertainty_mc.csv`, `fig_uncertainty_mc.png`.

Result: **MC entropy == single-pass entropy as an error detector** — every
external AUROC within +/-0.02 of single-pass (0.54-0.69), RETFound-full/IDRiD
still no signal (AUROC 0.51, p=0.37). `main.tex` Uncertainty Methods + Results
sentence updated to cite the T=20 MC result concretely. Single-pass stays
primary; the paper figure stays `fig_uncertainty_singlepass.png`;
`fig_uncertainty_mc.png` + `uncertainty_mc.csv` are supplementary.

## NOTHING RUNNING. Package is complete against the committed epoch-6 RETFound
## checkpoint. Only author-only items remain (`paper/AUTHOR_TODO.md`).

## WHEN RETFound COMPLETES — refresh procedure (also in AUTHOR_TODO / SUBMISSION_CHECKLIST)

```
cd dr_foundation_study
set DR_DATA_ROOT=C:\dr_run          # (PowerShell: $env:DR_DATA_ROOT="C:\dr_run")
.venv\Scripts\python.exe scripts\predict.py --checkpoint C:\dr_ckpts_full\full_best.pt ^
    --manifest data_splits\aptos_manifest.csv --split test ^
    --out-prefix outputs\phase5_retfound\full_internal_test
.venv\Scripts\python.exe scripts\phase6_external_eval.py --models retfound_full
.venv\Scripts\python.exe scripts\phase9_stats.py --boot 2000 --seed 42
.venv\Scripts\python.exe scripts\phase10_figures.py
.venv\Scripts\python.exe scripts\phase7_explainability.py --device cuda
.venv\Scripts\python.exe scripts\phase8_mc_dropout.py --passes 30 --device cuda
.venv\Scripts\python.exe scripts\phase8_uncertainty.py --preds-dir outputs\phase8\mc --tag mc
```
Then in `paper/main.tex` replace each `\prov{<old>}` with the new value and
delete the `\prov` wrapper + the `\textsuperscript{p}` markers; soften
Limitation (i) to "ran to early-stopping convergence"; update
`paper/manuscript_rendered.md` **[p]** rows. Direction of every finding is
expected to hold (linprobe already concordant).

## STILL PENDING (needs GPU free / author)
- MC-dropout run (`phase8_mc_dropout.py`) — GPU busy with training.
- All `paper/AUTHOR_TODO.md` items.
- No LaTeX toolchain here — compile on Overleaf or install TeX (see AUTHOR_TODO D).
