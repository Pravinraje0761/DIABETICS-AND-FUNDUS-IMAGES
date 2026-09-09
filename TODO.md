# TODO -- diabetic-fundus external-validation paper

Status snapshot (2026-09-08, evening): manuscript = `paper/main_standalone.tex`
(IEEEtran, 10 figures, 8 tables, all refs/cites resolve). Analyses Phases 2-12
done. What was done vs. blocked: `REVIEW_RESPONSE_STATUS.md`.

RETFound full fine-tune v3 was **paused at epoch 7** to free the machine. Best
weights saved: `C:\dr_ckpts_full\full_best.pt` = epoch 6, val QWK **0.930**
(trajectory 0.61 -> 0.82 -> 0.88 -> 0.90 -> 0.91 -> 0.93; heading to ~0.94
plateau by epoch ~16-19, same as prior attempts).

---

## 1. Finish the RETFound full fine-tune (the one blocker for removing `\prov{}`)

Best on hardware that does not thermal/memory-throttle a ViT-L (free Colab/Kaggle
T4/L4, or a desktop with a Defender exclusion + >=32 GB RAM):

```
# upload data_processed/, data_splits/, weights/RETFound_mae_natureCFP.pth, scripts/
pip install -r requirements.txt
python scripts/train_retfound.py --mode full \
    --weights weights/RETFound_mae_natureCFP.pth --batch-size 16 --ckpt-dir ./ckpts
```

If staying on this laptop: the v3 settings (`--num-workers 2 --save-every 2`,
`DR_DATA_ROOT=C:\dr_run`) were the most stable yet -- it reached epoch 7 without
dying. Relaunch and let it run overnight; `--save-every 2` keeps the best weights
on disk if it stalls.

### After it completes -- the refresh (also in `SESSION_STATUS.md`)

```
set DR_DATA_ROOT=C:\dr_run     # PowerShell: $env:DR_DATA_ROOT="C:\dr_run"
copy <new full_best.pt> outputs\phase5_retfound\full_best.pt
.venv\Scripts\python.exe scripts\predict.py --checkpoint outputs\phase5_retfound\full_best.pt ^
    --manifest data_splits\aptos_manifest.csv --split test ^
    --out-prefix outputs\phase5_retfound\full_internal_test
.venv\Scripts\python.exe scripts\phase6_external_eval.py --models retfound_full
.venv\Scripts\python.exe scripts\phase9_stats.py --boot 2000 --seed 42
.venv\Scripts\python.exe scripts\phase11_review_response.py --boot 2000 --seed 42
.venv\Scripts\python.exe scripts\phase10_figures.py
.venv\Scripts\python.exe scripts\phase10b_figures.py
.venv\Scripts\python.exe scripts\phase11_figures.py
.venv\Scripts\python.exe scripts\phase7_explainability.py --device cuda --n-correct 40 --n-incorrect 25
.venv\Scripts\python.exe scripts\phase8_mc_dropout.py --passes 30 --device cuda --models retfound_full
.venv\Scripts\python.exe scripts\phase8_uncertainty.py --preds-dir outputs\phase8\mc --tag mc
copy outputs\figures\*.pdf outputs\phase7\fig_explainability.png outputs\phase8\fig_uncertainty_singlepass.png paper\figures\
```

Then in `paper/main_standalone.tex`:
- replace each `\pv{<num>}` with the new value and delete the `\pv{}` wrapper
  (28 uses) + the `\textsuperscript{p}` after "RETFound (full fine-tune)" in
  Tables 2, 3 and the calibration table;
- delete the `\newcommand{\pv}` line and its comment;
- in Limitation "Internal validity", change the epoch-6 wording to "ran to
  early-stopping convergence per the pre-specified protocol";
- update `paper/manuscript_rendered.md` **[p]** rows.

---

## 2. Author-only items (cannot be fabricated) -- see `paper/AUTHOR_TODO.md`

- [ ] Full legal name, affiliation, city/country, ORCID -> `% >>> BEFORE SUBMISSION`
      block in `main_standalone.tex`.
- [ ] Push repo public (GitHub), tag a release, mint a Zenodo DOI, paste it into
      Code-availability + Reproducibility. `.gitignore` excludes `data_processed/`,
      `*.pt`, `weights/`, raw images.
- [ ] Revoke the Hugging Face token pasted during Phase 5 setup.
- [ ] Ethics: replace the `[bracketed]` sentence with the target journal's exact
      wording for secondary use of public, de-identified data.
- [ ] Funding, Competing interests, Author contributions, Use-of-AI disclosure:
      confirm/replace the `[bracketed]` drafts.
- [ ] `DATA_PROVENANCE.md` "Still requires the author's own records": dataset
      download dates; Kaggle slug/URL/licence for the Messidor-2 adjudicated
      grades; the APTOS competition-rules research-use clause.

---

## 3. Analyses worth adding before submission (paper's "highest-value next experiments")

1. [ ] **Controlled degradation ablation** -- the "why RETFound fails" experiment.
   Re-run external inference for RETFound-full with (a) CLAHE on/off
   (`phase3_preprocess.py --clahe`), (b) input resolution 224 vs 384,
   (c) colour-transfer normalisation matching external channel stats to APTOS.
   Report which recovers external QWK. New script `scripts/phase13_ablation.py`
   (not written).
2. [ ] **APTOS-only vs APTOS + one-other-set training** for RETFound-full; tests
   whether single-source training is the cause.
3. [ ] **Blinded ophthalmologist rating** of >=60 saliency maps vs lesion masks,
   2 raters, Cohen's kappa. Use the **IDRiD Segmentation subset** (ships
   pixel-level MA/HE/EX masks; download from IEEE DataPort) for true
   saliency-lesion IoU/Dice -- replaces the automated location proxy for RQ2.
4. [ ] **Full IDRiD 516** Disease-Grading set: re-download, re-run
   `phase6_external_eval.py` + `phase9_stats.py` + `phase11_review_response.py`,
   confirm the drop is unchanged; identify + characterise the 61 missing images.
5. [x] **Multiple training seeds.** DONE for EfficientNet-B3 AND RETFound linear
   probing: seeds 43/44/45 trained for each
   (`outputs/phase4_baseline/efficientnet_b3_seed4*`,
   `C:\dr_ckpts_seeds\linprobe_seed4*_best.pt` +
   `outputs/phase5_retfound/linprobe_seed4*`), all 4 seeds/model scored on
   APTOS-internal + Messidor-2 + IDRiD by `scripts/phase16_seed_sweep.py`
   (`outputs/phase16/`). Internal QWK SD 0.003 (EffNet) / 0.008 (RETFound-LP);
   Messidor-2 QWK 0.500 +/- 0.024 vs 0.395 +/- 0.015, non-overlapping ranges;
   Messidor-2 drop 0.379 +/- 0.025 vs 0.461 +/- 0.017 -> gap is not seed noise;
   RETFound-full's +0.124 excess drop is ~5x the baseline seed SD. Folded into
   Limitations "Sensitivity analyses" (vi) + repro table + supplement list +
   `manuscript_rendered.md`. RETFound-full x3 seeds still blocked by laptop
   throttle (needs Colab/Kaggle) -- noted as remaining in "Not yet run".
6. [ ] **Class-weighting sensitivity** (focal loss / no weights) as a robustness
   check (currently inverse-frequency CE only).
7. [ ] **Post-hoc temperature scaling** on a dedicated calibration split
   (`predict.py --split val` for APTOS val logits); re-report ECE post-scaling in
   the calibration table.
8. [ ] **Embedding-based near-duplicate check** across datasets (frozen
   EfficientNet-B3 features + cosine sim); the perceptual dhash in
   `phase12_leakage.py` is noisy on cropped fundus frames, MD5 catches only exact
   copies.

---

## 4. Journal targeting -- see `paper/SUBMISSION_CHECKLIST.md`

- [ ] Pick a target. IEEEtran build suits IEEE J-BHI / Access / TMI. For a fast
      MDPI turnaround (Diagnostics, Journal of Imaging) port
      `main_standalone.tex` to `mdpi.cls` (move title/author/abstract/keywords to
      its macros; body unchanged). TVST / Ophthalmology Science also fit a
      negative external-validation result.
- [ ] Download the author kit; check word/figure limits, abstract format, figure
      DPI/format (figures are 300-dpi PDF + PNG in `paper/figures/`).
- [ ] Confirm the APC; arrange funding/waiver before submitting.
- [ ] Cover letter `paper/COVER_LETTER.md` (fill journal name + reviewer list).
- [ ] Suggested reviewers: 3-5, no COI.

---

## 5. Build / mechanical

- [ ] Compile `paper/main_standalone.tex` on Overleaf: upload it +
      `paper/references.bib` + the whole `paper/figures/` folder (10 files).
      `IEEEtran.cls`/`.bst` ship with Overleaf. No LaTeX toolchain on this machine.
- [ ] Fix overfull-hbox warnings on wide `table*` floats if any (cosmetic; shrink
      `\tabcolsep` or drop a font size per table).
- [ ] Decide whether to keep both `paper/main.tex` (article class) and
      `paper/main_standalone.tex` (IEEEtran) or delete the unused one.
- [ ] Kill the stray session background task still watching
      `logs/phase5_retfound_full_v3.log` for a completion line that will never
      appear (harmless; clears on shell/session close).
