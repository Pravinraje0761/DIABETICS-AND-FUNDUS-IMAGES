# Author action list — everything the pipeline cannot supply

_Last refreshed 2026-09-10 (after Phase 7/8/9 extensions, commit `ced8492`)._

**Current state.** All analysis is done and committed: Phases 2–16 plus the
Phase 7/8/9 extensions (per-class uncertainty, MC-dropout re-run of the full
statistical pipeline, blinded-rating packet + scorer) and the Phase 17
class-weighting / loss-function robustness check. The manuscript is
`paper/main_standalone.tex` (IEEEtran, primary); `paper/main.tex` is the
superseded `article`-class draft, kept only for a possible journal-class port.
Every table/abstract number has been cross-checked against `outputs/phase6..17/`.

The items below need a human decision, a credential, a licence-gated download, or
a fact only you hold. Ordered by how much they block submission.

---

## A. Blocks submission

1. **Author identity block** — `main_standalone.tex`, the `% >>> BEFORE
   SUBMISSION` comment block (~line 49) and `\thanks{}`: full legal name,
   affiliation, city/country, corresponding-author email, ORCID iD.

2. **Declarations** (`\section*{Declarations}`, ~line 1405) — three bracketed
   fills:
   - **Code availability**: `[repository URL / archival DOI …]` → the public
     GitHub URL and the Zenodo DOI (see item 4).
   - **Ethics approval**: replace `[Replace with the target journal's exact
     required wording for secondary use of public, de-identified data.]` with the
     chosen journal's exact phrasing.
   - **Use of AI tools**: the disclosure sentence is drafted; adjust it to the
     target journal's AI-use policy.
   Funding, Competing interests, and Author contributions are drafted for a
   single author ("P.") — confirm or replace.

3. **Dataset provenance facts** (`DATA_PROVENANCE.md`, "Still requires the
   author's own records"): dataset download dates; the Kaggle slug / URL /
   licence for the Messidor-2 adjudicated grades (Krause et al.); the current
   adcis.net Messidor-2 licence clause; the APTOS competition-rules research-use
   clause. These feed the Data-availability and Ethics statements and TRIPOD+AI
   item 4b.

4. **Repository + DOI**: push the repo public (GitHub), tag a release, mint a
   Zenodo DOI, and paste the DOI into the Code-availability statement.
   `.gitignore` already excludes raw images, `*.pt`, `weights/`,
   `data_processed*/`, and the regenerable Phase-7 rating-packet images.

5. **Revoke the Hugging Face token** pasted into the terminal during Phase 5
   setup (regenerate one if you still need gated RETFound access).

6. **Reconcile two stale working-tree files** (mechanical, do once):
   `outputs/phase5_retfound/full_history.csv` and `full_hyperparams.json` in the
   working tree describe the abandoned "v3" retrain, not the committed epoch-6
   run that all `\pv{}` numbers trace to. The v3 trace is preserved in
   `logs/phase5_retfound_full_v3.log`. Run, by hand (a repo guard blocks it in
   automation):
   `git checkout HEAD -- outputs/phase5_retfound/full_history.csv outputs/phase5_retfound/full_hyperparams.json`

---

## B. Strongly recommended before submission — a compute job, not an author fact

7. **Completed RETFound full fine-tune.** The results use an epoch-6 checkpoint
   (`outputs/phase5_retfound/full_best.pt`, val QWK 0.935) that had already
   reached its converged validation QWK, but the run was stopped early because a
   ViT-L full fine-tune thermally/memory-throttles on this laptop. Seventeen
   numbers in `main_standalone.tex` therefore carry the `\pv{}` superscript-p,
   and Table captions + Limitations carry an "epoch-6" caveat.

   Run to completion in a resource-unconstrained environment (free Colab / Kaggle
   T4 or L4):
   ```
   # upload data_processed/, data_splits/, weights/RETFound_mae_natureCFP.pth, scripts/, requirements.txt
   pip install -r requirements.txt
   python scripts/train_retfound.py --mode full \
       --weights weights/RETFound_mae_natureCFP.pth --batch-size 16 --ckpt-dir ./ckpts
   ```
   Then the refresh:
   ```
   copy <new full_best.pt> outputs/phase5_retfound/full_best.pt
   python scripts/predict.py --checkpoint outputs/phase5_retfound/full_best.pt \
       --manifest data_splits/aptos_manifest.csv --split test \
       --out-prefix outputs/phase5_retfound/full_internal_test
   python scripts/phase6_external_eval.py --models retfound_full
   python scripts/phase9_stats.py --boot 2000 --seed 42
   python scripts/phase9_stats.py --preds-dir outputs/phase8/mc --tag mc
   python scripts/phase11_review_response.py --boot 2000 --seed 42
   python scripts/phase10_figures.py && python scripts/phase10b_figures.py && python scripts/phase11_figures.py
   python scripts/phase7_explainability.py --device cuda --n-correct 40 --n-incorrect 25
   python scripts/phase8_mc_dropout.py --passes 20 --device cuda --models retfound_full
   python scripts/phase8_uncertainty.py            # single-pass
   python scripts/phase8_uncertainty.py --preds-dir outputs/phase8/mc --tag mc
   python scripts/phase13_temperature_scaling.py --calib-preds outputs/phase13/calib
   copy outputs/figures/*.pdf outputs/phase7/fig_explainability.png \
        outputs/phase8/fig_uncertainty_singlepass.png paper/figures/
   ```
   Then in `main_standalone.tex`: replace every `\pv{<num>}` with the new value
   and delete the `\pv{}` wrapper (17 uses) + the `\textsuperscript{p}` on the
   RETFound (full fine-tune) rows of Tables `tab:master`, `tab:drop`, and the
   calibration table; delete the `\newcommand{\pv}` line + its comment; and in
   Limitations change the epoch-6 wording to "ran to its pre-specified
   early-stopping convergence". Also update `paper/manuscript_rendered.md`.

   The direction of every comparison is not expected to move: the RETFound
   linear-probe row (which always ran to completion) already shows the same
   pattern, and the seed sweep, the CLAHE / colour-transfer / resolution
   ablations, and the loss ablation (Phase 17) all corroborate it.

8. **RETFound full fine-tune × 3 extra seeds** (43/44/45) in the same clean
   environment, to lift the seed sweep from the linear-probe regime to the full
   fine-tune. Currently listed as "Not yet run" in Limitations.

---

## C. Journal-specific — do once a target is chosen (see `SUBMISSION_CHECKLIST.md`)

9.  Download the current author kit for the target journal. **Recommended
    target: _Translational Vision Science & Technology_ (TVST, ARVO)** — see
    `SUBMISSION_CHECKLIST.md` §1 for the rationale and alternatives.
10. Port `main_standalone.tex` to the journal's class if required (TVST accepts a
    Word or a reasonably standard LaTeX/PDF submission; MDPI needs `mdpi.cls` +
    `Definitions/mdpi.bst` and the author/affiliation/keyword blocks moved to its
    macros). Confirm abstract format, word / figure limits, figure DPI + colour
    space, and any graphical-abstract requirement.
11. Confirm the APC on the journal's own page and arrange funding or a waiver
    **before** submitting.
12. Check for an open Special Issue on AI/ML in ophthalmic imaging and mention it
    in the cover letter (`paper/COVER_LETTER.md`, fill journal name + reviewer
    list).
13. Suggested reviewers: 3–5, no COI — from the RETFound authors, DR-AI
    external-validation groups, and TRIPOD+AI authors.
14. Fill the **TRIPOD+AI checklist** (`paper/TRIPOD-AI_checklist.md`) with the
    section + line references from the line-numbered compiled PDF; upload it as a
    supplementary file.

---

## D. Optional — reviewers may ask; each needs data or hardware not available here

15. **Blinded expert rating** of the Phase 7 saliency maps. The packet, rubric,
    and scorer are built (`scripts/phase7_rating_packet.py`,
    `outputs/phase7/rating_packet/`): 160 clean two-panel overlays, a blank
    rating sheet, and a withheld key. Give the packet to ≥2 practising
    ophthalmologists, collect `rater_<initials>.csv` files, then
    `python scripts/phase7_rating_packet.py --score outputs/phase7/rating_packet/rater_*.csv`
    → `outputs/phase7/rating_agreement.csv` + `phase7_rating_summary.md`
    (expert-plausible rate, expert-vs-proxy Cohen's κ, inter-rater κ per cell).
16. **IDRiD Segmentation subset** (pixel-level MA/HE/EX masks; IEEE DataPort
    doi:10.21227/H25W98, separate from the Disease-Grading subset used here) for
    a true saliency–lesion IoU/Dice, replacing the automated location proxy.
17. **Full 516-image IDRiD Disease-Grading set** — re-download, re-run
    `phase6_external_eval.py` + `phase9_stats.py` + `phase11_review_response.py`,
    confirm the drop is unchanged, and characterise the 61 images missing from
    the mirror used here.
18. **Retraining-based mechanistic ablations** — colour- and
    resolution-normalised *fine-tuning* (the Phase 15 ablations apply the
    transforms at inference only), a training-set-size sweep, and a second
    downstream task or a second retinal foundation model. All need sustained
    ViT-L training, i.e. the same clean environment as item 7.

---

## E. Build

- **No LaTeX toolchain on this machine.** Compile on Overleaf: upload
  `paper/main_standalone.tex`, `paper/references.bib`, and the whole
  `paper/figures/` folder (10 files). `IEEEtran.cls` / `IEEEtran.bst` ship with
  Overleaf. Locally:
  `pdflatex main_standalone && bibtex main_standalone && pdflatex main_standalone && pdflatex main_standalone`.
- Fix any overfull-hbox warnings on the wide `table*` floats (cosmetic; shrink
  `\tabcolsep` or drop a font size per table).
- Author-fill points are the `% >>> BEFORE SUBMISSION` comment and the
  `[bracketed]` notes in the Declarations.
- `paper/manuscript_rendered.md` is a plain-text rendering and currently **lags
  the `.tex`** (it carries a staleness banner); regenerate or drop it before
  sharing.
- Decide whether to keep both `paper/main.tex` and `paper/main_standalone.tex`
  or delete the superseded `article`-class draft.
