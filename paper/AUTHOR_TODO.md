# Author action list — everything the pipeline cannot supply

The analysis, figures, tables, checklist and manuscript prose are complete and
internally consistent (see `paper/main.tex`, `outputs/phase7..10/`,
`paper/TRIPOD-AI_checklist.md`). The items below need a human decision, a
credential, or a fact only you hold. Ordered by how much they block submission.

## A. Blocks submission

1. **Author identity block** (`main.tex`, `\author`/`\affil`/`\thanks`):
   full names, affiliations, city/country, corresponding-author email, ORCIDs.
2. **Declarations** (`main.tex` `\section*{Declarations}`):
   - Funding — source(s) or "This research received no external funding."
   - Conflicts of Interest — declare or "The authors declare no conflict of interest."
   - Author Contributions — CRediT roles per author.
   - Use of AI tools — a disclosure sentence is drafted in `main.tex`; confirm it
     matches the target journal's policy and the actual division of labour.
   - Ethics — a wording is drafted; **replace with the target journal's exact
     required phrasing for secondary use of fully public, de-identified data.**
3. **Dataset provenance facts** (`DATA_PROVENANCE.md` "Still requires the author's
   own records"): download dates; the Kaggle slug/URL/licence for the Messidor-2
   adjudicated grades; current adcis.net Messidor-2 licence clause; the APTOS
   competition-rules research-use clause. These feed the Data Availability +
   Ethics statements and TRIPOD+AI item 4b.
4. **Repository + DOI**: push the repo (public GitHub), tag a release, mint a
   Zenodo DOI, and paste the DOI into `main.tex` Data/Code Availability
   (`\todo{repository URL / DOI}`). `.gitignore` already excludes raw images,
   `*.pt`, `weights/`, `data_processed/`.
5. **Revoke the Hugging Face token** that was pasted into the terminal during
   Phase 5 setup (regenerate a new one if you still need RETFound access).

## B. Journal-specific, do once a target is chosen (Phase 13)

6. Download the current author kit for the target journal (primary suggestion:
   *Diagnostics*, MDPI; see `SUBMISSION_CHECKLIST.md` for alternatives incl.
   *Translational Vision Science & Technology* and *Ophthalmology Science*, which
   suit a clean negative external-validation result).
7. Port `main.tex` from the generic `article` class to the journal class
   (`mdpi.cls` + `Definitions/mdpi.bst` for MDPI): move the author/affil/keywords
   blocks to the class macros; confirm structured-abstract format; check word and
   figure limits; RGB vs CMYK; graphical-abstract requirement.
8. Confirm the APC on the journal's own page and arrange funding/waiver **before**
   submitting.
9. Check for an open Special Issue on AI/ML in ophthalmic imaging; mention it in
   the cover letter (`paper/COVER_LETTER.md`).
10. Suggested reviewers: 3-5, no COI - pick from the RETFound authors, DR-AI
    external-validation groups, and TRIPOD+AI authors.

## C. Optional, strengthens the paper (reviewers may ask)

11. **Blinded expert rating** of the Phase 7 saliency maps by an ophthalmologist
    (Cohen's kappa vs. a second rater) to replace the automated plausibility
    proxy, or keep the proxy and the stated limitation.
12. **Full IDRiD 516**: re-download from IEEE DataPort (doi:10.21227/H25W98),
    re-run `phase6_external_eval.py` then `phase9_stats.py`; confirm the drop is
    unchanged (expected).
13. **Calibration**: add reliability diagrams + expected calibration error per
    model per dataset, or cite as future work (TRIPOD+AI item 14, currently
    "Partial").
14. **Second foundation model** or a second downstream task, to broaden the claim
    beyond RETFound-MAE / DR grading. Currently scoped explicitly in Limitations.

## D. Build

- **Two manuscript files.** `paper/main_standalone.tex` is the IEEEtran
  two-column journal build (matches the GI-endoscopy paper's format: TikZ
  pipeline figure, 7 figures, structured Limitations + Reproducibility +
  Declarations, numbered `\cite`, `\bibliographystyle{IEEEtran}`). Suits IEEE
  J-BHI / IEEE Access / TMI. `paper/main.tex` is the earlier journal-agnostic
  `article`-class draft (same content) -- keep whichever the target venue wants.
- No LaTeX toolchain is installed on this machine. Compile on Overleaf: upload
  `paper/main_standalone.tex`, `paper/references.bib`, and the whole
  `paper/figures/` folder (7 files). `IEEEtran.cls`/`IEEEtran.bst` are on
  Overleaf by default. Or install MiKTeX/TeX Live and run
  `pdflatex main_standalone && bibtex main_standalone && pdflatex main_standalone && pdflatex main_standalone`.
- Author-fill points in `main_standalone.tex` are `% >>> BEFORE SUBMISSION`
  comments and `[bracketed notes]`; RETFound-full numbers carry the `\pv{}`
  superscript-p (defined at the top).
- A plain-text rendering is in `paper/manuscript_rendered.md`.
