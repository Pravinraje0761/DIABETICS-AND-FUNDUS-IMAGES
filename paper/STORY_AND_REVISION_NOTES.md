# Story, evidence map, and open items — revised manuscript (2026-10-21)

Manuscript: `paper/main_standalone.tex` (IEEEtran, 22 pages, compiles cleanly).
Overleaf package: `paper/overleaf_package.zip`.

## 1. The study in five lines

1. **Central question.** When a retinal foundation model (RETFound) and an
   ImageNet CNN are trained and evaluated identically, does RETFound lose less
   DR-grading agreement on external data?
2. **Main contribution.** A protocol-matched, multiplicity-controlled test of
   that question. It shows that fully fine-tuned RETFound generalises neither
   better nor measurably worse than the CNN, and that frozen RETFound features
   generalise worse.
3. **Strongest evidence.** The paired difference in internal-to-external QWK
   drop (Table V): full fine-tune +0.018 [−0.024, 0.058] on Messidor-2 and
   +0.022 [−0.033, 0.083] on IDRiD, Holm p = 0.79. Linear probe +0.076
   (Holm p = 0.030) and +0.105 (Holm p = 0.016). The full-fine-tune result is
   stable under every sensitivity analysis.
4. **Logic chain.**
   - *Gap:* external DR AI degrades, and foundation models are expected to
     help, but they have not been compared with the standard CNN under one
     protocol with the drop as the estimand.
   - *Design:* develop on one dataset; score once, unchanged, on two external
     datasets that differ in labels and in camera; compare paired drops.
   - *Findings:* parity for the full fine-tune, deficit for the linear probe;
     entropy uninformative under shift.
5. **What it cannot establish.**
   - Anything clinical or prospective.
   - Any claim about foundation models in general, since one checkpoint was
     tested on one task.
   - Lesion-level explanation, since only a location proxy was used.
   - Run-to-run variability of the full fine-tune, since it was a single seed.
   - Results on the official IDRiD labels.

## 2. Argument outline (what each part does)

| Section | Job in the argument | What a skeptical reviewer asks — and where it is answered |
|---|---|---|
| Introduction | Problem (screening scale, external degradation) → known (RETFound promise; Hou et al. compared two foundation models on the same datasets) → three unresolved points → why they matter → RQ1–3 → findings in brief, including the interrupted-run episode | "Hasn't this been done?" → Background: Hou et al. compared foundation models with each other, by AUC, without a supervised-CNN comparator or the drop as estimand |
| Background and positioning | Places the study against the closest work; no novelty overclaim | "Why these baselines?" → ImageNet CNNs are the usual reference point |
| Methods: datasets, audit, reference standards | Makes data provenance auditable; shows labels differ by protocol | "Is IDRiD trustworthy?" → missing IDs, label discrepancy, reweighting |
| Methods: split, leakage, lock | Shows the external sets could not have shaped the models | "Leakage? Tuning on test?" → git-dated evidence; `CHECKPOINTS.md` |
| Methods: statistics | Fixes one primary outcome and a four-contrast confirmatory family with Holm correction | "Multiplicity? Correlated eyes?" → Holm/Bonferroni; m = n/2 bootstrap |
| Results 1–3 (RQ1) | Internal parity → primary contrasts → robustness → referable-DR | "Is the null just low power?" → CIs bound the plausible effect sizes |
| Results 4–6 (RQ2) | Can errors be flagged? Entropy, calibration, OOD | "Would MC dropout help?" → stochastic-pass analysis, corrected |
| Results 7–9 (RQ3 + probes) | Where models look; sensitivity to input distribution; head-artefact check | "Is the linear-probe gap a weak head?" → 210-head sweep (Appendix A) |
| Discussion | Answers the RQs; the interrupted-run lesson; labels vs images; literature; speculative mechanism; restrained implications | "What does this mean clinically?" → nothing clinical; external validation is the evidence standard |
| Limitations / Conclusion | Bound every claim to the design | — |

## 3. Claim-to-evidence table (central conclusions)

| Claim in the paper | Evidence | Strength / caveat |
|---|---|---|
| Fully fine-tuned RETFound and EfficientNet-B3 are indistinguishable internally | Table IV; paired QWK diff +0.006 [−0.014, 0.029] (`phase23/qwk_pairwise_diff.csv`) | Internal split is not patient-disjoint |
| Full fine-tune generalises neither better nor measurably worse | Table V: ΔDrop +0.018 / +0.022, Holm p = 0.79; stable under m = n/2, IDRiD reweighting, stochastic passes | Single seed; CIs exclude an advantage > ~0.02–0.03 and a deficit > ~0.06–0.08 |
| Linear probe generalises worse | Table V: +0.076 (Holm 0.030), +0.105 (Holm 0.016); seed distributions do not overlap (Phase 16); 210-head sweep | Under the worst-case m = n/2 bootstrap, the Messidor-2 CI reaches 0 |
| Every model degrades substantially, most on Messidor-2 | Table IV; QWK 0.85–0.88 → 0.39–0.50 | Part of the drop reflects a change of reference standard |
| Referable-DR: CNN better than full fine-tune on IDRiD only | Table VI: 0.962 vs 0.885, Holm p < 0.001; Messidor-2 Holm p = 0.11 | Secondary; IDRiD copy has label issues |
| Entropy is a weak external error signal; none for full FT on IDRiD | Section "Can the models' uncertainty flag their errors?": AUROC 0.49–0.68; 10 % referral captures 11–17 % of errors | Exploratory |
| RETFound is better calibrated externally; temperature scaling does not transfer | Table VII; Phase 13 | Exploratory |
| RETFound attribution mass sits more on the border on Messidor-2 | Section "Where do the models look?": 17–29 % vs < 4 % | Location proxy only; attribution methods differ |
| An interrupted run reversed the conclusion | Discussion; `outputs/_archive_provisional_epoch6/` | Documented with both checkpoint hashes |

## 4. [AUTHOR INPUT NEEDED]

1. **Target journal.** It sets the template, word limits and abstract format.
   The current abstract is ~310 words and many journals cap it at 250. The
   current IEEEtran layout suits IEEE J-BHI / Access.
2. **Name form.** The title page uses "Pravin Raj E" (your earlier choice). Your
   latest note says "pravin raj". Confirm which form matches your ORCID record
   (0009-0000-2975-1631, now on the title page).
3. **Code availability.** The public repository URL and Zenodo DOI (marked in
   Declarations).
4. **Ethics wording.** Confirm it against the journal's required wording
   (marked in Declarations).
5. **Data provenance.** In `DATA_PROVENANCE.md`: the Kaggle slugs and versions
   of the Messidor-2 grade and IDRiD mirrors, and the APTOS competition-rules
   clause.
6. **Cover letter.** Fill in the journal name and suggested reviewers in
   `COVER_LETTER.md`.
7. **Recommended before submission, not blocking:**
   - Re-run IDRiD on the official 516-image release with official labels.
   - Add 2–3 more seeds of the RETFound full fine-tune.
   - Recruit two blinded raters for the saliency packet.

## 5. Revision note (story and structure)

- **Introduction** is now a single argument, not a background list:
  - it moves from why the problem matters, to what is known, to three named
    gaps, to why they matter, to the questions;
  - it ends with a short "what we found" that includes the null result and the
    interrupted-run episode.
- **Literature:** the closest prior study (Hou et al. 2025, same datasets) and
  two recent external-validation studies are now cited. The incorrect claim
  that RETFound's cross-dataset DR performance had "not been independently
  tested" was removed.
- **"Related Work" became "Background and positioning":** it no longer repeats
  the introduction, and it states exactly how this study differs from Hou et al.
- **Research questions were reordered** to follow the story (performance → can
  errors be caught → where the models look). The Methods and Results follow the
  same order, and the Results open with a roadmap and question-led subsection
  titles.
- **Head-depth sweep:** the three-table sweep moved to Appendix A, with a
  one-paragraph answer in the Results. It was a digression from the main line.
- **Discussion** opens by answering the question in three sentences. It adds
  "What changed from the interrupted run" as an explicit lesson, keeps the
  mechanism discussion labelled speculative, and replaces clinical-impact
  language with a restrained "Implications".
- **Removed:** an uncited claim about typical APTOS QWK. The bibliography is
  corrected (`efron1994bootstrap` is now `@book`), and three entries were added,
  with metadata retrieved from PubMed.
