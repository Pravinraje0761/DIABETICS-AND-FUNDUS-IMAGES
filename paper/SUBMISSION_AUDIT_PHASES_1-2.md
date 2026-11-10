# International Ophthalmology submission — Phase 1 & 2 audit (with preliminary Phase 3–5 findings)

Prepared 2026-11-06. **No manuscript text has been changed.** This report covers:

- **Verified facts:** marked ✔, each with its source.
- **Recommendations:** marked →.
- **Editorial opinion:** marked (opinion).

Two kinds of source are used:

- **Journal requirements** come from the official pages, read 2026-11-06: https://link.springer.com/journal/10792/submission-guidelines and https://link.springer.com/journal/10792/how-to-publish-with-us.
- **Literature** was retrieved from PubMed on 2026-11-06. For the ten International Ophthalmology papers, I read the abstracts only, not the full text.

---

## PHASE 1 — Document inspection

### 1.1 What exists

| Item | Status |
|---|---|
| Manuscript | `paper/main_standalone.tex`: IEEEtran, about 22 pp. Abstract 308 words, unstructured. Main text (Introduction→Conclusion) about 9,630 words, plus about 1,950 words of Reproducibility and Appendix. 10 figures, 11 tables (3 in the appendix), 27 cited references |
| Supporting documents | TRIPOD+AI checklist; cover letter (generic); `DATA_PROVENANCE.md`; `CHECKPOINTS.md` (hashes plus evidence that the external-set lock was kept); response-to-reviewers |
| Experiment outputs | Phases 2–27. Per-image prediction files exist for every model × dataset. Checkpoint hashes are recorded |

### 1.2 Numbers checked against the raw prediction files ✔

| Quantity | Manuscript | Recomputed | Match |
|---|---|---|---|
| QWK internal / Messidor-2 / IDRiD, all 4 models | Table IV | `phase6/master_results.csv`; independent recompute in Phases 11, 24 and 25 | ✔ (<5e-4) |
| Confirmatory ΔDrop (RETFound full vs EfficientNet-B3): +0.018 / +0.022; linear probe: +0.076 / +0.105 | Abstract, Table V | Phase 23 = Phase 25 exactly | ✔ |
| External ECE: RETFound 0.09–0.13 vs CNN 0.26–0.29 | Abstract, Table VII | `phase11/calibration.csv` (0.095–0.129 vs 0.260–0.294) | ✔ |
| Referable-DR AUC on IDRiD, EfficientNet-B3 vs RETFound full: 0.962 vs 0.885 | Abstract | `phase19` | ✔ |
| Dataset counts: APTOS 3662 (train 2562 / val 367 / test 733); Messidor-2 1744 of 1748 (4 ungradable excluded); IDRiD 455 of 516 | Tables I and III | manifests and audit | ✔ |

**The numbers match the primary outputs.** Nothing in the manuscript disagrees with the authoritative files.

### 1.3 Statements made out of date by the round-2 analyses (Phases 24–26b, 2026-11-06)

1. **RQ3.** The manuscript says there are "no lesion masks; location proxy only". Quantitative lesion-mask localisation now exists (Phase 26: IDRiD Segmentation, 81 images, with controls; Phase 26b: resolution-matched control). The finding is new:
   - RETFound's maps sit on expert-annotated lesions more than the CNN maps do:
     - pixel AUROC: 0.67 vs 0.61
     - pointing hit (saliency peak lands on a lesion): 0.64 vs 0.48
     - top-10% precision (share of the most-salient 10% of pixels that are lesion): 0.19 vs 0.12
   - CNN maps are not significantly better than a centre-Gaussian prior (EfficientNet-B3: lift p = 0.10, AUROC p = 0.08).
   - **Caveat:** about two-thirds of RETFound's lift advantage comes from its finer 14×14 map grid. At the matched 7×7 resolution:
     - The lift difference is no longer significant: 1.44 vs 1.37, p = 0.17.
     - The pixel-AUROC advantage holds: +0.055, p = 0.0004.
     - The top-10%-precision advantage holds: +0.036, p = 0.003.
   - In absolute terms all maps localise weakly. The abstract's sentence about saliency on the image border must change.
2. **RQ2.** The manuscript says entropy "carried none for fully fine-tuned RETFound". This is a property of entropy as the score, not of the model:
   - MSP (1 − maximum softmax probability) and margin (gap between the top two probabilities) recover a significant internal signal (0.635 / 0.666 vs 0.516).
   - On IDRiD, no score works (0.46–0.53).
   - Selective-prediction accuracy gains on Messidor-2 are misleading, because they come from referring diseased cases. The missed-referable metric (Phase 24) is the clinically correct way to report it.
3. **The linear-probe Messidor-2 deficit (+0.076) disappears after prevalence reweighting** (+0.004 [−0.060, 0.069]). The IDRiD deficit persists and grows. The claim "the linear probe generalised worse on both sets" must be qualified per dataset.
4. **Shift decomposition (Phase 25).** Prevalence reweighting plus merging grades 0 and 1 removes 52–73% of the Messidor-2 drop. The two external sets fail in opposite directions:
   - Messidor-2: mild and moderate cases are under-called.
   - IDRiD: grade-0 images are over-called as mild.

   This is the most clinically relevant result in the study and is not yet in the manuscript.

### 1.4 Unresolved data issues

| Issue | Severity | Fix |
|---|---|---|
| IDRiD is a third-party mirror: 455 of 516 images, and at least 7 labels differ from the official grades | **Major** | Download the official **"B. Disease Grading"** subset from the same IEEE DataPort page you used for "A. Segmentation". Then re-run **inference only** (no retraining) on all 516 images with official labels. Minutes on your GPU |
| Messidor-2 has 2 eyes per examination, and the pairing was not used, so bootstrap units are not independent | **Major** | Get the ADCIS Messidor-2 examination list (`messidor-2.csv`, distributed with the official release) and switch to an examination-level cluster bootstrap. Poyrazer et al. 2026 did exactly this on Messidor-2, so reviewers will expect it |
| APTOS has no patient IDs, so the internal split is image-level | Minor (stated) | Keep as a limitation. It does not affect the external sets |

---

## PHASE 2 — Journal verification

### 2.1 Compliance table (International Ophthalmology, official guidelines read 2026-11-06)

M = mandatory, R = recommended.

| # | Requirement | Official instruction | Current status | Required correction | Priority |
|---|---|---|---|---|---|
| 1 | Article type | Original Article (Reviews and Letters also accepted; no case reports) | Original research | Submit as Original Article | M |
| 2 | Scope | Clinical research across ophthalmology; "very high quality … clinical and social impact" | Methodological AI study on public datasets | Reframe around the clinical question (grading reliability across sites and reference standards) (opinion) | M |
| 3 | Source files | Editable **.docx or LaTeX**; LaTeX users should use Springer Nature's template and `sn-basic.bst` | IEEEtran LaTeX | **Convert to the Springer Nature LaTeX template (sn-jnl), or to .docx** | **M** |
| 4 | Abstract | **Structured, 150–250 words: Purpose / Methods / Results / Conclusion** | 308 words, unstructured | Rewrite | **M** |
| 5 | Keywords | **4–6** | 9 | Cut to 6 | **M** |
| 6 | Headings | At most 3 levels | 3 levels (section, subsection, subsubsection) | Keep at most 3 after restructuring | M |
| 7 | Word / page limit | **None stated** | ~9,630 words, 10 figures, 11 tables | No mandatory cut. Recommended (opinion): ~5,000–6,000 words, 5 figures, 5 tables; the rest to Online Resources | R |
| 8 | Title page | Concise title; names; affiliation (institution, department, city, state, country); corresponding author email; ORCID | Name, "independent researcher", email, ORCID | **Author input:** final name form; city and country for "Independent researcher" | M |
| 9 | Abbreviations | Define at first use; use consistently | Mostly done | Check after conversion | M |
| 10 | References | Numbered in square brackets; Springer style with DOI links; LTWA journal abbreviations | IEEEtran style; some entries have no DOI | Restyle with `sn-basic.bst`, add DOIs, add the new references (§3) | M |
| 11 | Tables | Arabic numerals; caption; footnotes as superscript letters or asterisks | Roman numerals (IEEE) | Renumber and restyle | M |
| 12 | Figures | EPS (vector) / TIFF; at least 300 dpi (halftone), 600 (combination), 1200 (line); Helvetica/Arial 8–12 pt; **no titles inside figures**; files named Fig1…; caption "**Fig. 1**" with no final punctuation; must be legible in greyscale; colour free online | PDF/PNG; TikZ pipeline; some PNGs | Re-export all figures to journal spec; remove in-figure titles; use patterns as well as colour | M |
| 13 | Supplementary | Cite as "Online Resource N". Each file must carry the article title, journal, author names, and the corresponding author's affiliation and email | Supplement described inline | Build Online Resources 1–N with headers | M |
| 14 | Statements & Declarations | Funding; competing interests; author contributions (encouraged); ethics; consent to participate; consent to publish; data availability (**required**). Missing statements mean the submission is returned as incomplete | Funding ✔, competing interests ✔, contributions ✔, ethics (wording open), data ✔ (repository URL missing). **No consent statements** | Add "Consent to participate / publish: not applicable (secondary analysis of public, de-identified datasets)" **only after you confirm it**; add repository URL/DOI | **M** |
| 15 | Competing interests / contributions in the submission system | Must also be entered in the online interface; only that version is published | — | Enter at submission | M |
| 16 | **AI / LLM use** | LLMs cannot be authors. **LLM use must be documented in the Methods section.** Copy-editing need not be declared | The statement sits in Declarations; it says AI assisted with code, figures and "substantial portions" of the text | **Move to Methods** (e.g. a "Use of generative AI" subsection) and keep it accurate | **M** |
| 17 | Ethics | Name the committee and reference, or state the reason for exemption; Declaration of Helsinki | Exemption reasoning present; wording flagged | Author confirms the final wording | M |
| 18 | Reporting guideline | EQUATOR guidelines recommended (STARD among the examples) | TRIPOD+AI checklist exists | Add **STARD-AI** (Sounderajah et al., *Nat Med* 2025) and **CLAIM 2024** (Tejani et al., *Radiol Artif Intell* 2024) as Online Resources | R |
| 19 | Originality | Not published elsewhere; no simultaneous submission; co-author and institution approval | — | **Author confirms:** is there any preprint (arXiv/medRxiv)? The `\markboth` says "Preprint — under review" | M |
| 20 | Reviewer suggestions | Welcome; need institutional email or another verifiable ID | None | Optional | R |
| 21 | Costs | **Hybrid journal.** Subscription route: no APC. Open Choice APC: £2,590 / $3,890 / €2,990. Colour illustrations free. No submission fee mentioned | — | **Choose the subscription route** (no charge). Decline Open Choice and paid services | — |
| 22 | Submission portal | https://submission.nature.com/new-submission/10792/3 (linked from the journal page) | — | — | — |
| 23 | Peer-review model | Not stated on the guideline page | — | Prepare an anonymisable version only if the system asks | — |

### 2.2 What International Ophthalmology has published: the ten papers you listed

All ten were found in PubMed. The table is based on abstracts only.

| # | Paper (first author, year) | DOI | Modality / task | Data | External test | Statistics reported | Relevance to us |
|---|---|---|---|---|---|---|---|
| 1 | Nagasawa 2019, UWF deep learning for treatment-naïve PDR (**PubMed type: Letter**) | [10.1007/s10792-019-01074-z](https://doi.org/10.1007/s10792-019-01074-z) | UWF fundus, PDR vs non-PDR | 378 images, single centre | No | AUC, sens/spec | Low: different modality and binary task |
| 2 | Yuan 2023, GAN with multi-attention for fundus lesion segmentation | [10.1007/s10792-023-02911-y](https://doi.org/10.1007/s10792-023-02911-y) | Fundus lesion segmentation | **IDRiD (MA/HE/EX/SE)** | No | Dice per lesion | **Medium:** same IDRiD lesion masks we now use. Cite when introducing them |
| 3 | Govindharaj 2025, "Capsule network-based deep learning for early and accurate **DR** detection" | [10.1007/s10792-024-03391-4](https://doi.org/10.1007/s10792-024-03391-4) | **The abstract describes glaucoma OD/OC segmentation, not DR** | "ten benchmark sets" | Unclear | Accuracy | **Do not cite as DR evidence:** the title and abstract disagree |
| 4 | Li 2025, HRF segmentation in OCT (a correction was published) | [10.1007/s10792-025-03439-z](https://doi.org/10.1007/s10792-025-03439-z) | OCT segmentation | Not stated in abstract | No | SE, Dice | Low: OCT |
| 5 | Mahapadi 2025, explainable NPDR detection (RQ-EGMCN) | [10.1007/s10792-025-03528-z](https://doi.org/10.1007/s10792-025-03528-z) | Fundus DR classification + saliency | One dataset | No | Accuracy/precision/recall ~95% | Medium: a fundus DR paper with qualitative saliency only. Contrast with our quantitative localisation |
| 6 | Lisha 2025, hybrid ILink-SqNet | [10.1007/s10792-025-03660-w](https://doi.org/10.1007/s10792-025-03660-w) | Fundus retinal disease classification | Not stated | No | Precision 0.951 | Low |
| 7 | Alizadeh Eghtedar 2024, CAD for OCT retinal disorders | [10.1007/s10792-024-03033-9](https://doi.org/10.1007/s10792-024-03033-9) | OCT classification + heatmaps | UCSD subset; 1,000 test images | No | Acc/sens/F1 ~97.6% | Low: OCT |
| 8 | Gueddena 2024, DME/AMD from OCT | [10.1007/s10792-024-03115-8](https://doi.org/10.1007/s10792-024-03115-8) | OCT classification | Duke public + Tunisian private | Two datasets | Accuracy 99.53% | Low: OCT. Partial precedent for a second dataset |
| 9 | Kumar 2024, SAU-Net with Harris hawks optimisation, vessel/OD segmentation | [10.1007/s10792-024-03279-3](https://doi.org/10.1007/s10792-024-03279-3) | Fundus segmentation | DRIVE, STARE | No | Accuracy, specificity | Low |
| 10 | Shwetar 2025, ML for ABCA4-related retinopathy (scoping review) | [10.1007/s10792-025-03690-4](https://doi.org/10.1007/s10792-025-03690-4) | Review | 15 studies | — | — | Low |

**What this tells us** (opinion, based on abstracts):
- International Ophthalmology publishes imaging-AI papers regularly. They are mostly method proposals tested on a single dataset and reported as headline accuracy, without confidence intervals, calibration or external testing.
- None compares pretraining strategies, and none evaluates generalisation as its primary question.
- Our manuscript is methodologically far more rigorous than this sample. The risk is the opposite: a clinical editor may find it statistics-heavy and long. The paper must lead with a clinical question and clinical answers, with the statistical machinery in Methods and the Online Resources.
- Only #2 (IDRiD lesion masks) and #5 (fundus DR with saliency) are directly relevant. #3 should be avoided.

---

## PRELIMINARY PHASE 3–5 FINDINGS (for discussion before restructuring)

### 3.1 Novelty: two 2026 papers narrow the gap (Critical until addressed)

According to PubMed:

- **Poyrazer et al. 2026, *Front Med*** ([10.3389/fmed.2026.1815982](https://doi.org/10.3389/fmed.2026.1815982)):
  - Design: **frozen** RETFound vs EfficientNet-B0 vs MedSigLIP. Developed on **APTOS 2019**, externally tested on **Messidor-2** with an **examination-level cluster bootstrap**.
  - Findings: RETFound was **worse** than the ImageNet baseline externally, all models failed on mild DR, and temperature scaling did not transfer.
  - **This overlaps directly with our linear-probe arm and our calibration findings.**
- **Yew et al. 2026, *Lancet Digit Health*** ([10.1016/j.landig.2026.101031](https://doi.org/10.1016/j.landig.2026.101031)):
  - Design: RETFound vs ResNet50 / ViT-B / SwinV2. DR trained on APTOS; IDRiD and Messidor-2 among the external sets.
  - Findings: performance was **similar when fine-tuned on full data**, with a RETFound advantage only in small-data regimes.
- Also relevant:
  - **Li et al. 2026, *Int J Med Inform*** ([10.1016/j.ijmedinf.2026.106503](https://doi.org/10.1016/j.ijmedinf.2026.106503)): foundation models generalised poorly to unseen ultra-wide-field populations, with poor calibration.
  - **Alutaibi et al. 2026, *Diagnostics*** ([10.3390/diagnostics16193109](https://doi.org/10.3390/diagnostics16193109)): EfficientNet-B3 vs RETFound. A pooled-cohort confound was later found and the headline conclusion withdrawn. This is a cautionary precedent supporting our provenance audit.
- Already cited: Hou et al. 2025 (DINOv2 vs RETFound on the same three datasets).

**Assessment.** The bare question "does RETFound generalise better than an ImageNet CNN for DR?" is already answered "no, or not under full-data fine-tuning" by two 2026 papers. A reviewer will find them, and the current manuscript cites neither. → What remains defensible and new, in order of clinical value:

1. **How much of the external drop is the model, and how much is the reference standard and case mix?**
   - 52–73% of the Messidor-2 drop is explained by prevalence plus the No-DR/mild boundary.
   - On IDRiD, EfficientNet-B3's drop disappears entirely after both corrections.
   - The two external sets fail in opposite directions.
   - None of the papers above decomposes the drop this way.
2. **Full fine-tuning under a matched protocol, with the internal-to-external drop as the pre-specified, multiplicity-controlled estimand on ordinal five-grade grading.** Poyrazer tested frozen encoders only; Yew used binary AUC.
3. **Quantitative lesion-mask localisation with controls.** RETFound localises lesions better than the CNN at matched resolution (pixel AUROC), but every model localises weakly in absolute terms, and the CNN maps are close to a centre prior.
4. **Whether uncertainty supports referral, compared against simple confidence baselines,** with a human-in-the-loop missed-referable metric.

**Proposed repositioning** (opinion; needs your agreement). Title along the lines of: *"Retinal foundation-model versus ImageNet pretraining for diabetic retinopathy grading across datasets: how much of the external performance drop reflects the reference standard?"*

- The central question in your master prompt stays the primary RQ.
- The decomposition becomes the key clinical message for ophthalmologists.
- The interrupted-run episode moves to Limitations or an Online Resource. A clinical journal will not reward it as a headline.

### 3.2 Methodological issues found (beyond §1.4)

| Issue | Category | Severity | Action |
|---|---|---|---|
| **The manuscript says "one fine-tuning protocol … identical", but recipes differ.** RETFound: layer decay 0.65, drop-path 0.2, **label smoothing 0.1**, batch 16, warm-up 3. CNNs: no label smoothing, batch 32, weight decay 1e-4 vs 0.05. Data, splits, preprocessing, augmentation, class weighting and early stopping *are* identical | Fairness / honesty | **Major** | Reword to "matched data pipeline and selection rule, architecture-standard optimiser recipes", and put a side-by-side configuration table in the paper. **Label smoothing confounds the calibration and entropy comparisons:** RETFound's "better calibration" and "flat entropy" may come from label smoothing, not pretraining. → Run one EfficientNet-B3 with label smoothing 0.1 (≈1 GPU-hour) as a sensitivity analysis, or soften the calibration claim |
| EfficientNet-B3 trained at 224 px (its native resolution is 300) | Fairness | Minor | Already tested in Phase 15c; state it |
| Full fine-tune has a single seed | Reproducibility | Major (reviewers will ask) | 2 more seeds ≈ 2 × 3.7 GPU-hours; or state it as a limitation with the CNN and linear-probe seed spreads as context |
| "Indistinguishable internally" | Statistics wording | Minor | No equivalence margin was pre-specified, so report as "no detectable difference (95% CI −0.014 to 0.029)" and avoid "indistinguishable" or "equivalent" |
| No label-harmonisation table | Reporting | Major for an ophthalmology journal | Add a table of the original label definitions (APTOS 0–4 clinician scale; Messidor-2 adjudicated ICDR; IDRiD ICDR-based) → common ICDR scale, noting that matching numbers ≠ matching criteria |
| Subjective plausibility rubric without raters | Explainability | Minor now | With lesion-mask results available, either drop the rubric or obtain 2 blinded raters. Phase 27 is ready |
| External sets not used for tuning | Leakage | ✔ Satisfactory | Git-dated lock evidence in `CHECKPOINTS.md` |
| Duplicates / leakage | Leakage | ✔ Satisfactory | MD5 plus embedding checks; no cross-dataset duplicates |

### 4. Strict-editor view (preliminary)

| Category | Rating | Category | Rating |
|---|---|---|---|
| Journal scope | Minor | Explainability | Satisfactory once Phase 26 is in |
| Clinical relevance | **Major** | Uncertainty | Satisfactory once Phase 24 is in |
| Scientific novelty | **Critical** (uncited 2026 overlap) | Clinical interpretation | Minor |
| Methodological validity | **Major** (recipe mismatch, label-smoothing confound) | Presentation | **Major** (length, IEEE format, 21 display items) |
| Statistical validity | Major (Messidor-2 clustering) | Reporting guidelines | Minor (add STARD-AI, CLAIM) |
| External validation quality | Major (IDRiD mirror) | Research ethics | Minor (wording, consent statements) |
| Reproducibility | Satisfactory (code, hashes, logs) | Submission compliance | **Critical** (abstract, format, AI-use placement) |
| Dataset transparency | Satisfactory | | |

**Five most serious rejection risks:**
1. Novelty overlap with Poyrazer 2026 and Yew 2026, neither cited.
2. Non-compliant format: unstructured 308-word abstract, IEEE template, AI-use statement not in Methods. This is desk-rejection material, but cheap to fix.
3. External test built on an incomplete, partly mislabelled IDRiD mirror when the official release is freely available.
4. Messidor-2 eye pairs treated as independent when the pairing is obtainable.
5. A "matched protocol" claim that the hyperparameter files contradict, with label smoothing confounding the calibration and uncertainty findings.

**As it stands:** desk rejection is likely, mainly because of format and length, and the novelty overlap would sink it at review. **After the P1 items below:** the study is a reasonable candidate to send to review (opinion).

### 5. Experimental improvement plan

| Priority | Item | Why | Effort | Needs from you |
|---|---|---|---|---|
| **P1** | Re-score on the **official IDRiD Disease-Grading release** (516 images, official labels) | Removes the mirror's missing images and label errors | ~15 min GPU + my reruns | Download "B. Disease Grading" from IEEE DataPort |
| **P1** | **Messidor-2 examination pairing** → cluster bootstrap for every CI and test | Statistical validity; Poyrazer did it | My reruns (CPU) | The ADCIS `messidor-2.csv` pairs file (comes with the official ADCIS Messidor-2 download) |
| **P1** | Cite and differentiate the 2026 papers; reposition | Novelty | Writing | Your agreement on the positioning |
| **P2** | EfficientNet-B3 with label smoothing 0.1 | Separates pretraining from label smoothing in the calibration and entropy findings | ~1 GPU-hour | Run a .bat I'll prepare |
| **P2** | 2 more RETFound full fine-tune seeds | Single-seed criticism | ~7–8 GPU-hours | GPU time |
| **P3** | 2 blinded raters for the saliency packet | Only if the subjective rubric stays | Human time | Raters |

### Information I need from you (I will not guess)

1. **Positioning:** agree with §3.1, or keep the current framing?
2. **Name and affiliation:** "Pravin Raj E" or "Pravin Raj"? City and country for "Independent researcher"?
3. **Prior posting:** has this manuscript been posted as a preprint or submitted anywhere? (The header says "Preprint — under review".)
4. **Repository:** public GitHub URL and Zenodo DOI (or permission to leave "available on request" until acceptance).
5. **Ethics and consent wording:** confirm that the "not applicable — public, de-identified, secondary analysis" wording is acceptable to you.
6. **Data:** can you download the official IDRiD "B. Disease Grading" and the ADCIS Messidor-2 pairs file?
7. **GPU:** are you willing to run the label-smoothing control (~1 h) and 2 extra seeds (~8 h)?

**Is the manuscript ready to proceed to Phase 6 (restructuring)?** Partly. The restructure and the journal conversion can start now. The final numbers in Results must wait for the two P1 data fixes, because both change IDRiD values and every CI.
