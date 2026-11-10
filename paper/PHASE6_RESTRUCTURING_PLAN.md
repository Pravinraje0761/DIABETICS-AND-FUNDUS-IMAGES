# Phase 6 — restructuring plan (International Ophthalmology, Original Article)

Status 2026-11-06.

- **Data fixes in progress:**
  - Official IDRiD prepared: 516 images; the mirror had **27 wrong labels**, not "at least 7".
  - GPU jobs: official-IDRiD predictions, the label-smoothing control, and 2 extra seeds.
- **Messidor-2 eye pairing:** waiting on ADCIS. Until then, the halved-sample analysis is kept as a genuine worst-case bound. With at most 2 images per examination, the design effect is at most 2.
- **Numbers:** every number in Results will be taken from the official-IDRiD re-run. Nothing below commits to a figure.

## Format decision

LaTeX with the **Springer Nature template (`sn-jnl`, `sn-basic` reference style)**, compiled on Overleaf. The journal accepts LaTeX with this template, and it keeps your equations and tables intact. A .docx export can follow if the submission system asks for one.

## Title (working)

*Retinal foundation-model versus ImageNet pretraining for diabetic retinopathy grading across datasets: how much of the external performance drop reflects the reference standard?*

## Abstract: structured, 150–250 words

Sections: **Purpose** (1–2 sentences) · **Methods** (datasets, models, primary estimand, analyses) · **Results** (primary contrast, decomposition, uncertainty, localisation) · **Conclusion** (no claim of superiority, equivalence or clinical readiness).

## Keywords (6)

Diabetic retinopathy; Fundus photography; Deep learning; Foundation model; External validation; Reference standard.

## Main-text structure (at most 3 heading levels; target 5,000–6,000 words)

| Section | Content | Words |
|---|---|---|
| **Introduction** (4 paragraphs) | 1. Burden of DR screening. 2. CNNs vs retinal foundation models. 3. What is known, citing Hou 2025, **Poyrazer 2026** and **Yew 2026**: frozen RETFound transfers worse, and fine-tuned models perform about the same, but nobody has asked how much of the external drop comes from the reference standard and case mix. 4. Objective and pre-specified hypotheses | ~650 |
| **Methods** | Study design · Datasets and reference standards (**label-harmonisation table**) · Preprocessing · Models and fine-tuning (**side-by-side recipe table, honest wording**) · External testing and leakage controls · Outcomes and statistics (primary estimand, Holm, worst-case clustering) · Shift decomposition · Uncertainty and selective referral · Saliency localisation against lesion masks · Ethics · Use of generative AI (**moved here, as the journal requires**) | ~2,000 |
| **Results** | 1. Datasets (Table 1, Fig 1). 2. Internal performance. 3. External performance and the primary contrast (Table 2, Fig 2). 4. Where the drop comes from: decomposition (Table 3, Fig 3). 5. Referable DR: sensitivity/specificity at the clinical threshold. 6. Uncertainty and referral vs simple confidence (Fig 4). 7. Lesion localisation (Fig 5) | ~1,600 |
| **Discussion** | Principal findings · Comparison with Poyrazer, Yew, Hou and Li · What the reference-standard effect means for ophthalmologists reading AI validation papers · Uncertainty and explainability in context · Strengths and limitations · Future work | ~1,300 |
| **Conclusion** | 3–4 sentences | ~100 |

## Display items in the main text (5 figures + 4 tables)

| Item | Content | Source |
|---|---|---|
| Table 1 | Datasets: source, camera, population, reference standard, grade distribution, referable prevalence | Phases 11, 25, 28 |
| Table 2 | Training configuration side by side (what is matched, what differs) | hyperparameter JSONs |
| Table 3 | Internal and external QWK, primary ΔDrop with 95% CI and Holm p; referable sensitivity/specificity | Phases 23, 19 (re-run) |
| Table 4 | Shift decomposition: raw / prevalence-reweighted / 4-level / both | Phase 25 (re-run) |
| Fig 1 | Study design (redrawn without in-figure title; legible in greyscale) | new |
| Fig 2 | Forest plot: internal vs external QWK and ΔDrop | Phase 10 (re-run) |
| Fig 3 | Confusion structure at the No-DR/mild boundary, Messidor-2 vs IDRiD | Phase 25 |
| Fig 4 | Missed-referable rate vs referral rate, MSP vs entropy (human-in-the-loop) | Phase 24 |
| Fig 5 | Lesion-mask localisation: examples + summary with controls | Phase 26/26b |

## Moved to Online Resources (supplementary)

- **Online Resource 1:** extended methods (preprocessing details, full hyperparameters, environment, checkpoint hashes, leakage and duplicate checks, dataset audits including the IDRiD mirror).
- **Online Resource 2:** extended results tables (per-class CIs, calibration and temperature scaling, DeLong, OOD detection, input-distribution ablations, seed and loss sensitivity, head-depth sweep, resolution-matched localisation, MC dropout).
- **Online Resource 3:** the reporting checklists: STARD-AI, CLAIM 2024, TRIPOD+AI.
- **Online Resource 4:** per-image predictions and split lists (CSV).

Removed from the main narrative: the interrupted-run story (one sentence in Limitations plus Online Resource 1), the head-depth appendix (→ Online Resource 2), and the automated plausibility proxy (superseded by lesion masks).

## Wording rules applied throughout

- "No detectable difference (95% CI …)", never "indistinguishable" or "equivalent". No equivalence margin was pre-specified.
- "Matched data pipeline, selection rule and loss weighting; architecture-standard optimiser recipes", never "identical protocol".
- Saliency is described as localisation evidence *against lesion masks*, not as proof of clinical interpretability.
- No clinical-readiness or safety claims.

## Decision needed from you

Approve this structure, or tell me what to change. Once the GPU run finishes and I have re-run the statistics on the official IDRiD, I'll write the manuscript to this plan.
