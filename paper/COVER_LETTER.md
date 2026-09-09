# Cover letter

> Replace `[Journal]`, the bracketed fields, and the reviewer list before
> sending. Keep to one page. A journal-specific hook is offered at the end.

---

Dear Editors,

Please consider our manuscript, **"Cross-Dataset External Validation of a Retinal
Foundation Model for Diabetic Retinopathy Grading: Performance, Explainability,
and Uncertainty,"** for publication in *[Journal]*.

Retinal foundation models are increasingly promoted for clinical retinal image
analysis, yet independent, protocol-matched external validation for diabetic
retinopathy (DR) severity grading remains scarce. Under a single, fully matched
preprocessing, fine-tuning and evaluation protocol, we compared a fine-tuned
retinal foundation model (RETFound, ViT-Large, masked-autoencoder pretraining on
~1.6 M fundus images) with an ImageNet-pretrained convolutional baseline
(EfficientNet-B3; ResNet-50 as secondary) on two fully independent external
datasets, Messidor-2 (n = 1,744, adjudicated reference standard) and IDRiD
(n = 455), after development on APTOS 2019 only.

Our principal finding is **negative and, we believe, useful**: fine-tuning
RETFound on APTOS did **not** produce better cross-dataset generalisation than
the ImageNet CNN. Every model - foundation and baseline alike - lost substantial
ordinal agreement on external data (quadratic-weighted kappa falling from
approximately 0.87 internally to 0.37-0.50 on Messidor-2), and RETFound's
external kappa was no higher than the baseline's on either external set. The
result is consistent across RETFound linear-probing and full fine-tuning. We add
protocol-matched explainability (Grad-CAM++ / attention rollout against an
a-priori plausibility rubric) and predictive-uncertainty analyses (Monte-Carlo
dropout, selective prediction), bootstrap 95% confidence intervals for every
headline metric, DeLong tests for paired AUC comparisons, and we release all
code, split definitions and per-image predictions. Reporting follows TRIPOD+AI
(completed checklist included as Supplementary Material).

We believe this is relevant to *[Journal]*'s readership because it directly and
transparently tests a widely repeated assumption about foundation models in
ophthalmic imaging, and because it exemplifies the external-validation-first
evidence standard that the field increasingly asks for. The work is original,
not under consideration elsewhere, and all authors have approved this
submission. We declare **[no conflicts of interest / see disclosure]**. Use of an
AI coding assistant is disclosed in the manuscript.

Suggested reviewers (no conflict of interest with the authors):
- [Name, affiliation, email] - retinal foundation models / RETFound
- [Name, affiliation, email] - external validation of DR AI
- [Name, affiliation, email] - reporting standards for prediction models (TRIPOD+AI)
- [Name, affiliation, email] - uncertainty / explainability in medical imaging

We would be glad to respond to any questions.

Sincerely,
[Corresponding author], on behalf of all authors
[Affiliation] - [email] - [ORCID]

---

### Optional journal-specific hooks (pick one, delete the rest)

- **Diagnostics / Journal of Imaging (MDPI):** "...and because it fits the
  journal's scope in AI-assisted diagnostic imaging; we note the open Special
  Issue on [exact title] and would be happy to be considered for it."
- **Translational Vision Science & Technology (ARVO):** "...and because a
  rigorously reported negative external-validation result, with a full
  reproducibility package, advances translational understanding of where retinal
  AI does and does not transfer."
- **Ophthalmology Science:** "...and because it addresses clinical readiness of
  retinal AI head-on, with external validation and an uncertainty-aware referral
  analysis rather than internal test performance alone."
