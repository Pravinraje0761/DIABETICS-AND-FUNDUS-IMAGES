# Cover letter

> Replace `[Journal]` and the reviewer list before sending. Keep to one page.
> A journal-specific hook is offered at the end.

---

Dear Editors,

Please consider the manuscript **"Cross-Dataset External Validation of a Retinal
Foundation Model for Diabetic Retinopathy Grading: Performance, Explainability,
and Uncertainty"** for publication in *[Journal]*.

Retinal foundation models are widely expected to generalise better than
ImageNet-pretrained networks, yet independent, protocol-matched external
validation for diabetic retinopathy (DR) severity grading remains scarce. Under
one matched preprocessing, fine-tuning and evaluation protocol, we compared
RETFound (ViT-Large, masked-autoencoder pretraining on ~1.6 M fundus images;
linear probing and full fine-tuning) with ImageNet-pretrained EfficientNet-B3
(and ResNet-50), developed on APTOS 2019 only and evaluated unchanged on
Messidor-2 (n = 1,744, adjudicated reference standard) and IDRiD (n = 455).

The primary outcome is the quadratic-weighted kappa, and the confirmatory
inference is a pre-specified, Holm-controlled family of four paired contrasts in
internal-to-external kappa drop. Fully fine-tuned RETFound generalised **neither
better nor measurably worse** than the CNN (excess drop +0.018 on Messidor-2 and
+0.022 on IDRiD, both intervals including zero), while RETFound used as a frozen
feature extractor generalised **worse** (Holm p = 0.030 and 0.016). Every model
lost substantial agreement externally (kappa ~0.85-0.88 internally, 0.39-0.50 on
Messidor-2), and softmax entropy did not reliably flag the errors. We also report
that an earlier, hardware-interrupted RETFound training run had suggested a
significant disadvantage that disappeared when the pre-specified protocol was
run to completion, which we think is itself a useful caution for the field.

The submission includes a full reproducibility package: code, split lists,
per-image predictions, checkpoint SHA-256 hashes, an exact environment lock, a
documented audit of each dataset copy against its canonical release, and the
completed TRIPOD+AI checklist. The work is original and not under consideration
elsewhere. The author declares no conflicts of interest. Use of an AI coding
assistant is disclosed in the manuscript.

Suggested reviewers (no conflict of interest):
- [Name, affiliation, email] - retinal foundation models / RETFound
- [Name, affiliation, email] - external validation of DR AI
- [Name, affiliation, email] - reporting standards for prediction models (TRIPOD+AI)
- [Name, affiliation, email] - uncertainty / explainability in medical imaging

Sincerely,
Pravin Raj E
Independent researcher - epravinraj0761@gmail.com

---

### Optional journal-specific hooks (pick one, delete the rest)

- **Diagnostics / Journal of Imaging (MDPI):** "...and because it fits the
  journal's scope in AI-assisted diagnostic imaging."
- **Translational Vision Science & Technology (ARVO):** "...and because a
  rigorously reported null external-validation result, with a full
  reproducibility package, advances translational understanding of where retinal
  AI does and does not transfer."
- **Ophthalmology Science:** "...and because it tests a widely repeated
  assumption about retinal foundation models with external validation rather
  than internal test performance."
