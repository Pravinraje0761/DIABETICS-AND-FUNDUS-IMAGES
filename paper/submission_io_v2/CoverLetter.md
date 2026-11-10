---
geometry: a4paper,margin=2.4cm
fontsize: 11pt
---

Pravin Raj E  
Department of Artificial Intelligence and Data Science  
Chennai Institute of Technology, Chennai, Tamil Nadu, India  
pravinraje.aids2024@citchennai.net · ORCID 0009-0000-2975-1631

10 November 2026

The Editor-in-Chief  
*International Ophthalmology*

Dear Editor,

I submit the Original Article **"Retinal foundation-model versus ImageNet pretraining for deep learning grading of diabetic retinopathy: an external validation study"** for consideration in *International Ophthalmology*.

Automated diabetic retinopathy (DR) grading performs well on the data it is developed on but loses accuracy when the camera, population or grading protocol changes. Retinal foundation models such as RETFound are often expected to reduce this loss. This study tests that expectation against a strong conventional baseline, in the terms set out in the journal's recent editorial on retinal AI (Jin and Grzybowski, 2026): incremental value over a strong baseline, distribution shift, transparent reference standards, independent external validation, uncertainty and referral.

Models were developed on APTOS 2019 and evaluated unchanged on Messidor-2 (adjudicated specialist grades) and the official IDRiD release. Averaged over repeated training runs, fully fine-tuned RETFound did not lose less grading agreement than an ImageNet-pretrained EfficientNet-B3, and a frozen RETFound encoder lost more. The study also shows that:

- external performance varied materially between training runs that differed only in the random seed, so single-run comparisons can mislead; the change from a single-run to a multi-run primary analysis is disclosed and both are reported;
- on Messidor-2 every model preserved specificity but graded almost all mild DR as no DR, whereas on IDRiD no DR was often graded mild; matching case mix and merging no and mild DR accounted for most of the Messidor-2 drop, which argues for reporting reference standards and prevalence-independent measures in external validation;
- RETFound's apparent calibration advantage was reproduced by label smoothing alone, and simple confidence was as informative as more elaborate uncertainty scores;
- confidence-based referral of 30% of images to a human grader more than halved missed referable DR on one external dataset but not on the other;
- tested quantitatively against expert lesion masks, saliency maps localised lesions only weakly.

The work is an evaluation of cross-dataset reliability, not a proposal for a new classifier, and makes no claim of clinical readiness. It is reported following TRIPOD+AI, reports sensitivity, specificity and predictive values for referable DR, includes an audit that found 5.9% of labels in a widely shared IDRiD copy differed from the official release, and is accompanied by per-image predictions and the full analysis code.

I confirm that this manuscript is original, has not been published previously, and is not under consideration by any other journal. It has not been posted as a preprint. I am the sole author and have approved the submitted version. The study used only public, de-identified datasets. It received no funding, and I have no competing interests. The use of AI assistants in preparing the code and text is described in the Methods. I would prefer the standard (subscription) publication route.

Thank you for considering this submission.

Yours sincerely,

Pravin Raj E
