### Uncertainty quantification (drop-in Methods text)

Predictive uncertainty was quantified as the Shannon entropy of the predictive
softmax distribution. In the primary analysis this was the entropy of a single
forward pass; in a sensitivity analysis it was the entropy of the mean softmax
over T = 30 stochastic forward passes with dropout (and, for the Vision
Transformer, stochastic-depth) layers held active at inference (Monte-Carlo
dropout; Gal and Ghahramani). For each model and external dataset we tested
whether misclassified images carried higher entropy than correctly classified
images with a one-sided Mann-Whitney U test, reporting the rank-biserial
correlation as the effect size, and we computed the area under the ROC curve for
predictive entropy used directly as a binary error detector. To translate this
into an operating characteristic we plotted selective-prediction curves: images
were ranked by entropy and the most uncertain fraction referred out, and the
accuracy, quadratic-weighted kappa and fraction of all errors captured among the
referred images were recorded as a function of retained coverage. Predictive
entropy was additionally broken down by International Clinical Diabetic
Retinopathy grade, stratified both by the reference grade and by the predicted
grade, to check whether the models expressed greater uncertainty on the rarer
sight-threatening grades and whether the large number of grade-0 predictions on
the external sets were confidently made. Analyses used SciPy and scikit-learn;
code is released as scripts/phase8_uncertainty.py and
scripts/phase8_mc_dropout.py.
