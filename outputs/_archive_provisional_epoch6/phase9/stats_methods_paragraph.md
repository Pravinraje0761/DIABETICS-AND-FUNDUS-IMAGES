### Statistical analysis (drop-in Methods text)

All uncertainty estimates were obtained by non-parametric bootstrap resampling of
the relevant test set(s) with replacement (2000 iterations; NumPy PCG64 generator,
seed 42), recomputing each metric on every resample and taking the 2.5th and
97.5th percentiles as the 95% confidence interval. For the internal-to-external
performance drop, the internal APTOS test set and the external test set were
resampled independently within each iteration; for the between-model difference
in that drop, the same resampled image indices were scored for both models so the
comparison is paired at the image level. Areas under the one-vs-rest receiver
operating characteristic curve were compared between models on the same test set
with the two-sided DeLong test (fast midrank algorithm of Sun and Xu), applied to
the clinically standard referable-diabetic-retinopathy binarisation (International
Clinical Diabetic Retinopathy grade >= 2; score = summed probability of grades
2-4) and, secondarily, to each grade one-vs-rest. The significance threshold was
alpha = 0.05, two-sided, with no adjustment for multiple comparisons given the
small pre-specified family of confirmatory contrasts (foundation model vs.
EfficientNet-B3 baseline on each external dataset); secondary per-grade tests are
reported as exploratory. Analyses used Python 3.14, NumPy, pandas, SciPy and
scikit-learn; the bootstrap and DeLong code is released with the study
(`scripts/phase9_stats.py`).
