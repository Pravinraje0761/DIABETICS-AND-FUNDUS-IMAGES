### Explainability (drop-in Methods text)

For a grade-stratified sample of correctly and incorrectly classified external
images (image identifiers drawn from the Phase 6 prediction files with a fixed
seed), a saliency map was generated for the model's predicted class. For the
convolutional networks this was a Grad-CAM++ map computed at the last
convolutional block; for the Vision Transformer it was a method-matched Grad-CAM
map computed at the last transformer block (gradient-weighted patch-token
activations for the predicted class, class token removed, reshaped to the patch
grid and bilinearly upsampled). Attention-rollout maps are additionally provided
in the Supplement but were more diffuse and border-biased. Because the public
IDRiD mirror used here
provides no lesion segmentation masks and a second expert grader was not
available, clinical plausibility was scored with a pre-registered automated proxy
rather than by hand: a map was counted "proxy-plausible" when its peak lay within
the fundus, outside an automatically estimated optic-disc disk, and outside the
outer border annulus, and when at least half of the top-decile saliency mass fell
in that same lesion-eligible region. The proxy cannot exclude vessel-centred
activation and is reported as a screen; the absence of blinded expert rating is
stated as a limitation. Code: scripts/phase7_explainability.py.
