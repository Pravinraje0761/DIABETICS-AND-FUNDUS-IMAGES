# Blinded saliency-plausibility rating - instructions

You are shown 160 fundus images. Each file `img/R####.png` has two panels:

* **left**  - the pre-processed colour fundus photograph;
* **right** - the same photograph with the model's saliency map (the region the
  model used for its diabetic-retinopathy grade) overlaid as a red/blue heatmap
  (red = most important).

You are **blinded** to the model, the dataset, the reference grade and the
model's prediction. Rate each map in `rating_sheet_BLANK.csv` (save as
`rater_<yourinitials>.csv`):

| column | values | meaning |
|---|---|---|
| `focus_plausible` | 1 / 0 | 1 if the *bulk* of the red saliency mass sits on retinal tissue a grader could plausibly use for DR assessment (lesions, macula, arcades) and is **not** dominated by the optic disc, the image border, or areas outside the retina. 0 otherwise. |
| `lesion_aligned` | 1 / 0 / NA | 1 if the red saliency overlaps a visible DR lesion (haemorrhage, microaneurysm, hard/soft exudate, neovascularisation). 0 if no such lesion is under the saliency. NA if you cannot tell / image ungradable. |
| `confidence` | 1-5 | 1 = pure guess, 5 = certain. |
| `notes` | free text | optional (e.g. "on disc", "vessel only", "border artefact"). |

Please rate every row. Do not confer with other raters. Return the completed
CSV; keep `blind_id` unchanged.
