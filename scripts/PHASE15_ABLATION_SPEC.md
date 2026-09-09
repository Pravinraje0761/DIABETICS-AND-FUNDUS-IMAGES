# Phase 15 -- controlled degradation ablation (spec, not yet executed)

The paper's #1 stated follow-up: move from *documenting* RETFound's external
collapse to *explaining* it. Three preprocessing/representation knobs, each
toggled in isolation, re-running external inference only (no retraining) for the
primary model pair **EfficientNet-B3** and **RETFound full fine-tune** on the
three datasets (APTOS-internal test, Messidor-2, IDRiD).

Primary readout per (model, dataset, variant): QWK, accuracy, macro-F1,
referable-DR AUC, and the **change in internal->external QWK drop vs. the
baseline variant**. A variant "recovers" generalisation if it shrinks
RETFound's excess drop (currently +0.124 on Messidor-2) toward zero.

Output: `outputs/phase15/ablation_master.csv` + `ablation_summary.md`, plus
per-(model,dataset,variant) `*_preds.csv` in the Phase-6 schema so
`phase9_stats.py` / `phase13_temperature_scaling.py` can consume them.

---

## Variant A -- CLAHE on/off  (cheapest; do first)

`scripts/preprocessing.py::preprocess_fundus(..., clahe=True)` already exists;
it needs `pip install opencv-python-headless` (add to `requirements.txt`).

- **Do NOT** run `phase3_preprocess.py --clahe` -- it overwrites
  `data_processed/` and the committed manifests.
- New script applies CLAHE to the *already-processed* 224px PNG at load time
  (`_apply_clahe` on the uint8 array before the ImageNet-norm transform). Caveat
  to state in the paper: CLAHE is applied post-resize rather than on the
  full-res frame, so this is an approximation of the true preprocessing toggle.
- Alternative (cleaner, more work): re-preprocess the 3 datasets from originals
  with `clahe=True` into `data_processed_clahe/` + sibling manifests, then run
  normal `predict.py`.
- Hypothesis: CLAHE normalises inter-site contrast; if RETFound's Messidor-2
  drop shrinks materially with CLAHE, acquisition-contrast shift is a driver.

## Variant B -- colour-transfer normalisation  (numpy only, no new deps)

Match each external image's colour statistics to APTOS. Two options:
1. **Per-channel z-score match** (simple): compute APTOS-train per-channel mean
   and std of raw [0,255] pixels over the fundus mask; for each external image
   `x_c <- (x_c - mean_ext_c) / std_ext_c * std_aptos_c + mean_aptos_c`, clip
   [0,255].
2. **Reinhard LAB transfer** (standard): same idea in LAB space.

Compute APTOS-train stats once (`data_splits/aptos_manifest.csv`, `split==train`,
processed PNGs), cache to `outputs/phase15/aptos_channel_stats.json`. Apply the
transfer at inference on the 224px image before ImageNet normalisation.
Hypothesis: if colour transfer recovers external QWK, RETFound fitted APTOS
colour/illumination statistics rather than lesion morphology (this is the
mechanism the Discussion currently argues for -- Variant B is its direct test).

## Variant C -- input resolution 224 vs 384  (DONE, commit pending)

**Executed** by `scripts/phase15c_resolution.py` (not `phase15_ablation.py`).
Re-preprocessed all three test sets from source at 384 px
(`data_processed_res384/`, `data_splits/*_manifest_res384.csv`), re-ran EffNet-B3
and RETFound-full at 224 (recomputed, asserted == `outputs/phase6/` <5e-4) and
384. RETFound built with `build_retfound_backbone(..., img_size=384)` (new kwarg)
+ `_interpolate_pos_embed` -> pos_embed (1,197,1024)->(1,577,1024), all blocks
load, forward on (1,3,384,384)->(1,5) OK. **Result: 384 does NOT recover external
QWK.** RETFound Messidor-2 0.368->0.393 (still << EffNet 0.499/0.425 at any res),
IDRiD 0.720->0.683; internal ~unchanged 0.869->0.872. EffNet loses internal QWK
at 384 (0.876->0.824). Delta-drop-vs-224 small and inconsistent-sign for both
(+0.022/-0.024 EffNet, -0.022/+0.040 RETFound on Messidor-2/IDRiD). Outputs in
`outputs/phase15c/`. Folded into Results sec:results-ablation + Limitations +
repro table + supplement list. Original plan below (superseded by the actual
script, which recomputes 224 rather than trusting the cache):

Real resolution needs re-preprocessing from **originals** at 384 (the committed
PNGs are 224 -> upscaling adds no information). Steps:

1. New `scripts/phase15_preprocess_res.py` -- reuse
   `preprocessing.preprocess_file(src, dst, size=384)` over the APTOS-test +
   Messidor-2 + IDRiD source images (paths are in the committed manifests'
   `source_path` column), writing `data_processed_res384/<dataset>/` +
   `data_splits/<dataset>_manifest_res384.csv` (same rows, new `processed_path`).
2. **EfficientNet-B3 at 384**: trivial -- conv net, any input size. Build the
   model, `load_state_dict` the committed `efficientnet_b3_best.pt`, run
   inference with `dataset.build_dataloader(..., size=384)`.
3. **RETFound (ViT-L/16) at 384**: needs positional-embedding interpolation.
   - `timm.create_model("vit_large_patch16_224", pretrained=False,
     num_classes=5, img_size=384, global_pool="avg", drop_path_rate=0.0)`
   - load `full_best.pt`'s `model_state` with `strict=False`; before loading,
     interpolate `pos_embed` from the 14x14 grid to 24x24 -- reuse
     `retfound._interpolate_pos_embed(model, state)` which already does exactly
     this bicubic resize (it keys off the shape mismatch).
   - sanity: assert the transformer-block tensors all load (mirror
     `retfound.load_retfound_weights`'s `block_missing` check); assert a forward
     pass on a (1,3,384,384) tensor returns (1,5).
   Caveat for the paper: the model was *trained* at 224, so 384 inference is
   off-distribution for the pos-embed; this tests sensitivity, not a fair
   384-trained comparison.

---

## Runner

`scripts/phase15_ablation.py`:
- args: `--variants A B C`, `--models baseline_efficientnet_b3 retfound_full`,
  `--device cuda`, `--out-dir outputs/phase15`.
- for each (variant, model, dataset): build model, load committed checkpoint,
  build a dataloader whose transform includes the variant's op, run
  `predict.infer`, `predict.save_predictions` to
  `outputs/phase15/<model>__<dataset>__<variant>`.
- assemble `ablation_master.csv` (model,dataset,variant,n,accuracy,qwk,macro_f1,
  referable_auc) and, joining APTOS-internal to each external row, the
  drop and delta-drop-vs-baseline-variant.
- `ablation_summary.md`: one table per variant + a "which variant most reduces
  RETFound's Messidor-2 excess drop" verdict line.

## Manuscript integration (after it runs)

- New Results subsection "Controlled degradation ablation" (after Calibration)
  with one small table (variant x model x [Messidor-2 QWK, delta-drop]).
- Discussion "Why the foundation model degrades" -- replace "We did not run a
  controlled mechanistic ablation" with the finding.
- Limitations "Highest-value next experiments" item (1) -- mark done, keep the
  training-set-size / multi-dataset sweep as the remaining part.
- `references.bib`: add Reinhard et al. 2001 if Variant B uses LAB transfer.

## Rough cost

Variant A + B: ~4 inference passes each over ~2900 external images x 2 models =
minutes on the RTX 5060. Variant C: + a one-off 384 re-preprocess (~2900 imgs,
~5-10 min) then same inference. Total well under an hour of GPU once the code is
written and debugged.
