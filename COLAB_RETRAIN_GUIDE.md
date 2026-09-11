# Colab / Kaggle retraining guide

Four analyses are prepared (data + code) but not yet run, because they need
sustained GPU time this laptop's RTX 5060 (8 GB) could not reliably provide for
RETFound full fine-tuning (`PHASE5_REPORT.md`, `AUTHOR_TODO.md` B7 — two local
attempts converged to val QWK ≈0.94 but were killed by thermal throttling /
host-memory exhaustion before a clean 30-epoch finish). All four reuse the
*existing* Phase 4/5/6/9/17 code paths unmodified — only new manifests, new CLI
flags on `train_baseline.py` / `train_retfound.py`, and three new scorer
scripts (Phase 18/19 is scoring-only and already run locally; 20 too) were
added this session.

1. **Training-time colour/resolution retraining** (vs. the existing
   inference-time-only ablation, `scripts/phase15_ablation.py` /
   `phase15c_resolution.py`) — does training under CLAHE / colour-transfer /
   384px change what the model learns, not just how an already-trained model
   reacts to the shift at test time.
2. **Training-set-size sensitivity** — does RETFound need more labelled data
   before any generalisation advantage appears.
3. **RETFound full-fine-tune loss sensitivity** — the existing Phase 17 check
   (plain CE / focal loss vs. the primary weighted CE) currently only covers
   the linear probe; extend it to full fine-tuning.
4. **A clean 30-epoch RETFound full fine-tune** (removes the `\pv{}`
   provisional-value markers throughout the paper) — not new work, but the
   same environment fixes everything else here, so do it in the same session.

## 0. Environment

```bash
# Colab: New notebook, Runtime -> Change runtime type -> GPU (T4/L4/A100)
!pip install -q torch==2.11.0 torchvision==0.26.0 timm==1.0.29 \
    scikit-learn==1.9.0 scipy==1.18.1 pandas==3.0.5 opencv-python-headless
```

Upload (Colab: left sidebar Files, or mount Drive and `%cd`) into a working
directory matching this repo's layout:

```
dr_foundation_study/
  scripts/                       # the whole folder
  data_splits/                   # committed manifests, small
  data_processed/                # committed-size PNGs (224px, all datasets) -- gitignored, zip+upload
  weights/RETFound_mae_natureCFP.pth   # gated HF download, ~1.2 GB -- see PHASE5_SETUP.md
  outputs/phase4_baseline/efficientnet_b3_best.pt   # for phase17/21 to alias frac100/weighted_ce
  outputs/phase5_retfound/{linprobe,full}_best.pt   # same
```

Also upload whichever of the three new variant trees you need (produced
locally by step 1 below and zipped) — or run step 1 directly in the Colab
runtime if you upload the raw source datasets instead (larger transfer, but
avoids a local CPU run):

```
data_processed_clahe224/aptos/          data_splits/aptos_manifest_clahe224.csv
data_processed_colourxfer224/aptos/     data_splits/aptos_manifest_colourxfer224.csv
data_processed_res384/aptos/            data_splits/aptos_manifest_res384_full.csv
```

## 1. Prepare the variant training data (CPU-only — can run locally first)

```bash
.venv/Scripts/python.exe scripts/phase18_retrain_variants_prep.py
# or one variant at a time:
.venv/Scripts/python.exe scripts/phase18_retrain_variants_prep.py --variants clahe224
```

Reprocesses APTOS (all 3 splits: train/val/test) from source images for CLAHE
and 384px, and applies colour-transfer to the existing 224px processed images
— see the script's module docstring for exactly what each variant does and
why. Zip the three new `data_processed_*` folders + 3 new manifest CSVs and
upload, or copy them onto the Colab/Kaggle instance directly.

## 2. Training-time colour/resolution retraining

```bash
# EfficientNet-B3 (fast, ~30 min/variant on a T4)
python scripts/train_baseline.py --arch efficientnet_b3 \
    --manifest data_splits/aptos_manifest_clahe224.csv --tag train_clahe
python scripts/train_baseline.py --arch efficientnet_b3 \
    --manifest data_splits/aptos_manifest_colourxfer224.csv --tag train_colourxfer
python scripts/train_baseline.py --arch efficientnet_b3 \
    --manifest data_splits/aptos_manifest_res384_full.csv --img-size 384 --tag train_res384

# RETFound full fine-tune (slow, ~2-4 h/variant on a T4 — the point of using Colab)
python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
    --manifest data_splits/aptos_manifest_clahe224.csv --tag train_clahe
python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
    --manifest data_splits/aptos_manifest_colourxfer224.csv --tag train_colourxfer
python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
    --manifest data_splits/aptos_manifest_res384_full.csv --img-size 384 --tag train_res384
```

`--img-size 384` builds the RETFound ViT-L backbone at `img_size=384` directly;
`retfound.py:load_retfound_weights` bicubically interpolates the pretrained
224-native (14×14) positional-embedding grid onto the 24×24 grid once at load
time, then the model trains normally at 384 from there (already smoke-tested
this session on CPU with `--img-size 256`, confirmed working end-to-end).

Then evaluate each new checkpoint under its OWN transform on the external sets
(fair comparison: trained-under-X, evaluated-under-X), reusing the existing
Phase 15 ablation script by pointing its `CKPTS` dict at the new checkpoint —
easiest is a one-off edit or a small wrapper; the eval logic (transform
construction, metrics, drop table) needs no changes:

```bash
# example for one new checkpoint; repeat per variant/model
python scripts/phase15_ablation.py --variants clahe \
    --models baseline_efficientnet_b3   # after retargeting CKPTS to *_train_clahe_best.pt
```

## 3. Training-set-size sensitivity

```bash
for frac in 25 50 75; do
  python scripts/train_baseline.py --arch efficientnet_b3 \
      --train-frac 0.$frac --tag frac$frac
  python scripts/train_retfound.py --mode linprobe --weights weights/RETFound_mae_natureCFP.pth \
      --train-frac 0.$frac --tag frac$frac
  python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
      --train-frac 0.$frac --tag frac$frac
done
python scripts/phase21_training_size_sensitivity.py --device cuda
```

`--train-frac` stratifies by grade and uses `--seed` (default 42) as the
subsample seed — val/test are always the full split, only the train split
shrinks. `phase21_training_size_sensitivity.py` auto-discovers
`{arch}_frac{25,50,75}_best.pt` / `{linprobe,full}_frac{25,50,75}_best.pt` and
scores them alongside the existing frac100 (primary) checkpoints.

## 4. RETFound full-fine-tune loss sensitivity

```bash
python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
    --tag noweight_loss --no-class-weights
python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
    --tag focal_loss --loss focal --focal-gamma 2
python scripts/phase17_loss_sensitivity.py --families retfound_full --device cuda
```

`phase17_loss_sensitivity.py` already has a `retfound_full` entry in its
`CKPTS` dict (`weighted_ce` aliased to the existing primary `full_best.pt`,
verified scoring correctly this session); it will pick up
`full_noweight_loss_best.pt` / `full_focal_loss_best.pt` automatically once
trained. **Do not** run this without `--families retfound_full` unless you
also have all the `efficientnet_b3`/`retfound_linprobe` loss-variant
checkpoints locally — a partial run overwrites the committed
`outputs/phase17/loss_{master,drop}.csv` / `loss_summary.md` with an
incomplete result (this happened once during this session's verification and
was reverted with `git checkout HEAD --`; safest is to run it in a scratch
copy of `outputs/phase17/` and merge manually, or run all three families in
one invocation so the write is complete).

## 5. A clean RETFound full fine-tune

```bash
python scripts/train_retfound.py --mode full --weights weights/RETFound_mae_natureCFP.pth \
    --batch-size 16 --ckpt-dir ./ckpts
```

~30 epochs, early-stopping on val QWK, patience 7; expect 2–5 h on a T4/L4.
Download `ckpts/full_best.pt`, copy over `outputs/phase5_retfound/full_best.pt`,
then re-run the refresh pipeline in `TODO.md` §1 ("After it completes") and
remove every `\pv{}` wrapper in `paper/main_standalone.tex`.

## Bringing results back

Download the new `outputs/phase{15,17,18,21}/...` files (or the whole
`outputs/` tree) back to this machine, `git add` them, and fold the numbers
into `paper/main_standalone.tex` (Limitations "Sensitivity analyses performed"
and "Highest-value next experiments" already name every one of these checks
and are written to be updated once each completes — search for "Colab" /
"prepared" / "not yet run" in that section).
