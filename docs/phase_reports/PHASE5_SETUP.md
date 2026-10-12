# Phase 5 setup — obtaining the RETFound weights

The RETFound fine-tuning code (`scripts/train_retfound.py`, `scripts/retfound.py`)
is written and GPU-tested with a random-init ViT-L. It only needs the pretrained
**MAE ViT-Large** checkpoint, which is **gated** on Hugging Face and cannot be
downloaded automatically. Five minutes of manual steps:

## 1. Request access (one-time)

1. Sign in / create an account at https://huggingface.co
2. Open the model page and click **"Agree and access repository"**:
   - **https://huggingface.co/YukunZhou/RETFound_mae_natureCFP**
     (RETFound MAE, colour-fundus, the Nature 2023 weights — recommended for
     citation stability, this is what `--weights` expects by default)
   - Access is normally granted immediately / within a few hours.
3. Create a read token: https://huggingface.co/settings/tokens -> "New token" ->
   type **Read**.

## 2. Download the checkpoint into `weights/`

From `dr_foundation_study/`:

```powershell
.venv\Scripts\python.exe -m pip install -q "huggingface_hub[cli]"
.venv\Scripts\huggingface-cli login          # paste the read token
.venv\Scripts\huggingface-cli download YukunZhou/RETFound_mae_natureCFP `
    RETFound_mae_natureCFP.pth --local-dir weights
```

You should end up with `weights/RETFound_mae_natureCFP.pth` (~1.2 GB).
`weights/` is gitignored — the checkpoint is never committed (redistribution is
not permitted; cite the source instead).

> If the exact filename differs, run
> `.venv\Scripts\huggingface-cli download YukunZhou/RETFound_mae_natureCFP --local-dir weights`
> to pull the whole repo, then point `--weights` at the `.pth` it contains.

## 3. Run both fine-tuning variants (the Phase 5 ablation)

```powershell
.venv\Scripts\python.exe scripts\train_retfound.py --mode linprobe `
    --weights weights\RETFound_mae_natureCFP.pth
.venv\Scripts\python.exe scripts\train_retfound.py --mode full `
    --weights weights\RETFound_mae_natureCFP.pth
```

- **linprobe** — backbone frozen, head only. ~40 epochs, base LR 1e-3.
- **full** — everything unfrozen, layer-wise LR decay 0.65, drop-path 0.2,
  label smoothing 0.1. ~30 epochs, base LR 1e-4, early-stop on val QWK.

On the RTX 5060 (8 GB) both run at batch 16 with AMP. If you hit CUDA OOM on
`--mode full`, add `--batch-size 8`.

Outputs land in `outputs/phase5_retfound/` with the **same file layout and
metrics** as `outputs/phase4_baseline/`, so the internal-test tables are
directly comparable. Pick whichever variant wins (or report both — a valid
finding either way, per the methodology).

## 4. Provenance to record (for the Methods / Data Availability section)

| Field | Value |
|---|---|
| Model | RETFound (masked-autoencoder ViT-Large/16, colour-fundus pretraining) |
| Weights source | Hugging Face `YukunZhou/RETFound_mae_natureCFP` (gated), accessed `<fill in date>` |
| Citation | Zhou Y., et al. *A foundation model for generalizable disease detection from retinal images.* Nature, 2023;622:156-163. |
| Code | https://github.com/rmaphoh/RETFound_MAE |
| Licence | RETFound weights: research-use, non-commercial (per the model card). Do not redistribute — link only. |
| Fine-tuning framework | Backbone rebuilt with `timm==1.0.29` (`vit_large_patch16_224`, global-avg-pool); MAE encoder weights loaded, fresh linear head. Load report saved to `outputs/phase5_retfound/<mode>_weight_load_report.json`. |
