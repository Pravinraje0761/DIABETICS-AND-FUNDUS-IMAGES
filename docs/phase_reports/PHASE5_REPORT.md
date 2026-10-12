# Phase 5 — RETFound Fine-Tuning: Completion Report

**Status:** complete, with one caveat (full fine-tune compute-truncated — see §4).
**Date:** 2026-09-08.
Methodology reference: `../analysis_plan.txt`, Phase 5.

Both variants were trained on the APTOS train split with the **same** data,
normalisation, metrics, early-stopping rule and internal-test protocol as the
Phase 4 baseline, so all internal-test numbers are directly comparable.
APTOS only — Messidor-2 / IDRiD untouched until Phase 6.

Reproduce:
```
.venv/Scripts/python.exe scripts/train_retfound.py --mode linprobe --weights weights/RETFound_mae_natureCFP.pth
.venv/Scripts/python.exe scripts/train_retfound.py --mode full     --weights weights/RETFound_mae_natureCFP.pth --batch-size 16
```

---

## 1. Setup

| Item | Value |
|---|---|
| Backbone | ViT-Large/16, RETFound MAE-pretrained on colour fundus |
| Weights | Hugging Face `YukunZhou/RETFound_mae_natureCFP` (gated), `RETFound_mae_natureCFP.pth` (full MAE checkpoint, `model`/`optimizer`/`scaler`/`args`) |
| Build | `timm==1.0.29` `vit_large_patch16_224`, global-avg-pool. MAE encoder loaded; pretrained `norm` remapped to `fc_norm`; fresh 5-way head. Load report: 294 tensors, only `head.{weight,bias}` missing, 0 unexpected — `outputs/phase5_retfound/<mode>_weight_load_report.json`. |
| Data | APTOS split (train 2562 / val 367 / test 733), ImageNet norm, mild aug (H/V flip, +/-15 deg rot, brightness/contrast 0.1) |
| Loss | CE, inverse-freq class weights, label smoothing 0.1 |
| Optimiser | AdamW, warmup 3 ep + cosine decay, AMP fp16 |
| Early stop | val quadratic-weighted kappa, patience 7 |
| Hardware | local RTX 5060 (8 GB), torch 2.11.0+cu128 |

**linprobe:** backbone frozen, head + `fc_norm` trainable. base LR 1e-3, 40 epochs max.
**full:** all layers unfrozen, layer-wise LR decay 0.65, drop-path 0.2. base LR 1e-4, batch 16, 30 epochs max.

---

## 2. Training

| Variant | Best epoch | Best val QWK | Stopped | Notes |
|---|--:|--:|---|---|
| linprobe | 25 | **0.916** | early-stop @ 32 (clean) | ~44 s/epoch |
| full | 6 | **0.935** | **manually stopped @ epoch 7** | see section 4 |

Curves: `outputs/phase5_retfound/{linprobe,full}_history.csv`.

- **linprobe** reached val QWK 0.916 with only a linear head on frozen features —
  within 1.5 pts of a *fully fine-tuned* EfficientNet-B3 (0.930). Strong evidence
  the RETFound representation is linearly informative for DR grading.
- **full** climbed fast (val QWK 0.59 -> 0.82 -> 0.87 -> 0.90 -> 0.91 -> **0.935**
  by epoch 6) and was still improving when stopped.

---

## 3. APTOS internal-test results (n = 733)

| Model | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |
|---|--:|--:|--:|--:|--:|
| EfficientNet-B3 (Phase 4 baseline) | **0.804** | **0.876** | **0.644** | **0.641** | **0.928** |
| ResNet-50 (Phase 4) | 0.789 | 0.875 | 0.616 | 0.630 | 0.924 |
| RETFound linprobe | 0.759 | 0.847 | 0.607 | 0.633 | 0.908 |
| RETFound full (epoch 6) | 0.769 | 0.869 | 0.616 | 0.629 | 0.915 |

### Per-class sensitivity (internal test)

| Grade (n) | Eff-B3 | ResNet-50 | RETFound-linprobe | RETFound-full |
|---|--:|--:|--:|--:|
| 0 No DR (361) | 0.961 | 0.958 | 0.953 | 0.953 |
| 1 Mild (74) | 0.527 | 0.662 | 0.608 | 0.527 |
| 2 Moderate (200) | 0.790 | 0.730 | 0.595 | 0.660 |
| 3 Severe (39) | 0.487 | 0.513 | 0.513 | 0.513 |
| 4 Proliferative (59) | 0.441 | 0.288 | 0.508 | 0.492 |

### Reading

- On the **same-distribution** APTOS test set, the ImageNet CNNs slightly
  **out-perform** both RETFound variants on the headline metrics (QWK 0.876 vs
  0.847 / 0.869). This is not surprising — RETFound's claimed advantage is
  *generalisation*, tested in Phase 6, not in-distribution accuracy.
- RETFound has **flatter per-class sensitivity** — noticeably better on the rare
  proliferative grade (0.49-0.51 vs the CNNs' 0.29-0.44) at some cost on the
  mid grades. Balanced accuracy is essentially tied across all four models
  (0.63-0.64).
- **val->test gap:** RETFound shows a larger optimism gap (linprobe val 0.916 ->
  test 0.847; full val 0.935 -> test 0.869) than EfficientNet-B3 (0.930 -> 0.876).
  The 367-image validation set is small; the test numbers are the reliable ones.

---

## 4. Caveat — full fine-tune was truncated

The full run was **manually stopped at epoch 7 of 30**. Cause: the project lives
under a OneDrive-synced folder, and the ~1.2 GB `full_best.pt` checkpoint was
re-saved (and re-uploaded by OneDrive) on **every** epoch, because every epoch
was a new best. Epoch time went 90 s -> 417 s -> 888 s as the sync backlog grew;
at that rate the run would have taken ~4 h.

Consequences:
- The `full` internal-test numbers use the **epoch-6 checkpoint** (val QWK 0.935,
  still rising). A completed run would plausibly reach a somewhat higher
  internal QWK and *might* narrow the Phase 6 external gap.
- The **linprobe** result is unaffected — it ran to a clean early-stop at
  epoch 32 — and it too shows no RETFound generalisation advantage (Phase 6),
  so the direction of the finding is consistent across both variants.

**Recommended before final write-up:** re-run `--mode full` to completion with
checkpoints written **outside** the OneDrive tree. Move `data_processed/` and
`.venv` out of OneDrive too, or pause OneDrive for the run. Expected wall-clock
on the RTX 5060 once the I/O bottleneck is gone: ~45-60 min.

---

## 5. Artefacts

| Path | Contents |
|---|---|
| `outputs/phase5_retfound/linprobe_best.pt` | linear-probe checkpoint (val QWK 0.916) |
| `outputs/phase5_retfound/full_best.pt` | full-FT checkpoint, **epoch 6** (val QWK 0.935) |
| `outputs/phase5_retfound/<mode>_hyperparams.json` | full config -> Methods table |
| `outputs/phase5_retfound/<mode>_history.csv` | per-epoch curves |
| `outputs/phase5_retfound/<mode>_weight_load_report.json` | which tensors loaded from the MAE checkpoint |
| `outputs/phase5_retfound/<mode>_internal_test_{preds.csv,metrics.json}` | internal-test predictions + metrics |
| `outputs/phase5_retfound/<mode>_summary.json` | headline numbers |
| `logs/phase5_retfound_train.log` | full training stdout |

`*_best.pt` gitignored; everything else committed. Weights (`weights/`) gitignored.

---

## 6. Carried into Phase 6

- RETFound models for external validation: **`linprobe_best.pt`** and
  **`full_best.pt`**, run unmodified via `scripts/predict.py`.
- Both are in the Phase 6 master table alongside the two CNN baselines
  (4 models x 3 datasets).
- The truncation caveat above must be repeated in the manuscript Limitations.
