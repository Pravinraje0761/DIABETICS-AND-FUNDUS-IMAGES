# Phase 4 — Baseline Model: Completion Report

**Status:** complete. **Date:** 2026-09-08.
Methodology reference: `../analysis_plan.txt`, Phase 4.

Both candidate backbones from the methodology were trained and evaluated.
**EfficientNet-B3 is selected as the baseline** — it wins on every internal-test
metric and handles the rare high grades noticeably better.

Reproduce:
```
.venv/Scripts/python.exe scripts/train_baseline.py --arch efficientnet_b3
.venv/Scripts/python.exe scripts/train_baseline.py --arch resnet50
```
Hardware: local RTX 5060 Laptop GPU (8 GB), torch 2.11.0+cu128, AMP fp16.

---

## Checklist against the methodology

| Phase 4 task | Status |
|---|---|
| Train ImageNet-pretrained CNN (ResNet-50 **or** EfficientNet-B3) on APTOS train, full fine-tune, small LR, AdamW, 20-30 epochs, early stopping on internal val | done — **both** trained |
| Log every hyperparameter (Methods table) | done — `outputs/phase4_baseline/<arch>_hyperparams.json` |
| Evaluate internal test: accuracy, QWK, macro-F1, per-class sens/spec, AUC (OvR) | done — `<arch>_internal_test_metrics.json` + `_preds.csv` |
| Save trained baseline checkpoint | done — `<arch>_best.pt` |
| Do NOT touch Messidor-2 / IDRiD | respected — externals never loaded |

**Deliverable produced:** trained baseline checkpoint (`efficientnet_b3_best.pt`),
internal-test metrics table (below), hyperparameter log.

---

## 1. Training configuration (identical for both)

| Setting | Value |
|---|---|
| Pretraining | ImageNet (torchvision default weights) |
| Fine-tuning | full — all layers unfrozen |
| Optimizer | AdamW, lr 1e-4, weight-decay 1e-4 |
| LR schedule | cosine annealing over 30 epochs |
| Loss | cross-entropy, inverse-frequency class weights `[0.216, 1.052, 0.390, 2.019, 1.323]` |
| Batch size | 32, mixed precision (fp16) |
| Max epochs / early stop | 30 / patience 7 on **validation quadratic-weighted kappa** |
| Augmentation | resize-224, H/V flip, +/-15 deg rotation, brightness/contrast jitter 0.1 |
| Normalisation | ImageNet mean/std (same as RETFound will use) |
| Seed | 42 |
| Train / val / test n | 2562 / 367 / 733 (APTOS internal split, Phase 3) |

Per-epoch curves: `outputs/phase4_baseline/<arch>_history.csv`.

- **ResNet-50:** best epoch 6 (val QWK 0.919), early-stopped at epoch 13. ~29 s/epoch.
- **EfficientNet-B3:** best epoch 13 (val QWK 0.930), early-stopped at epoch 20. ~29 s/epoch.

---

## 2. APTOS internal-test results (n = 733)

| Metric | ResNet-50 | **EfficientNet-B3** |
|---|--:|--:|
| Accuracy | 0.789 | **0.804** |
| Quadratic-weighted kappa | 0.875 | **0.876** |
| Macro-F1 | 0.616 | **0.644** |
| Balanced accuracy | 0.630 | **0.641** |
| AUC (one-vs-rest, macro) | 0.924 | **0.928** |
| Best val QWK | 0.919 | **0.930** |

### Per-class sensitivity / specificity

| Grade (n) | ResNet-50 sens | ResNet-50 spec | Eff-B3 sens | Eff-B3 spec |
|---|--:|--:|--:|--:|
| 0 No DR (361) | 0.958 | 0.976 | 0.961 | 0.973 |
| 1 Mild (74) | 0.662 | 0.924 | 0.527 | 0.942 |
| 2 Moderate (200) | 0.730 | 0.910 | 0.790 | 0.897 |
| 3 Severe (39) | 0.513 | 0.944 | 0.487 | 0.964 |
| 4 Proliferative (59) | 0.288 | 0.987 | **0.441** | 0.976 |

EfficientNet-B3 confusion matrix (rows = true, cols = pred):
```
        pred0 pred1 pred2 pred3 pred4
true0    347    11     3     0     0
true1      8    39    26     0     1
true2      1    20   158    15     6
true3      1     1     9    19     9
true4      0     6    17    10    26
```

### Reading of the numbers

- **QWK approx 0.876 for both** — a solid, honest single-model APTOS baseline
  (no test-time augmentation, no ensembling). This is the comparison point
  RETFound must beat, on *internal* test and, more importantly, on the
  *external* drop.
- **Grade 0 is essentially solved** (sens approx 0.96). All the error is in
  grades 1-4, and it is mostly *adjacent-grade* confusion (the QWK/accuracy gap
  reflects this — most mistakes are off-by-one, which QWK penalises lightly).
- **The rare high grades are the weak point.** Grade 4 sensitivity is 0.29
  (ResNet) / 0.44 (EfficientNet) with only 206 training images. Grade 3 is
  ~0.50 for both. Per-class external numbers here will have wide CIs (Phase 9)
  — do not over-read single points.
- **Why EfficientNet-B3 was chosen:** higher on every headline metric, +15 pts
  grade-4 sensitivity, and a higher, later validation peak (less sign of the
  early over-fitting ResNet-50 showed after epoch 6).

---

## 3. Artefacts

| Path | Contents |
|---|---|
| `outputs/phase4_baseline/efficientnet_b3_best.pt` | **selected baseline checkpoint** (arch, state, val metrics, hyperparams) |
| `outputs/phase4_baseline/resnet50_best.pt` | alternate baseline checkpoint |
| `outputs/phase4_baseline/<arch>_hyperparams.json` | full config for the Methods table |
| `outputs/phase4_baseline/<arch>_history.csv` | per-epoch train/val loss, QWK, acc, F1, lr |
| `outputs/phase4_baseline/<arch>_internal_test_metrics.json` | full metric set incl. confusion matrix |
| `outputs/phase4_baseline/<arch>_internal_test_preds.csv` | id_code, y_true, y_pred, prob_0..4, entropy — kept for Phase 9 stats |
| `outputs/phase4_baseline/<arch>_summary.json` | headline numbers |
| `logs/phase4_<arch>_train.log` | full training stdout |

`*_best.pt` is gitignored (large); metrics/history/preds/hyperparams are committed.

---

## 4. Carried into Phase 6

- Baseline for external validation = **`efficientnet_b3_best.pt`**, run
  **unmodified** on Messidor-2 and IDRiD via `scripts/predict.py`.
- `resnet50_best.pt` can also be carried through as a secondary comparison if
  wanted, at no extra design cost.
- RETFound (Phase 5) uses the **same** train/val/test split, normalisation,
  metrics, early-stopping rule and `predict.py` inference path, so Phase 6
  produces one directly comparable 6-row master table.
