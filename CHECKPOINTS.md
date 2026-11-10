# Model versions, checkpoints, and external-evaluation lock

Every model that produced a number in the manuscript, the checkpoint it was
scored from, and that file's SHA-256. Checkpoints are not in git (size, and the
RETFound licence); the hashes let anyone holding the files verify they are the
ones scored. **Frozen** = the checkpoint used for the primary comparison.

## 1. Primary comparison (Tables 2–5, all figures)

| Model | Checkpoint | Selected by | SHA-256 | Status |
|---|---|---|---|---|
| EfficientNet-B3 (primary baseline) | `outputs/phase4_baseline/efficientnet_b3_best.pt` | best APTOS-val QWK, patience 7 | `a595133256a678cbf5ef6daca64842f989dc761c61042919bfe3d3b9926c480b` | **frozen** |
| ResNet-50 (secondary baseline) | `outputs/phase4_baseline/resnet50_best.pt` | best APTOS-val QWK, patience 7 | `225610ffa51b778c6bfd3abd302aaad9cd68c7157053e73431d9aa009a696520` | **frozen** |
| RETFound linear probe | `outputs/phase5_retfound/linprobe_best.pt` | best APTOS-val QWK, patience 7 | `e7795a1bd2db209573e3a868cb927b30428db265a98f64733b5e1ec081fe31c9` | **frozen** |
| RETFound full fine-tune (completed run, v4) | `outputs/phase5_retfound/full_best.pt` (copied from `C:\dr_ckpts_full_v4\full_best.pt`) | best APTOS-val QWK over the completed protocol (≤30 epochs, patience 7): epoch 21 of 28, val QWK 0.939 | `b1d4a7b23b79413c2d313c4e742c79f7cf1dc6bfc3a388ef541868b511fd5933` | **frozen** |
| RETFound full fine-tune (superseded, provisional) | `outputs/phase5_retfound/full_best_provisional_epoch6.pt` | best APTOS-val QWK at epoch 6 of an interrupted run | `4344dd30356e7ae56143c0f6db670bc080db9b4d0baeb8fc866426334990322c` | superseded; its outputs archived in `outputs/_archive_provisional_epoch6/` |
| Pretrained RETFound encoder | `weights/RETFound_mae_natureCFP.pth` (Hugging Face, gated) | — | `e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485` | input weights |

## 2. Sensitivity-analysis checkpoints (Limitations only; never used to choose the primary models)

| Analysis | Checkpoint | SHA-256 |
|---|---|---|
| Seed 43 / 44 / 45, EfficientNet-B3 | `outputs/phase4_baseline/efficientnet_b3_seed4{3,4,5}_best.pt` | `87d0f899…61df95` / `fe2e7133…a1afa86a` / `eecb0e1a…0216a` (full hashes in `outputs/checkpoint_sha256.txt`) |
| Seed 43 / 44 / 45, RETFound linear probe | `C:\dr_ckpts_seeds\linprobe_seed4{3,4,5}_best.pt` (outside the repo) | not hashed (outside the connected folder); predictions in `outputs/phase16/` |
| Loss sensitivity, EfficientNet-B3 (CE no weights / focal) | `efficientnet_b3_{noweight,focal}_loss_best.pt` | `17fbc8d7…393146` / `b159feb2…5e45f` |
| Loss sensitivity, RETFound linear probe | `linprobe_{noweight,focal}_loss_best.pt` | `488f0e69…9444` / `7e974138…1beb` |
| Head-depth sweep | frozen features of the two primary checkpoints above; 210 heads re-fitted in memory | `outputs/phase22/manifest.json` |

## 3. Evidence that Messidor-2 and IDRiD did not influence model or hyperparameter selection

1. **Training code and every hyperparameter default were committed before any
   external image was scored, and never changed afterwards.**
   - `train_baseline.py` committed in `5a0fe2c` (2026-09-08 12:28 IST) and
     `train_retfound.py` in `894f389` (12:44 IST).
   - The first external scoring is commit `a7f04ac` (13:16 IST, "preliminary
     baseline external results"; RETFound was still training).
   - `git diff 894f389 8156b18 -- scripts/train_retfound.py` and
     `git diff 5a0fe2c 5638e8f -- scripts/train_baseline.py` are empty. The only
     model-code change in that window is `4e6330e` (13:12 IST, *before* external
     scoring): a weight-loading fix that maps the pretrained final LayerNorm to
     the pooling norm.
   - Every later edit to the two trainers added infrastructure flags only
     (`--save-every`, `--ckpt-dir`, `--tag`, `--manifest`, `--img-size`,
     `--train-frac`, `--loss`, `--resume`) whose defaults reproduce the original
     behaviour. All training defaults (epochs, LR, weight decay, layer decay,
     drop-path, label smoothing, warm-up, patience, batch size, seed) are
     byte-identical between the original commits and the current files.
2. **Checkpoint selection used APTOS validation QWK only.** Early stopping and
   best-epoch selection read `val_qwk` from the APTOS validation split; no
   external metric is computed inside either trainer.
3. **No threshold, temperature or calibration fitted on external data.** The
   models are scored by argmax; temperature scaling (Phase 13) is fitted on
   APTOS-val or an APTOS-train subset and only *applied* externally.
4. **Post-hoc external analyses did not change the primary models.** The
   inference-time ablations (Phase 15/15c), the seed and loss sweeps
   (Phase 16/17) and the head-depth sweep (Phase 22) scored external sets for
   additional configurations *after* the primary results existed. None of them
   replaced a primary checkpoint; the Phase 22 decision was to keep the
   pre-specified linear head.
5. **The completed RETFound full fine-tune (v4) re-ran the identical,
   pre-specified protocol** (same code defaults, seed 42, same data) to
   convergence. Its only addition is resume-from-last-epoch, used solely to
   survive hardware interruptions; checkpoint choice is again by APTOS-val QWK.
   The interrupted epoch-6 checkpoint it replaces is retained, hashed, and its
   outputs archived for comparison.
6. **Disclosed contact with external images before scoring:** the Phase 2 data
   audit (file counts, label parsing, grade distribution) and the Phase 3
   preprocessing run + visual label–image alignment audit, which applied the
   already-fixed preprocessing to all three datasets.

## 4. Re-verify

```
sha256sum outputs/phase4_baseline/*.pt outputs/phase5_retfound/*.pt weights/*.pth
git log --format='%h %ad %s' --date=iso
git diff 894f389 8156b18 -- scripts/train_retfound.py
```
