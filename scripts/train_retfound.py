"""
Phase 5 - fine-tune RETFound (ViT-Large/16, MAE-pretrained on retinal images)
on the APTOS 2019 train split. Two variants, the paper's strongest ablation:

  --mode linprobe   freeze the backbone, train only the linear head.
                    Fast; measures raw representation quality.
  --mode full       unfreeze everything, layer-wise LR decay.
                    Usually stronger, slower, more overfitting-prone on ~2.5k images.

Same data, normalisation, metrics, early-stopping rule and internal-test
protocol as the Phase 4 baseline, so the numbers are directly comparable.
APTOS ONLY - Messidor-2 / IDRiD are untouched until Phase 6.

Prereq: RETFound weights downloaded to weights/ (gated - see docs/phase_reports/PHASE5_SETUP.md).

Run:
  .venv/Scripts/python.exe scripts/train_retfound.py --mode linprobe --weights weights/RETFound_mae_natureCFP.pth
  .venv/Scripts/python.exe scripts/train_retfound.py --mode full     --weights weights/RETFound_mae_natureCFP.pth

Outputs (outputs/phase5_retfound/):
  <mode>_best.pt  <mode>_history.csv  <mode>_hyperparams.json
  <mode>_internal_test_preds.csv  <mode>_internal_test_metrics.json
  <mode>_summary.json  <mode>_weight_load_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import FundusDataset, build_dataloader  # noqa: E402
from losses import build_criterion  # noqa: E402
from metrics import compute_metrics, format_metrics  # noqa: E402
from predict import infer, save_predictions  # noqa: E402
from retfound import ARCH_NAME, build_retfound_backbone, load_retfound_weights  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
APTOS_MANIFEST = PROJECT_DIR / "data_splits" / "aptos_manifest.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "phase5_retfound"
N_BLOCKS = 24  # ViT-Large


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _layer_id(name: str) -> int:
    """0 = stem, 1..N_BLOCKS = transformer blocks, N_BLOCKS+1 = head/norm."""
    if name.startswith(("cls_token", "pos_embed", "patch_embed")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return N_BLOCKS + 1


def param_groups_lwd(model: nn.Module, base_lr: float, weight_decay: float,
                     decay: float = 0.65) -> list[dict]:
    """Layer-wise LR decay: deeper layers get a larger LR, stem the smallest."""
    groups: dict[tuple[int, bool], dict] = {}
    n_layers = N_BLOCKS + 1
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _layer_id(name)
        no_decay = p.ndim == 1 or name.endswith(".bias")
        key = (lid, no_decay)
        if key not in groups:
            scale = decay ** (n_layers - lid)
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "layer_id": lid,
                "lr_scale": scale,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def validate(model, loader, criterion, device):
    model.eval()
    running, n = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels_d = labels.to(device, non_blocking=True)
            logits = model(images)
            running += criterion(logits, labels_d).item() * images.size(0)
            n += images.size(0)
            ps.append(torch.softmax(logits, dim=1).float().cpu().numpy())
            ys.extend(int(v) for v in labels)
    prob = np.concatenate(ps, axis=0)
    m = compute_metrics(np.asarray(ys), prob.argmax(1), prob)
    return running / max(n, 1), m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="path to RETFound .pth")
    ap.add_argument("--mode", required=True, choices=["linprobe", "full"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="default 40 for linprobe, 30 for full")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-3 linprobe / 1e-4 full (base LR for LWD)")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=0.65)
    ap.add_argument("--drop-path", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--save-every", type=int, default=10_000,
                    help="flush the ~1.2 GB checkpoint to disk every N epochs "
                         "(and always once at the end). Default: only at the end. "
                         "Set 1 for the legacy per-epoch behaviour.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"],
                    help="'ce' (primary) or 'focal' (Phase 17 loss-sensitivity "
                         "ablation); focal keeps the class-weight alpha and the "
                         "--label-smoothing value.")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--ckpt-dir", default=None,
                    help="write <mode>_best.pt here instead of outputs/phase5_retfound/ "
                         "(use a path OUTSIDE OneDrive - the 1.2 GB ViT-L checkpoint "
                         "re-synced every epoch is what stalled the first full run)")
    ap.add_argument("--tag", default="",
                    help="extra suffix for output filenames, e.g. --tag seed43 -> "
                         "full_seed43_best.pt. Keeps multi-seed / ablation runs from "
                         "overwriting the primary linprobe_/full_ checkpoints.")
    ap.add_argument("--manifest", default=None,
                    help="override the APTOS manifest CSV (default: "
                         "data_splits/aptos_manifest.csv). Use a variant manifest "
                         "for Phase 18 training-time ablations, e.g. "
                         "data_splits/aptos_manifest_res384_full.csv or "
                         "data_splits/aptos_manifest_clahe224.csv")
    ap.add_argument("--img-size", type=int, default=224,
                    help="input resolution AND ViT pos_embed grid to build at "
                         "(default 224, RETFound's native size). A 224-trained "
                         "checkpoint's pos_embed is bicubically interpolated onto "
                         "any other size automatically (retfound.py); pair with "
                         "--manifest pointing at images preprocessed at this size.")
    ap.add_argument("--train-frac", type=float, default=None,
                    help="Phase 18 training-data-size sensitivity: stratified "
                         "subsample of the TRAIN split only (e.g. 0.25/0.5/0.75); "
                         "val/test always use the full split. Fixed seed = --seed.")
    args = ap.parse_args()

    epochs = args.epochs or (40 if args.mode == "linprobe" else 30)
    base_lr = args.lr or (1e-3 if args.mode == "linprobe" else 1e-4)
    drop_path = 0.0 if args.mode == "linprobe" else args.drop_path

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else OUT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = args.mode + (f"_{args.tag}" if args.tag else "")

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # -- model -------------------------------------------------------------- #
    img_size = args.img_size if args.img_size != 224 else None
    model = build_retfound_backbone(num_classes=5, drop_path=drop_path, img_size=img_size)
    report = load_retfound_weights(model, args.weights)
    (OUT_DIR / f"{tag}_weight_load_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"loaded RETFound weights: {report['n_loaded']} tensors "
          f"({len(report['missing_keys'])} missing, "
          f"{len(report['unexpected_keys'])} unexpected)")
    model.to(device)

    if args.mode == "linprobe":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("head.", "fc_norm."))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"linear probe - training only: {trainable}")

    # -- data -------------------------------------------------------------- #
    manifest = Path(args.manifest) if args.manifest else APTOS_MANIFEST
    train_loader = build_dataloader(manifest, "train", args.batch_size,
                                    train=True, num_workers=args.num_workers, shuffle=True,
                                    size=args.img_size, train_frac=args.train_frac,
                                    frac_seed=args.seed)
    val_loader = build_dataloader(manifest, "val", max(args.batch_size, 32),
                                  train=False, num_workers=args.num_workers, shuffle=False,
                                  size=args.img_size)
    train_ds: FundusDataset = train_loader.dataset  # type: ignore[assignment]

    class_weights = None if args.no_class_weights else train_ds.class_weights().to(device)
    criterion = build_criterion(args.loss, class_weights,
                                label_smoothing=args.label_smoothing,
                                focal_gamma=args.focal_gamma)

    # -- optimiser ------------------------------------------------------------ #
    if args.mode == "full":
        groups = param_groups_lwd(model, base_lr, args.weight_decay, args.layer_decay)
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    else:
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=base_lr,
                                      weight_decay=args.weight_decay)

    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    hyperparams = {
        "phase": 5,
        "arch": ARCH_NAME,
        "backbone": "ViT-Large/16, RETFound MAE-pretrained (retinal)",
        "mode": args.mode,
        "manifest": str(manifest.relative_to(PROJECT_DIR)) if manifest.is_relative_to(PROJECT_DIR) else str(manifest),
        "img_size": args.img_size,
        "train_frac": args.train_frac,
        "epochs_max": epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "base_lr": base_lr,
        "weight_decay": args.weight_decay,
        "layer_decay": args.layer_decay if args.mode == "full" else None,
        "drop_path": drop_path,
        "label_smoothing": args.label_smoothing,
        "warmup_epochs": args.warmup_epochs,
        "lr_schedule": "linear warmup + cosine decay",
        "loss": (("CrossEntropy" if args.loss == "ce"
                  else f"Focal(gamma={args.focal_gamma})")
                 + ("" if args.no_class_weights else " (inverse-freq class weights)")),
        "early_stopping": f"val quadratic-weighted kappa, patience={args.patience}",
        "augmentation": "resize224, H/V flip, +/-15deg rotation, brightness/contrast jitter 0.1",
        "normalisation": "ImageNet mean/std",
        "amp": use_amp,
        "seed": args.seed,
        "device": str(device),
        "train_n": len(train_ds),
        "val_n": len(val_loader.dataset),
        "weights_file": args.weights,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{tag}_hyperparams.json").write_text(
        json.dumps(hyperparams, indent=2), encoding="utf-8"
    )
    print(json.dumps(hyperparams, indent=2))

    # -- train loop ------------------------------------------------------------ #
    history: list[dict] = []
    best_qwk, best_epoch, since_best = -np.inf, -1, 0
    best_path = ckpt_dir / f"{tag}_best.pt"
    if ckpt_dir != OUT_DIR:
        print(f"checkpoints -> {best_path} (outside outputs/ to avoid OneDrive sync)")

    # The ViT-L checkpoint is ~1.2 GB. Re-writing it every epoch (each early epoch
    # is a new best) let Windows Defender's on-write scan backlog compound and
    # stalled the run from ~90 s/epoch to 400-900 s/epoch. Fix: keep the best
    # weights in RAM and flush to disk only every --save-every epochs and once at
    # the end. Set --save-every 1 to restore the old per-epoch behaviour.
    best_state: dict | None = None

    def _snapshot(ep: int, vm: dict) -> dict:
        return {
            "arch": ARCH_NAME, "num_classes": 5,
            "model_state": {k: v.detach().to("cpu", copy=True)
                            for k, v in model.state_dict().items()},
            "epoch": ep, "mode": args.mode, "val_metrics": vm,
            "hyperparams": hyperparams, "weight_load_report": report,
        }

    def _flush(reason: str) -> None:
        if best_state is None:
            return
        tmp = best_path.with_suffix(".pt.tmp")
        torch.save(best_state, tmp)
        tmp.replace(best_path)
        print(f"  -> checkpoint flushed to {best_path.name} "
              f"(best epoch {best_state['epoch']}, {reason})")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        train_loss = running / max(seen, 1)

        val_loss, val_m = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[-1]["lr"]
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_qwk": round(val_m["quadratic_weighted_kappa"], 5),
            "val_acc": round(val_m["accuracy"], 5),
            "val_macro_f1": round(val_m["macro_f1"], 5),
            "lr": lr_now,
            "seconds": round(dt, 1),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{tag}_history.csv", index=False)
        print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_QWK={row['val_qwk']:.4f}  "
              f"val_acc={row['val_acc']:.4f}  lr={lr_now:.2e}  {dt:.0f}s")

        if val_m["quadratic_weighted_kappa"] > best_qwk + 1e-5:
            best_qwk, best_epoch, since_best = val_m["quadratic_weighted_kappa"], epoch, 0
            best_state = _snapshot(epoch, val_m)
            print(f"  -> new best (val QWK {best_qwk:.4f}) held in memory")
            if epoch % args.save_every == 0:
                _flush("periodic")
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    _flush("final")
    print(f"\nbest epoch {best_epoch}, val QWK {best_qwk:.4f}")

    # -- internal test ------------------------------------------------------- #
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    test_loader = build_dataloader(manifest, "test", max(args.batch_size, 32),
                                   train=False, num_workers=args.num_workers, shuffle=False,
                                   size=args.img_size)
    res = infer(model, test_loader, device)
    preds_path, metrics_path = save_predictions(res, OUT_DIR / f"{tag}_internal_test")
    test_m = json.loads(metrics_path.read_text())
    print("\n" + format_metrics(test_m, title=f"RETFound-{args.mode} - APTOS internal TEST"))

    summary = {
        "arch": ARCH_NAME,
        "mode": args.mode,
        "best_epoch": best_epoch,
        "val_qwk": best_qwk,
        "internal_test": {
            k: test_m[k] for k in
            ["n", "accuracy", "quadratic_weighted_kappa", "macro_f1",
             "balanced_accuracy", "auc_ovr_macro"] if k in test_m
        },
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(f"\nwrote {preds_path.name}, {metrics_path.name}, {tag}_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
