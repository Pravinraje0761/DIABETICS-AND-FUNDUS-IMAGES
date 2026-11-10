"""
Phase 4 - baseline: fine-tune an ImageNet-pretrained CNN on the APTOS 2019
train split, early-stop on internal validation (quadratic-weighted kappa),
evaluate once on the internal test split.

Standard full fine-tuning: all layers unfrozen, small LR, AdamW, cosine decay,
weighted cross-entropy for the class imbalance, mixed precision on GPU.

APTOS ONLY. Messidor-2 / IDRiD are never loaded here (Phase 6).

Run (GPU):
  .venv/Scripts/python.exe scripts/train_baseline.py --arch resnet50
  .venv/Scripts/python.exe scripts/train_baseline.py --arch efficientnet_b3

Outputs (outputs/phase4_baseline/):
  <arch>_best.pt                     checkpoint (best val QWK)
  <arch>_history.csv                 per-epoch train/val curves
  <arch>_hyperparams.json           full config -> Methods table
  <arch>_internal_test_preds.csv    id_code, y_true, y_pred, prob_0..4, entropy
  <arch>_internal_test_metrics.json acc, QWK, macro-F1, per-class sens/spec, AUC
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
from models import ARCHS, build_model  # noqa: E402
from predict import infer, save_predictions  # noqa: E402
from preprocessing import RETFOUND_INPUT_SIZE  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
APTOS_MANIFEST = PROJECT_DIR / "data_splits" / "aptos_manifest.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "phase4_baseline"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    max_batches: int = 0):
    model.train()
    running, n = 0.0, 0
    for i, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
        n += images.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running, n = 0.0, 0
    ys, ps = [], []
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
    ap.add_argument("--arch", default="resnet50", choices=ARCHS)
    ap.add_argument("--manifest", default=None,
                    help="override the APTOS manifest CSV (default: "
                         "data_splits/aptos_manifest.csv). Use a variant manifest "
                         "for Phase 18 training-time ablations, e.g. "
                         "data_splits/aptos_manifest_res384_full.csv or "
                         "data_splits/aptos_manifest_clahe224.csv")
    ap.add_argument("--img-size", type=int, default=RETFOUND_INPUT_SIZE,
                    help="input resolution for train/val/test loaders (default "
                         "224). Pair with --manifest pointing at a manifest whose "
                         "processed_path images were preprocessed at this size.")
    ap.add_argument("--train-frac", type=float, default=None,
                    help="Phase 18 training-data-size sensitivity: stratified "
                         "subsample of the TRAIN split only (e.g. 0.25/0.5/0.75); "
                         "val/test always use the full split. Fixed seed = --seed.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7,
                    help="early stop after N epochs with no val-QWK improvement")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="",
                    help="suffix for all output filenames, e.g. --tag seed43 -> "
                         "efficientnet_b3_seed43_best.pt. Keeps multi-seed / "
                         "ablation runs from overwriting the primary checkpoint.")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"],
                    help="'ce' (primary) or 'focal' (Phase 17 loss-sensitivity "
                         "ablation). Focal keeps the same class-weight vector as "
                         "alpha unless --no-class-weights is also given.")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="debug: cap train batches per epoch (0 = all)")
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.arch + (f"_{args.tag}" if args.tag else "")

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest = Path(args.manifest) if args.manifest else APTOS_MANIFEST
    if not manifest.is_file():
        print(f"missing {manifest} - run phase3 scripts first")
        return 1

    train_loader = build_dataloader(manifest, "train", args.batch_size,
                                    train=True, num_workers=args.num_workers, shuffle=True,
                                    size=args.img_size, train_frac=args.train_frac,
                                    frac_seed=args.seed)
    val_loader = build_dataloader(manifest, "val", max(args.batch_size, 64),
                                  train=False, num_workers=args.num_workers, shuffle=False,
                                  size=args.img_size)
    train_ds: FundusDataset = train_loader.dataset  # type: ignore[assignment]

    model = build_model(args.arch, pretrained=True).to(device)

    class_weights = None if args.no_class_weights else train_ds.class_weights().to(device)
    criterion = build_criterion(args.loss, class_weights,
                                focal_gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    hyperparams = {
        "phase": 4,
        "arch": args.arch,
        "manifest": str(manifest.relative_to(PROJECT_DIR)) if manifest.is_relative_to(PROJECT_DIR) else str(manifest),
        "img_size": args.img_size,
        "train_frac": args.train_frac,
        "pretrained": "ImageNet (torchvision default weights)",
        "fine_tuning": "full (all layers unfrozen)",
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_schedule": "CosineAnnealingLR",
        "loss": (("CrossEntropy" if args.loss == "ce"
                  else f"Focal(gamma={args.focal_gamma})")
                 + ("" if args.no_class_weights else " (inverse-freq class weights)")),
        "class_weights": None if class_weights is None else [round(float(w), 4) for w in class_weights],
        "early_stopping": f"val quadratic-weighted kappa, patience={args.patience}",
        "augmentation": "resize224, H/V flip, +/-15deg rotation, brightness/contrast jitter 0.1",
        "normalisation": "ImageNet mean/std",
        "amp": use_amp,
        "seed": args.seed,
        "device": str(device),
        "train_n": len(train_ds),
        "val_n": len(val_loader.dataset),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{stem}_hyperparams.json").write_text(
        json.dumps(hyperparams, indent=2), encoding="utf-8"
    )
    print(json.dumps(hyperparams, indent=2))

    history: list[dict] = []
    best_qwk = -np.inf
    best_epoch = -1
    epochs_since_best = 0
    best_path = OUT_DIR / f"{stem}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, use_amp,
                                     max_batches=args.limit_train_batches)
        val_loss, val_m = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
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
        pd.DataFrame(history).to_csv(OUT_DIR / f"{stem}_history.csv", index=False)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_QWK={row['val_qwk']:.4f}  "
              f"val_acc={row['val_acc']:.4f}  lr={lr_now:.2e}  {dt:.0f}s")

        if val_m["quadratic_weighted_kappa"] > best_qwk + 1e-5:
            best_qwk = val_m["quadratic_weighted_kappa"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "arch": args.arch,
                    "num_classes": 5,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "hyperparams": hyperparams,
                },
                best_path,
            )
            print(f"  -> new best (val QWK {best_qwk:.4f}), saved {best_path.name}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early stopping at epoch {epoch} "
                      f"(no val-QWK improvement for {args.patience} epochs)")
                break

    print(f"\nbest epoch {best_epoch}, val QWK {best_qwk:.4f}")

    # -- internal test, once, with the best checkpoint ---------------------- #
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    test_loader = build_dataloader(manifest, "test", max(args.batch_size, 64),
                                   train=False, num_workers=args.num_workers, shuffle=False,
                                   size=args.img_size)
    res = infer(model, test_loader, device)
    preds_path, metrics_path = save_predictions(
        res, OUT_DIR / f"{stem}_internal_test"
    )
    test_m = json.loads(metrics_path.read_text())
    print("\n" + format_metrics(test_m, title=f"{args.arch} - APTOS internal TEST"))
    print(f"\nwrote {preds_path.name}, {metrics_path.name}")

    summary = {
        "arch": args.arch,
        "best_epoch": best_epoch,
        "val_qwk": best_qwk,
        "internal_test": {
            k: test_m[k] for k in
            ["n", "accuracy", "quadratic_weighted_kappa", "macro_f1",
             "balanced_accuracy", "auc_ovr_macro"] if k in test_m
        },
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
