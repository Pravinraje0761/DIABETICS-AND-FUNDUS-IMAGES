"""
Phase 8 - Monte-Carlo dropout inference (T stochastic forward passes).

Re-runs every trained checkpoint over the internal APTOS test set and the two
external datasets with dropout layers - and, for the RETFound ViT, stochastic
-depth (DropPath) layers - held ACTIVE at inference. The predictive distribution
is the mean softmax over T passes; predictive entropy of that mean is the
uncertainty score. Output schema matches Phase 6
(id_code, y_true, y_pred, prob_0..4, entropy) so scripts/phase8_uncertainty.py
can consume it directly:

  .venv/Scripts/python.exe scripts/phase8_uncertainty.py --preds-dir outputs/phase8/mc --tag mc

Run (needs a free GPU; ~Tx slower than a single pass):
  .venv/Scripts/python.exe scripts/phase8_mc_dropout.py --passes 30
  .venv/Scripts/python.exe scripts/phase8_mc_dropout.py --passes 30 --models retfound_full --device cuda

Outputs: outputs/phase8/mc/<model>__<dataset>_preds.csv  (+ _metrics.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import NUM_CLASSES, build_dataloader  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from predict import load_checkpoint, pick_device  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase8" / "mc"

CKPTS = {
    "baseline_efficientnet_b3": PROJECT_DIR / "outputs/phase4_baseline/efficientnet_b3_best.pt",
    "baseline_resnet50": PROJECT_DIR / "outputs/phase4_baseline/resnet50_best.pt",
    "retfound_linprobe": PROJECT_DIR / "outputs/phase5_retfound/linprobe_best.pt",
    "retfound_full": PROJECT_DIR / "outputs/phase5_retfound/full_best.pt",
}
DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}
STOCHASTIC_NAMES = ("Dropout", "DropPath", "StochasticDepth", "DropBlock")


def enable_stochastic(model: torch.nn.Module, min_p: float) -> int:
    """Put every dropout / stochastic-depth module into train() mode so it stays
    active under torch.no_grad(). Returns how many were switched on. If a ViT was
    trained with drop_path but no nn.Dropout, bump any zero prob to min_p so
    there is something to sample."""
    n = 0
    for m in model.modules():
        if m.__class__.__name__.startswith(STOCHASTIC_NAMES):
            m.train()
            n += 1
            for attr in ("p", "drop_prob"):
                if hasattr(m, attr) and isinstance(getattr(m, attr), float):
                    if getattr(m, attr) == 0.0 and min_p > 0:
                        setattr(m, attr, min_p)
    return n


@torch.no_grad()
def mc_infer(model, loader, device, passes: int):
    ids, ys, probs = [], [], []
    for images, labels, id_codes in loader:
        images = images.to(device, non_blocking=True)
        acc = torch.zeros(images.size(0), NUM_CLASSES, device=device)
        for _ in range(passes):
            acc += F.softmax(model(images), dim=1)
        probs.append((acc / passes).cpu().numpy())
        ids.extend(id_codes)
        ys.extend(int(v) for v in labels)
    prob = np.concatenate(probs, 0)
    y_true = np.asarray(ys, int)
    y_pred = prob.argmax(1)
    entropy = -(prob * np.log(prob + 1e-12)).sum(1)
    return ids, y_true, y_pred, prob, entropy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(CKPTS))
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--min-dropout-p", type=float, default=0.0,
                    help="floor applied to any zero-valued dropout/drop-path prob. "
                         "Default 0: MC only over layers that were STOCHASTIC "
                         "DURING TRAINING (nn.Dropout in the CNN heads; drop-path "
                         "0.2 in the RETFound full fine-tune). Forcing noise into "
                         "a frozen backbone that never saw it (linear probe, "
                         "drop-path 0) destroys its features.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"Phase 8 MC-dropout: T={args.passes}  device={device}  models={args.models}")

    for mkey in args.models:
        ckpt = CKPTS[mkey]
        if not ckpt.is_file():
            print(f"  skip {mkey}: checkpoint missing")
            continue
        model, _ = load_checkpoint(ckpt, device)
        # Only layers that were stochastic DURING TRAINING may be re-activated.
        # The linear probe's frozen backbone was trained with drop_path=0, but
        # load_checkpoint() rebuilds the ViT with the builder default (0.2); a
        # bug found in review (2026-10-21) meant its "MC" output injected
        # untrained stochastic depth. Respect the checkpoint's own drop_path.
        _hp = torch.load(ckpt, map_location="cpu", weights_only=False).get("hyperparams", {}) \
            if mkey.startswith("retfound") else {}
        if mkey.startswith("retfound") and float(_hp.get("drop_path", 0.2) or 0.0) == 0.0:
            n_on = 0
            print(f"  {mkey}: trained without stochastic layers -> deterministic (T passes identical)")
        else:
            n_on = enable_stochastic(model, args.min_dropout_p)
        print(f"  {mkey}: {n_on} stochastic modules active")
        for dkey, (manifest, split) in DATASETS.items():
            if not manifest.is_file():
                continue
            loader = build_dataloader(manifest, split=split,
                                      batch_size=args.batch_size, train=False,
                                      num_workers=args.num_workers, shuffle=False)
            ids, y_true, y_pred, prob, entropy = mc_infer(model, loader, device,
                                                          args.passes)
            df = pd.DataFrame({"id_code": ids, "y_true": y_true, "y_pred": y_pred})
            for c in range(prob.shape[1]):
                df[f"prob_{c}"] = prob[:, c]
            df["entropy"] = entropy
            pref = OUT_DIR / f"{mkey}__{dkey}"
            df.to_csv(pref.with_name(pref.name + "_preds.csv"), index=False)
            m = compute_metrics(y_true, y_pred, prob, num_classes=prob.shape[1])
            pref.with_name(pref.name + "_metrics.json").write_text(
                json.dumps(m, indent=2), encoding="utf-8")
            print(f"    {dkey:15s} n={len(df):4d}  acc={m['accuracy']:.3f}  "
                  f"QWK={m['quadratic_weighted_kappa']:.3f}  "
                  f"mean_entropy={float(entropy.mean()):.3f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nwrote {OUT_DIR}/  -> now run:\n  .venv/Scripts/python.exe "
          f"scripts/phase8_uncertainty.py --preds-dir outputs/phase8/mc --tag mc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
