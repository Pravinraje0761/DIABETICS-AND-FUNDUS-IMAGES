"""
Phase 17 -- class-weighting / loss-function robustness.

The manuscript's Limitations lists "alternative class-weighting" as not-yet-run.
The primary models use inverse-frequency weighted cross-entropy (RETFound adds
label smoothing 0.1). This runner re-trains the two adaptable families with two
alternative losses and scores every resulting checkpoint UNMODIFIED on the
Phase-6 code path, to check that the internal->external QWK drop ranking and the
RETFound--CNN gap direction do not depend on the loss choice.

Variants per family (each a `train_*.py --tag <...>` run):
  weighted_ce   inverse-frequency weighted CE                     [= primary model]
  ce_noweight   plain cross-entropy, no class weights             (--no-class-weights)
  focal_g2      focal loss, gamma=2, inverse-frequency alpha      (--loss focal)

Families:
  efficientnet_b3     (the selected CNN baseline; full fine-tune)
  retfound_linprobe   (RETFound linear probe; full fine-tune of RETFound-full
                       thermally/memory-throttles on this laptop -- see
                       AUTHOR_TODO.md B7 -- so the full regime is out of scope
                       here and the linear probe stands in for it, as in the
                       Phase 16 seed sweep)

This script only SCORES. Train the variant checkpoints first:
  .venv/Scripts/python.exe scripts/train_baseline.py --arch efficientnet_b3 --tag noweight_loss --no-class-weights
  .venv/Scripts/python.exe scripts/train_baseline.py --arch efficientnet_b3 --tag focal_loss --loss focal --focal-gamma 2
  .venv/Scripts/python.exe scripts/train_retfound.py --mode linprobe --weights weights/RETFound_mae_natureCFP.pth --tag noweight_loss --no-class-weights
  .venv/Scripts/python.exe scripts/train_retfound.py --mode linprobe --weights weights/RETFound_mae_natureCFP.pth --tag focal_loss --loss focal --focal-gamma 2
then:
  .venv/Scripts/python.exe scripts/phase17_loss_sensitivity.py --device cuda

Outputs (outputs/phase17/):
  <family>__<loss>__<dataset>_preds.csv / _metrics.json    Phase-6 schema
  loss_master.csv        family x loss x dataset: QWK / acc / macro-F1 / referable-AUC
  loss_drop.csv          internal->external QWK drop per family x loss
  loss_summary.md        manuscript digest
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_dataloader  # noqa: E402
from metrics import compute_metrics, format_metrics  # noqa: E402
from predict import infer, load_checkpoint, pick_device, save_predictions  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase17"
P4 = PROJECT_DIR / "outputs" / "phase4_baseline"
P5 = PROJECT_DIR / "outputs" / "phase5_retfound"

DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}

# family -> {loss label -> checkpoint path}
CKPTS = {
    "efficientnet_b3": {
        "weighted_ce": P4 / "efficientnet_b3_best.pt",
        "ce_noweight": P4 / "efficientnet_b3_noweight_loss_best.pt",
        "focal_g2": P4 / "efficientnet_b3_focal_loss_best.pt",
    },
    "retfound_linprobe": {
        "weighted_ce": P5 / "linprobe_best.pt",
        "ce_noweight": P5 / "linprobe_noweight_loss_best.pt",
        "focal_g2": P5 / "linprobe_focal_loss_best.pt",
    },
}


def referable_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(y_true) >= 2).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, prob[:, 2:].sum(axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--families", nargs="+", default=None,
                    help="subset of: " + ", ".join(CKPTS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    families = args.families or list(CKPTS)

    rows: list[dict] = []
    missing: list[str] = []
    for fam in families:
        for loss, ckpt in CKPTS[fam].items():
            if not ckpt.is_file():
                missing.append(f"{fam}/{loss}: {ckpt.name}")
                continue
            model, _ = load_checkpoint(ckpt, device)
            for dkey, (manifest, split) in DATASETS.items():
                if not manifest.is_file():
                    print(f"  {dkey}: manifest missing, skip")
                    continue
                loader = build_dataloader(manifest, split=split,
                                          batch_size=args.batch_size, train=False,
                                          num_workers=args.num_workers, shuffle=False)
                res = infer(model, loader, device)
                prefix = OUT_DIR / f"{fam}__{loss}__{dkey}"
                save_predictions(res, prefix)
                m = compute_metrics(res["y_true"], res["y_pred"], res["prob"])
                rauc = referable_auc(res["y_true"], res["prob"])
                rows.append({
                    "family": fam, "loss": loss, "dataset": dkey, "n": m["n"],
                    "qwk": round(m["quadratic_weighted_kappa"], 4),
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "referable_auc": round(rauc, 4) if rauc == rauc else None,
                })
                print(format_metrics(m, title=f"{fam} / {loss} x {dkey}"))
                print()
            del model

    if missing:
        print("MISSING checkpoints (train them first, see the module docstring):")
        for mm in missing:
            print("  " + mm)
    if not rows:
        return 1

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "loss_master.csv", index=False)

    drop_rows: list[dict] = []
    for fam in master["family"].unique():
        for loss in master[master["family"] == fam]["loss"].unique():
            s = (master[(master.family == fam) & (master.loss == loss)]
                 .set_index("dataset")["qwk"])
            if "APTOS-internal" not in s.index:
                continue
            for dkey in ("Messidor-2", "IDRiD"):
                if dkey in s.index:
                    drop_rows.append({
                        "family": fam, "loss": loss, "external": dkey,
                        "qwk_internal": float(s["APTOS-internal"]),
                        "qwk_external": float(s[dkey]),
                        "qwk_drop": round(float(s["APTOS-internal"] - s[dkey]), 4),
                    })
    drop = pd.DataFrame(drop_rows)
    drop.to_csv(OUT_DIR / "loss_drop.csv", index=False)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = ["# Phase 17 -- class-weighting / loss-function robustness", "",
         f"_Generated {ts}. Variant checkpoints scored unmodified on the Phase-6 "
         "code path. `weighted_ce` = the primary model._", "",
         "## QWK by family x loss x dataset", "",
         "| Family | Loss | Internal | Messidor-2 | IDRiD |",
         "|---|---|--:|--:|--:|"]
    for fam in master["family"].unique():
        for loss in ("weighted_ce", "ce_noweight", "focal_g2"):
            s = (master[(master.family == fam) & (master.loss == loss)]
                 .set_index("dataset")["qwk"])
            if s.empty:
                continue
            L.append(f"| {fam} | {loss} | "
                     f"{s.get('APTOS-internal', float('nan')):.3f} | "
                     f"{s.get('Messidor-2', float('nan')):.3f} | "
                     f"{s.get('IDRiD', float('nan')):.3f} |")
    L += ["", "## Internal->external QWK drop by family x loss", "",
          "| Family | Loss | Messidor-2 drop | IDRiD drop |",
          "|---|---|--:|--:|"]
    for fam in drop["family"].unique():
        for loss in ("weighted_ce", "ce_noweight", "focal_g2"):
            d = drop[(drop.family == fam) & (drop.loss == loss)].set_index("external")["qwk_drop"]
            if d.empty:
                continue
            L.append(f"| {fam} | {loss} | "
                     f"{d.get('Messidor-2', float('nan')):.3f} | "
                     f"{d.get('IDRiD', float('nan')):.3f} |")
    L += ["", "Per-image predictions: `outputs/phase17/<family>__<loss>__<dataset>_preds.csv`.",
          ""]
    (OUT_DIR / "loss_summary.md").write_text("\n".join(L), encoding="utf-8")

    print("=" * 70)
    print(master.pivot_table(index=["family", "loss"], columns="dataset",
                             values="qwk").to_string())
    print()
    print(drop.pivot_table(index=["family", "loss"], columns="external",
                           values="qwk_drop").to_string())
    print(f"\nwrote {OUT_DIR}/loss_master.csv, loss_drop.csv, loss_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
