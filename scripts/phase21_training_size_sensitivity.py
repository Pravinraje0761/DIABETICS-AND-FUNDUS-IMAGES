"""
Phase 21 -- training-data-size sensitivity ("does RETFound need more labelled
data before its advantage appears?").

Scores checkpoints trained on a stratified subsample of the APTOS train split
(scripts/train_baseline.py / train_retfound.py --train-frac 0.25/0.5/0.75,
--tag frac25/frac50/frac75) alongside the existing frac=1.0 primary model,
unmodified, on the identical Phase-6 code path. Training itself is
Colab/Kaggle-bound on this laptop for the RETFound-full family -- see
COLAB_RETRAIN_GUIDE.md for exact commands.

Checkpoint naming (train_*.py --tag fracNN):
  outputs/phase4_baseline/efficientnet_b3_frac25_best.pt
  outputs/phase5_retfound/{linprobe,full}_frac25_best.pt
  ... frac50, frac75; frac100 = the existing primary checkpoint (no retrain).

Outputs (outputs/phase21/):
  <family>__frac<NN>__<dataset>_preds.csv / _metrics.json   (Phase-6 schema)
  size_master.csv   family x train_frac x dataset: QWK / acc / macro-F1 / referable-AUC
  size_drop.csv     internal->external QWK drop per family x train_frac
  size_summary.md

Run (after training the variant checkpoints -- see module docstring commands
in scripts/train_baseline.py --help / train_retfound.py --help):
  .venv/Scripts/python.exe scripts/phase21_training_size_sensitivity.py --device cuda
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
OUT_DIR = PROJECT_DIR / "outputs" / "phase21"
P4 = PROJECT_DIR / "outputs" / "phase4_baseline"
P5 = PROJECT_DIR / "outputs" / "phase5_retfound"

DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}
FRACS = (25, 50, 75, 100)  # percent; 100 = primary checkpoint, no retrain needed

# family -> {frac (percent) -> checkpoint path}
CKPTS = {
    "efficientnet_b3": {
        100: P4 / "efficientnet_b3_best.pt",
        75: P4 / "efficientnet_b3_frac75_best.pt",
        50: P4 / "efficientnet_b3_frac50_best.pt",
        25: P4 / "efficientnet_b3_frac25_best.pt",
    },
    "retfound_linprobe": {
        100: P5 / "linprobe_best.pt",
        75: P5 / "linprobe_frac75_best.pt",
        50: P5 / "linprobe_frac50_best.pt",
        25: P5 / "linprobe_frac25_best.pt",
    },
    "retfound_full": {
        100: P5 / "full_best.pt",
        75: P5 / "full_frac75_best.pt",
        50: P5 / "full_frac50_best.pt",
        25: P5 / "full_frac25_best.pt",
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
        for frac, ckpt in CKPTS[fam].items():
            if not ckpt.is_file():
                missing.append(f"{fam}/frac{frac}: {ckpt.name}")
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
                prefix = OUT_DIR / f"{fam}__frac{frac}__{dkey}"
                save_predictions(res, prefix)
                m = compute_metrics(res["y_true"], res["y_pred"], res["prob"])
                rauc = referable_auc(res["y_true"], res["prob"])
                rows.append({
                    "family": fam, "train_frac_pct": frac, "dataset": dkey, "n": m["n"],
                    "qwk": round(m["quadratic_weighted_kappa"], 4),
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "referable_auc": round(rauc, 4) if rauc == rauc else None,
                })
                print(format_metrics(m, title=f"{fam} frac{frac}% x {dkey}"))
                print()
            del model

    if missing:
        print("MISSING checkpoints (train them first -- see module docstring / "
             "COLAB_RETRAIN_GUIDE.md):")
        for mm in missing:
            print("  " + mm)
    if not rows:
        return 1

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "size_master.csv", index=False)

    drop_rows: list[dict] = []
    for fam in master["family"].unique():
        for frac in master[master["family"] == fam]["train_frac_pct"].unique():
            s = (master[(master.family == fam) & (master.train_frac_pct == frac)]
                .set_index("dataset")["qwk"])
            if "APTOS-internal" not in s.index:
                continue
            for dkey in ("Messidor-2", "IDRiD"):
                if dkey in s.index:
                    drop_rows.append({
                        "family": fam, "train_frac_pct": frac, "external": dkey,
                        "qwk_internal": float(s["APTOS-internal"]),
                        "qwk_external": float(s[dkey]),
                        "qwk_drop": round(float(s["APTOS-internal"] - s[dkey]), 4),
                    })
    drop = pd.DataFrame(drop_rows)
    drop.to_csv(OUT_DIR / "size_drop.csv", index=False)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = ["# Phase 21 -- training-data-size sensitivity", "",
        f"_Generated {ts}. Stratified subsamples of the APTOS train split "
        f"(fixed seed = --seed at train time); val/test always use the full "
        f"split. frac100 = the primary model (no retrain)._", "",
        "## QWK by family x train-fraction x dataset", "",
        "| Family | Train % | Internal | Messidor-2 | IDRiD |",
        "|---|--:|--:|--:|--:|"]
    for fam in master["family"].unique():
        for frac in FRACS:
            s = (master[(master.family == fam) & (master.train_frac_pct == frac)]
                .set_index("dataset")["qwk"])
            if s.empty:
                continue
            L.append(f"| {fam} | {frac} | "
                    f"{s.get('APTOS-internal', float('nan')):.3f} | "
                    f"{s.get('Messidor-2', float('nan')):.3f} | "
                    f"{s.get('IDRiD', float('nan')):.3f} |")
    L += ["", "## Internal->external QWK drop by family x train-fraction", "",
         "If RETFound needs more labelled data before its advantage over the "
         "CNN baseline appears, its drop should shrink faster than the "
         "baseline's as train_frac_pct increases.", "",
         "| Family | Train % | Messidor-2 drop | IDRiD drop |",
         "|---|--:|--:|--:|"]
    for fam in drop["family"].unique():
        for frac in FRACS:
            d = (drop[(drop.family == fam) & (drop.train_frac_pct == frac)]
                .set_index("external")["qwk_drop"])
            if d.empty:
                continue
            L.append(f"| {fam} | {frac} | "
                    f"{d.get('Messidor-2', float('nan')):.3f} | "
                    f"{d.get('IDRiD', float('nan')):.3f} |")
    L += ["", "Per-image predictions: `outputs/phase21/<family>__frac<NN>__<dataset>_preds.csv`.", ""]
    (OUT_DIR / "size_summary.md").write_text("\n".join(L), encoding="utf-8")

    print("=" * 70)
    print(master.pivot_table(index=["family", "train_frac_pct"], columns="dataset",
                             values="qwk").to_string())
    print()
    print(drop.pivot_table(index=["family", "train_frac_pct"], columns="external",
                           values="qwk_drop").to_string())
    print(f"\nwrote {OUT_DIR}/size_master.csv, size_drop.csv, size_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
