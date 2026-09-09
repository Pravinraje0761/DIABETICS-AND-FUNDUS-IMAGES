"""
Phase 16 -- training-seed sensitivity for the selected baseline.

The manuscript's Limitations lists "multiple training seeds" as not-yet-run. This
runner closes that gap for EfficientNet-B3 (the selected CNN baseline) by taking
the three extra full-fine-tune runs (seeds 43/44/45; the primary model is seed
42) UNMODIFIED and scoring them on the identical Phase-6 code path:

  - APTOS internal test  (data_splits/aptos_manifest.csv : test)
  - Messidor-2           (external, all gradable)
  - IDRiD                (external)

For each seed x dataset it writes Phase-6-schema predictions + metrics, then
aggregates mean / SD / min / max of QWK (and referable-DR AUC) across the four
seeds, plus the internal->external QWK drop per seed. RETFound linear-probe seeds
are picked up automatically if their checkpoints exist under
outputs/phase5_retfound/linprobe_seed*_best.pt (or $DR_CKPTS_SEEDS).

Run: .venv/Scripts/python.exe scripts/phase16_seed_sweep.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
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
OUT_DIR = PROJECT_DIR / "outputs" / "phase16"
P4 = PROJECT_DIR / "outputs" / "phase4_baseline"
P5 = PROJECT_DIR / "outputs" / "phase5_retfound"
SEED_CKPT_DIRS = [P4, P5, Path(os.environ.get("DR_CKPTS_SEEDS", r"C:\dr_ckpts_seeds"))]

DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}
PRIMARY_SEED = 42


def _find(*names: str) -> Path | None:
    for d in SEED_CKPT_DIRS:
        for n in names:
            p = d / n
            if p.is_file():
                return p
    return None


def discover_models() -> dict[str, dict[int, Path]]:
    """{family: {seed: checkpoint_path}} for every seed checkpoint present."""
    models: dict[str, dict[int, Path]] = {"efficientnet_b3": {}, "retfound_linprobe": {}}

    primary_eff = _find("efficientnet_b3_best.pt")
    if primary_eff:
        models["efficientnet_b3"][PRIMARY_SEED] = primary_eff
    primary_lp = _find("linprobe_best.pt")
    if primary_lp:
        models["retfound_linprobe"][PRIMARY_SEED] = primary_lp

    for seed in range(40, 60):
        eff = _find(f"efficientnet_b3_seed{seed}_best.pt")
        if eff:
            models["efficientnet_b3"][seed] = eff
        lp = _find(f"linprobe_seed{seed}_best.pt", f"retfound_linprobe_seed{seed}_best.pt")
        if lp:
            models["retfound_linprobe"][seed] = lp

    return {k: v for k, v in models.items() if len(v) >= 2}


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
                    help="subset of efficientnet_b3 retfound_linprobe")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    models = discover_models()
    if args.families:
        models = {k: v for k, v in models.items() if k in args.families}
    if not models:
        print("no seed checkpoints found (need >=2 seeds for at least one family)")
        return 1
    for fam, seeds in models.items():
        print(f"{fam}: seeds {sorted(seeds)}")

    rows: list[dict] = []
    for fam, seed_ckpts in models.items():
        for seed, ckpt in sorted(seed_ckpts.items()):
            model, _ = load_checkpoint(ckpt, device)
            for dkey, (manifest, split) in DATASETS.items():
                if not manifest.is_file():
                    print(f"  {dkey}: manifest missing, skip")
                    continue
                loader = build_dataloader(manifest, split=split,
                                          batch_size=args.batch_size, train=False,
                                          num_workers=args.num_workers, shuffle=False)
                res = infer(model, loader, device)
                prefix = OUT_DIR / f"{fam}_seed{seed}__{dkey}"
                save_predictions(res, prefix)
                m = compute_metrics(res["y_true"], res["y_pred"], res["prob"])
                rauc = referable_auc(res["y_true"], res["prob"])
                rows.append({
                    "family": fam, "seed": seed, "dataset": dkey, "n": m["n"],
                    "qwk": round(m["quadratic_weighted_kappa"], 4),
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "referable_auc": round(rauc, 4) if rauc == rauc else None,
                })
                print(format_metrics(m, title=f"{fam} seed{seed} x {dkey}"))
                print()

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "seed_master.csv", index=False)

    # aggregate across seeds
    agg_rows: list[dict] = []
    drop_rows: list[dict] = []
    for fam in master["family"].unique():
        sub = master[master["family"] == fam]
        for dkey in DATASETS:
            d = sub[sub["dataset"] == dkey]["qwk"].to_numpy(dtype=float)
            if len(d) == 0:
                continue
            agg_rows.append({
                "family": fam, "dataset": dkey, "n_seeds": len(d),
                "qwk_mean": round(float(d.mean()), 4),
                "qwk_sd": round(float(d.std(ddof=1)), 4) if len(d) > 1 else 0.0,
                "qwk_min": round(float(d.min()), 4),
                "qwk_max": round(float(d.max()), 4),
                "qwk_range": round(float(d.max() - d.min()), 4),
            })
        # per-seed internal->external drop
        for seed in sorted(sub["seed"].unique()):
            s = sub[sub["seed"] == seed].set_index("dataset")["qwk"]
            if "APTOS-internal" not in s.index:
                continue
            for dkey in ("Messidor-2", "IDRiD"):
                if dkey in s.index:
                    drop_rows.append({
                        "family": fam, "seed": int(seed), "external": dkey,
                        "qwk_internal": float(s["APTOS-internal"]),
                        "qwk_external": float(s[dkey]),
                        "qwk_drop": round(float(s["APTOS-internal"] - s[dkey]), 4),
                    })

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT_DIR / "seed_qwk_aggregate.csv", index=False)
    drop = pd.DataFrame(drop_rows)
    drop.to_csv(OUT_DIR / "seed_drop.csv", index=False)

    drop_agg_rows: list[dict] = []
    for fam in drop["family"].unique():
        for dkey in ("Messidor-2", "IDRiD"):
            v = drop[(drop["family"] == fam) & (drop["external"] == dkey)]["qwk_drop"].to_numpy(float)
            if len(v):
                drop_agg_rows.append({
                    "family": fam, "external": dkey, "n_seeds": len(v),
                    "drop_mean": round(float(v.mean()), 4),
                    "drop_sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
                    "drop_min": round(float(v.min()), 4),
                    "drop_max": round(float(v.max()), 4),
                })
    drop_agg = pd.DataFrame(drop_agg_rows)
    drop_agg.to_csv(OUT_DIR / "seed_drop_aggregate.csv", index=False)

    lines = ["# Phase 16 -- training-seed sensitivity", "",
             f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
             "",
             "Extra full-fine-tune runs scored unmodified on the Phase-6 code path. "
             "Primary model = seed 42.", "",
             "## QWK across seeds (mean +/- SD [min, max])", "",
             "| Family | Dataset | seeds | QWK mean | SD | min | max | range |",
             "|---|---|--:|--:|--:|--:|--:|--:|"]
    for _, r in agg.iterrows():
        lines.append(f"| {r['family']} | {r['dataset']} | {int(r['n_seeds'])} | "
                     f"{r['qwk_mean']:.3f} | {r['qwk_sd']:.3f} | {r['qwk_min']:.3f} | "
                     f"{r['qwk_max']:.3f} | {r['qwk_range']:.3f} |")
    lines += ["", "## Internal->external QWK drop across seeds", "",
              "| Family | External | seeds | drop mean | SD | min | max |",
              "|---|---|--:|--:|--:|--:|--:|"]
    for _, r in drop_agg.iterrows():
        lines.append(f"| {r['family']} | {r['external']} | {int(r['n_seeds'])} | "
                     f"{r['drop_mean']:.3f} | {r['drop_sd']:.3f} | {r['drop_min']:.3f} | "
                     f"{r['drop_max']:.3f} |")
    lines += ["", "Per-seed detail: `seed_master.csv`, `seed_drop.csv`."]
    (OUT_DIR / "seed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 70)
    print(agg.to_string(index=False))
    print()
    print(drop_agg.to_string(index=False))
    print(f"\nwrote {OUT_DIR}/seed_master.csv, seed_qwk_aggregate.csv, seed_drop*.csv, seed_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
