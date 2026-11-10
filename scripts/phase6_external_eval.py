"""
Phase 6 - external validation. The core experiment.

Run each trained model, UNMODIFIED and with no further training, on:
  - APTOS internal test  (the Phase 3 split - re-scored here for one clean
    provenance / identical code path)
  - Messidor-2  (external, all 1,744 gradable)
  - IDRiD       (external, all 455)

Produces the master results table (models x datasets x metrics) that is the
spine of the Results section, plus the internal->external performance drop for
each model - the paper's headline number.

Raw per-image predictions (id_code, y_true, y_pred, prob_0..4, entropy) are kept
for every model x dataset combination - Phase 8 (uncertainty) and Phase 9
(bootstrap CIs, DeLong) consume them.

Run (after Phase 4 + Phase 5 checkpoints exist):
  .venv/Scripts/python.exe scripts/phase6_external_eval.py

Optional: --models baseline_efficientnet_b3 retfound_full   (subset)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_dataloader  # noqa: E402
from metrics import GRADE_NAMES, format_metrics  # noqa: E402
from predict import infer, load_checkpoint, pick_device, save_predictions  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase6"

CANDIDATE_CKPTS = {
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

HEADLINE = ["accuracy", "quadratic_weighted_kappa", "macro_f1",
            "balanced_accuracy", "auc_ovr_macro"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset of: " + ", ".join(CANDIDATE_CKPTS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    models = args.models or [k for k, p in CANDIDATE_CKPTS.items() if p.is_file()]
    missing = [m for m in models if not CANDIDATE_CKPTS[m].is_file()]
    if missing:
        print(f"skipping (checkpoint not found): {missing}")
        models = [m for m in models if m not in missing]
    if not models:
        print("no checkpoints available - run Phase 4 / Phase 5 first")
        return 1
    print(f"models: {models}")
    print(f"datasets: {list(DATASETS)}")

    rows: list[dict] = []
    per_class_rows: list[dict] = []

    for mkey in models:
        model, _ckpt = load_checkpoint(CANDIDATE_CKPTS[mkey], device)
        for dkey, (manifest, split) in DATASETS.items():
            if not manifest.is_file():
                print(f"  {dkey}: manifest missing, skip")
                continue
            loader = build_dataloader(manifest, split=split,
                                      batch_size=args.batch_size, train=False,
                                      num_workers=args.num_workers, shuffle=False)
            res = infer(model, loader, device)
            prefix = OUT_DIR / f"{mkey}__{dkey}"
            _, metrics_path = save_predictions(res, prefix)
            m = json.loads(metrics_path.read_text())

            row = {"model": mkey, "dataset": dkey, "n": m["n"]}
            row.update({k: m.get(k) for k in HEADLINE})
            rows.append(row)
            for gname in GRADE_NAMES:
                pc = m["per_class"][gname]
                per_class_rows.append({
                    "model": mkey, "dataset": dkey, "grade": gname,
                    "support": pc["support"],
                    "sensitivity": pc["sensitivity"],
                    "specificity": pc["specificity"],
                    "auc": m.get("auc_ovr_per_class", {}).get(gname),
                })
            print(format_metrics(m, title=f"{mkey}  x  {dkey}"))
            print()

    master = pd.DataFrame(rows)
    per_class_df = pd.DataFrame(per_class_rows)
    # Review fix (2026-10-21): a run with --models <subset> used to OVERWRITE the
    # master tables with only that subset. Keep rows of models not re-run.
    for _name, _df in (("master_results.csv", "master"),
                       ("master_results_per_class.csv", "per_class_df")):
        _f = OUT_DIR / _name
        if _f.is_file():
            _old = pd.read_csv(_f)
            _new = master if _df == "master" else per_class_df
            _keep = _old[~_old["model"].isin(_new["model"].unique())]
            _merged = pd.concat([_keep, _new], ignore_index=True)
            if _df == "master":
                master = _merged
            else:
                per_class_df = _merged
    master.to_csv(OUT_DIR / "master_results.csv", index=False)
    per_class_df.to_csv(OUT_DIR / "master_results_per_class.csv",
                                       index=False)

    # internal -> external drop, per model per metric
    drop_rows: list[dict] = []
    for mkey in master["model"].unique():
        sub = master[master["model"] == mkey].set_index("dataset")
        if "APTOS-internal" not in sub.index:
            continue
        base = sub.loc["APTOS-internal"]
        for dkey in ("Messidor-2", "IDRiD"):
            if dkey not in sub.index:
                continue
            ext = sub.loc[dkey]
            drow = {"model": mkey, "external_dataset": dkey}
            for k in HEADLINE:
                if base.get(k) is not None and ext.get(k) is not None:
                    drow[f"{k}_internal"] = round(float(base[k]), 4)
                    drow[f"{k}_external"] = round(float(ext[k]), 4)
                    drow[f"{k}_drop"] = round(float(base[k]) - float(ext[k]), 4)
            drop_rows.append(drow)
    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(OUT_DIR / "internal_vs_external_drop.csv", index=False)

    # markdown master table
    md = ["# Phase 6 - Master results", "",
          f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
          "", "## Master table (model x dataset)", "",
          "| Model | Dataset | n | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |",
          "|---|---|--:|--:|--:|--:|--:|--:|"]
    for _, r in master.iterrows():
        auc = r["auc_ovr_macro"]
        md.append(
            f"| {r['model']} | {r['dataset']} | {int(r['n'])} | "
            f"{r['accuracy']:.3f} | {r['quadratic_weighted_kappa']:.3f} | "
            f"{r['macro_f1']:.3f} | {r['balanced_accuracy']:.3f} | "
            f"{auc:.3f} |" if pd.notna(auc) else
            f"| {r['model']} | {r['dataset']} | {int(r['n'])} | "
            f"{r['accuracy']:.3f} | {r['quadratic_weighted_kappa']:.3f} | "
            f"{r['macro_f1']:.3f} | {r['balanced_accuracy']:.3f} | n/a |"
        )
    md += ["", "## Internal -> external drop (QWK)", "",
           "| Model | External set | QWK internal | QWK external | Drop |",
           "|---|---|--:|--:|--:|"]
    for _, r in drop_df.iterrows():
        if "quadratic_weighted_kappa_drop" in r and pd.notna(r.get("quadratic_weighted_kappa_drop")):
            md.append(
                f"| {r['model']} | {r['external_dataset']} | "
                f"{r['quadratic_weighted_kappa_internal']:.3f} | "
                f"{r['quadratic_weighted_kappa_external']:.3f} | "
                f"{r['quadratic_weighted_kappa_drop']:.3f} |"
            )
    md += ["",
           "Raw per-image predictions: `outputs/phase6/<model>__<dataset>_preds.csv` "
           "(kept for Phase 8 uncertainty + Phase 9 bootstrap/DeLong)."]
    (OUT_DIR / "master_results.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 70)
    print("MASTER TABLE")
    print(master.to_string(index=False))
    print("\nINTERNAL -> EXTERNAL DROP")
    cols = [c for c in ("model", "external_dataset", "quadratic_weighted_kappa_drop",
                        "accuracy_drop", "auc_ovr_macro_drop", "macro_f1_drop")
            if c in drop_df.columns]
    print(drop_df[cols].to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'master_results.csv'}")
    print(f"wrote {OUT_DIR / 'master_results.md'}")
    print(f"wrote {OUT_DIR / 'internal_vs_external_drop.csv'}")
    print(f"wrote {OUT_DIR / 'master_results_per_class.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
