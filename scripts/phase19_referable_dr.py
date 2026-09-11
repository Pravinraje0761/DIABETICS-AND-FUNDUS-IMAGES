"""
Phase 19 -- referable-DR as a standalone secondary clinical task.

Phase 9 already computes a referable-DR (ICDR grade >= 2) AUC as part of the
DeLong paired-AUC test, but only as one input to the 5-class-grading analysis.
This script promotes it to a full secondary task with its own operating-point
metrics, mirroring the primary-task structure (headline metric with CI,
internal->external drop with CI) so the manuscript can state explicitly
whether the RETFound-vs-baseline generalisation gap "persists beyond 5-class
grading" -- as requested in the review-response checklist.

No re-inference: everything is computed from the Phase 6 per-image prediction
CSVs (outputs/phase6/<model>__<dataset>_preds.csv), y_true/y_pred/prob_0..4.

Referable-DR label: ICDR grade >= 2 (the clinically standard referable
threshold, same binarisation Phase 9's DeLong test already uses).
Decision-based metrics use the 5-class model's own argmax prediction collapsed
to referable/non-referable (y_pred >= 2) -- i.e. what a deployed 5-class
grader would output, not a separately-tuned binary threshold. Threshold-free
metrics (AUROC, AUPRC) use the summed probability P(grade>=2). Sensitivity at
a fixed 85% specificity operating point (a common DR-screening target) is
reported alongside, from the ROC curve.

Outputs (outputs/phase19/):
  referable_dr_master.csv    model x dataset: n, n_referable, decision-based
                             sens/spec/PPV/NPV/accuracy, AUROC (95% CI),
                             AUPRC (95% CI), sens@85spec, spec@85sens
  referable_dr_drop.csv      internal->external AUROC drop per model, with CI
  referable_dr_summary.md    manuscript digest, cross-references Phase 9's
                             DeLong test (not recomputed here -- see
                             outputs/phase9/delong.csv, target
                             "referable_DR(grade>=2)")

Run:
  .venv/Scripts/python.exe scripts/phase19_referable_dr.py
  .venv/Scripts/python.exe scripts/phase19_referable_dr.py --boot 5000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parents[1]
PHASE6_DIR = PROJECT_DIR / "outputs" / "phase6"
PHASE9_DELONG = PROJECT_DIR / "outputs" / "phase9" / "delong.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "phase19"

MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
         "retfound_linprobe", "retfound_full"]
DATASETS = ["APTOS-internal", "Messidor-2", "IDRiD"]
EXTERNALS = ["Messidor-2", "IDRiD"]
BASELINE = "baseline_efficientnet_b3"
PROB_COLS = [f"prob_{c}" for c in range(5)]
ALPHA = 0.05


def load_preds(model: str, dataset: str) -> pd.DataFrame | None:
    p = PHASE6_DIR / f"{model}__{dataset}_preds.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["id_code"] = df["id_code"].astype(str)
    return df


def referable_arrays(df: pd.DataFrame) -> dict:
    y_true = (df["y_true"].to_numpy(int) >= 2).astype(int)
    y_pred = (df["y_pred"].to_numpy(int) >= 2).astype(int)
    score = df[["prob_2", "prob_3", "prob_4"]].to_numpy(float).sum(axis=1)
    return {"y_true": y_true, "y_pred": y_pred, "score": score}


def decision_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / len(y_true) if len(y_true) else float("nan")
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def threshold_free_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    if y_true.min() == y_true.max():
        return {"auroc": float("nan"), "auprc": float("nan"),
                "sens_at_85spec": float("nan"), "spec_at_85sens": float("nan")}
    auroc = float(roc_auc_score(y_true, score))
    auprc = float(average_precision_score(y_true, score))
    fpr, tpr, _ = roc_curve(y_true, score)
    spec_arr = 1.0 - fpr
    # nearest achievable operating point at/above the target
    ok = spec_arr >= 0.85
    sens_at_85spec = float(tpr[ok].max()) if ok.any() else float("nan")
    ok2 = tpr >= 0.85
    spec_at_85sens = float(spec_arr[ok2].max()) if ok2.any() else float("nan")
    return {"auroc": auroc, "auprc": auprc, "sens_at_85spec": sens_at_85spec,
            "spec_at_85sens": spec_at_85sens}


def boot_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


BOOT_KEYS = ["auroc", "auprc", "sensitivity", "specificity", "accuracy"]


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray,
                 B: int, seed: int) -> dict:
    point = {**decision_metrics(y_true, y_pred), **threshold_free_metrics(y_true, score)}
    n = len(y_true)
    rng = np.random.default_rng(seed)
    draws = {k: np.full(B, np.nan) for k in BOOT_KEYS}
    for b in range(B):
        idx = boot_indices(n, rng)
        yt, yp, sc = y_true[idx], y_pred[idx], score[idx]
        if len(np.unique(yt)) < 2:
            continue
        dm = decision_metrics(yt, yp)
        tm = threshold_free_metrics(yt, sc)
        draws["auroc"][b] = tm["auroc"]
        draws["auprc"][b] = tm["auprc"]
        draws["sensitivity"][b] = dm["sensitivity"]
        draws["specificity"][b] = dm["specificity"]
        draws["accuracy"][b] = dm["accuracy"]
    ci = {}
    for k in BOOT_KEYS:
        d = draws[k][~np.isnan(draws[k])]
        ci[k] = (np.nan, np.nan) if d.size == 0 else tuple(np.percentile(d, [2.5, 97.5]))
    return {"point": point, "ci": ci}


def run_master(B: int, seed: int) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            df = load_preds(model, ds)
            if df is None:
                continue
            arr = referable_arrays(df)
            res = bootstrap_ci(arr["y_true"], arr["y_pred"], arr["score"], B, seed)
            pt = res["point"]
            n_ref = int(arr["y_true"].sum())
            rows.append({
                "model": model, "dataset": ds, "n": len(df), "n_referable": n_ref,
                "sensitivity": pt["sensitivity"], "sens_ci_lo": res["ci"]["sensitivity"][0],
                "sens_ci_hi": res["ci"]["sensitivity"][1],
                "specificity": pt["specificity"], "spec_ci_lo": res["ci"]["specificity"][0],
                "spec_ci_hi": res["ci"]["specificity"][1],
                "ppv": pt["ppv"], "npv": pt["npv"],
                "accuracy": pt["accuracy"], "acc_ci_lo": res["ci"]["accuracy"][0],
                "acc_ci_hi": res["ci"]["accuracy"][1],
                "auroc": pt["auroc"], "auroc_ci_lo": res["ci"]["auroc"][0],
                "auroc_ci_hi": res["ci"]["auroc"][1],
                "auprc": pt["auprc"], "auprc_ci_lo": res["ci"]["auprc"][0],
                "auprc_ci_hi": res["ci"]["auprc"][1],
                "sens_at_85spec": pt["sens_at_85spec"],
                "spec_at_85sens": pt["spec_at_85sens"],
            })
    return pd.DataFrame(rows)


def run_drop(master: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """Internal->external AUROC drop per model, with a bootstrap CI (internal
    and external resampled independently), mirroring Phase 9's QWK-drop table."""
    rows = []
    for model in MODELS:
        di = load_preds(model, "APTOS-internal")
        if di is None:
            continue
        ai = referable_arrays(di)
        if ai["y_true"].min() == ai["y_true"].max():
            continue
        auroc_i = float(roc_auc_score(ai["y_true"], ai["score"]))
        for ds in EXTERNALS:
            de = load_preds(model, ds)
            if de is None:
                continue
            ae = referable_arrays(de)
            if ae["y_true"].min() == ae["y_true"].max():
                continue
            auroc_e = float(roc_auc_score(ae["y_true"], ae["score"]))

            rng = np.random.default_rng(seed)
            ni, ne = len(ai["y_true"]), len(ae["y_true"])
            draws = np.full(B, np.nan)
            for b in range(B):
                ii, ie = boot_indices(ni, rng), boot_indices(ne, rng)
                yti, sci = ai["y_true"][ii], ai["score"][ii]
                yte, sce = ae["y_true"][ie], ae["score"][ie]
                if len(np.unique(yti)) < 2 or len(np.unique(yte)) < 2:
                    continue
                draws[b] = (roc_auc_score(yti, sci) - roc_auc_score(yte, sce))
            d = draws[~np.isnan(draws)]
            lo, hi = (np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
            rows.append({"model": model, "external_dataset": ds,
                        "auroc_internal": auroc_i, "auroc_external": auroc_e,
                        "drop": auroc_i - auroc_e, "drop_ci_lo": lo, "drop_ci_hi": hi})
    return pd.DataFrame(rows)


def fmt_ci(pt, lo, hi, prec=3):
    if not np.isfinite(pt):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{pt:.{prec}f}"
    return f"{pt:.{prec}f} ({lo:.{prec}f}-{hi:.{prec}f})"


def write_summary(master: pd.DataFrame, drop: pd.DataFrame, B: int, seed: int) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = ["# Phase 19 -- referable-DR (ICDR grade >= 2) secondary task", "",
        f"_Generated {ts}. Referable = ICDR grade >= 2. Decision-based metrics "
        f"use the 5-class model's own argmax (y_pred >= 2); AUROC/AUPRC use "
        f"summed P(grade>=2). Percentile bootstrap, B={B}, seed={seed}._", "",
        "## 1. Headline metrics with 95% CI", "",
        "| Model | Dataset | n | n-ref | Sens | Spec | AUROC | AUPRC | "
        "Sens@85%spec | Spec@85%sens |",
        "|---|---|--:|--:|---|---|---|---|--:|--:|"]
    for r in master.itertuples():
        L.append(f"| {r.model} | {r.dataset} | {r.n} | {r.n_referable} | "
                 f"{fmt_ci(r.sensitivity, r.sens_ci_lo, r.sens_ci_hi)} | "
                 f"{fmt_ci(r.specificity, r.spec_ci_lo, r.spec_ci_hi)} | "
                 f"{fmt_ci(r.auroc, r.auroc_ci_lo, r.auroc_ci_hi)} | "
                 f"{fmt_ci(r.auprc, r.auprc_ci_lo, r.auprc_ci_hi)} | "
                 f"{r.sens_at_85spec:.3f} | {r.spec_at_85sens:.3f} |")

    L += ["", "## 2. Internal->external AUROC drop with 95% CI", "",
         "Mirrors the primary-task QWK-drop table (Phase 9 sec. 2) for the "
         "referable-DR binarisation, so the manuscript can state whether the "
         "generalisation gap persists beyond 5-class ordinal grading.", "",
         "| Model | External set | AUROC internal | AUROC external | Drop (95% CI) |",
         "|---|---|--:|--:|---|"]
    for r in drop.itertuples():
        L.append(f"| {r.model} | {r.external_dataset} | {r.auroc_internal:.3f} | "
                 f"{r.auroc_external:.3f} | {fmt_ci(r.drop, r.drop_ci_lo, r.drop_ci_hi)} |")

    L += ["", "## 3. Between-model AUROC difference (significance test)", "",
         "Not recomputed here -- see the existing DeLong paired-AUC test in "
         "`outputs/phase9/delong.csv` (target = `referable_DR(grade>=2)`), "
         "which uses this exact binarisation and score.", ""]
    (OUT_DIR / "referable_dr_summary.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not PHASE6_DIR.is_dir():
        print(f"missing {PHASE6_DIR} - run phase6_external_eval.py first")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 19: referable-DR secondary task, B={args.boot}, seed={args.seed}")
    master = run_master(args.boot, args.seed)
    master.to_csv(OUT_DIR / "referable_dr_master.csv", index=False)
    drop = run_drop(master, args.boot, args.seed)
    drop.to_csv(OUT_DIR / "referable_dr_drop.csv", index=False)
    write_summary(master, drop, args.boot, args.seed)

    print(master.to_string(index=False))
    print()
    print(drop.to_string(index=False))
    print(f"\nwrote {OUT_DIR}/referable_dr_master.csv, referable_dr_drop.csv, "
         "referable_dr_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
