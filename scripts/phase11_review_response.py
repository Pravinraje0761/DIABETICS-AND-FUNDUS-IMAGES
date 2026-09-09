"""
Phase 11 - peer-review-response analyses, all from the Phase 6 per-image
prediction files (outputs/phase6/<model>__<dataset>_preds.csv). No re-inference.

Produces (outputs/phase11/):
  verification.csv        every headline metric recomputed from preds vs the
                          value in outputs/phase6/master_results.csv (mismatch
                          flagged if |delta| > 5e-4)
  per_class_ci.csv        per ICDR grade x model x dataset: support, sensitivity,
                          specificity, precision, F1, each with a 95% bootstrap CI
  calibration.csv         ECE(15 bin), MCE, Brier, NLL, per model x dataset,
                          internal and external, with 95% bootstrap CIs on ECE
  reliability_<key>.csv   bin_conf / bin_acc / bin_count for reliability diagrams
  referable_roc_pr.csv    fpr/tpr and precision/recall for referable DR
                          (grade>=2), per model x dataset (for ROC/PR figures)
  multiplicity.csv        Holm-Bonferroni-adjusted p-values for the DeLong family
  training_accounting.csv exact train/val/test image and per-class counts
  domain_shift.md         APTOS vs Messidor-2 vs IDRiD across acquisition axes
  phase11_summary.md      digest

Run:
  .venv/Scripts/python.exe scripts/phase11_review_response.py --boot 2000 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import GRADE_NAMES, per_class_sens_spec  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
P6 = PROJECT / "outputs" / "phase6"
P9 = PROJECT / "outputs" / "phase9"
OUT = PROJECT / "outputs" / "phase11"
SPLIT = PROJECT / "data_splits"

MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
          "retfound_linprobe", "retfound_full"]
DATASETS = ["APTOS-internal", "Messidor-2", "IDRiD"]
PROB = [f"prob_{c}" for c in range(5)]


def load(m, d):
    p = P6 / f"{m}__{d}_preds.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["id_code"] = df["id_code"].astype(str)
    return df


def qwk(yt, yp):
    return float(cohen_kappa_score(yt, yp, labels=[0, 1, 2, 3, 4], weights="quadratic"))


def macro_auc(yt, pr):
    if len(np.unique(yt)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(yt, pr, multi_class="ovr", average="macro",
                                   labels=[0, 1, 2, 3, 4]))
    except ValueError:
        return np.nan


def verification() -> pd.DataFrame:
    master = pd.read_csv(P6 / "master_results.csv")
    rows = []
    for m in MODELS:
        for d in DATASETS:
            df = load(m, d)
            if df is None:
                continue
            yt = df["y_true"].to_numpy(int)
            yp = df["y_pred"].to_numpy(int)
            pr = df[PROB].to_numpy(float)
            sens, _ = per_class_sens_spec(
                confusion_matrix(yt, yp, labels=[0, 1, 2, 3, 4]))
            rec = {
                "accuracy": accuracy_score(yt, yp),
                "quadratic_weighted_kappa": qwk(yt, yp),
                "macro_f1": f1_score(yt, yp, labels=[0, 1, 2, 3, 4],
                                     average="macro", zero_division=0),
                "balanced_accuracy": float(np.nanmean(sens)),
                "auc_ovr_macro": macro_auc(yt, pr),
            }
            ref = master[(master.model == m) & (master.dataset == d)]
            for k, val in rec.items():
                mv = float(ref[k].iloc[0]) if len(ref) and k in ref else np.nan
                rows.append({"model": m, "dataset": d, "metric": k,
                             "recomputed": round(val, 6),
                             "in_master_csv": round(mv, 6),
                             "abs_delta": round(abs(val - mv), 6) if np.isfinite(mv) else np.nan,
                             "MATCH": bool(np.isfinite(mv) and abs(val - mv) < 5e-4)})
    return pd.DataFrame(rows)


def _ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if a.size else (np.nan, np.nan)


def per_class_ci(B, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in MODELS:
        for d in DATASETS:
            df = load(m, d)
            if df is None:
                continue
            yt = df["y_true"].to_numpy(int)
            yp = df["y_pred"].to_numpy(int)
            n = len(yt)
            boot = {c: {"sens": [], "spec": [], "prec": [], "f1": []} for c in range(5)}
            for _ in range(B):
                idx = rng.integers(0, n, n)
                cm = confusion_matrix(yt[idx], yp[idx],
                                      labels=[0, 1, 2, 3, 4]).astype(float)
                s, sp = per_class_sens_spec(cm)
                for c in range(5):
                    tp = cm[c, c]
                    prec = tp / cm[:, c].sum() if cm[:, c].sum() else np.nan
                    rec = s[c]
                    f1 = (2 * prec * rec / (prec + rec)
                          if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
                          else np.nan)
                    boot[c]["sens"].append(s[c]); boot[c]["spec"].append(sp[c])
                    boot[c]["prec"].append(prec); boot[c]["f1"].append(f1)
            cm0 = confusion_matrix(yt, yp, labels=[0, 1, 2, 3, 4]).astype(float)
            s0, sp0 = per_class_sens_spec(cm0)
            for c in range(5):
                tp = cm0[c, c]
                prec0 = tp / cm0[:, c].sum() if cm0[:, c].sum() else np.nan
                rec0 = s0[c]
                f10 = (2 * prec0 * rec0 / (prec0 + rec0)
                       if np.isfinite(prec0) and np.isfinite(rec0) and (prec0 + rec0) > 0
                       else np.nan)
                rows.append({
                    "model": m, "dataset": d, "grade": GRADE_NAMES[c],
                    "support": int(cm0[c].sum()),
                    "sensitivity": round(rec0, 4),
                    "sens_lo": round(_ci(boot[c]["sens"])[0], 4),
                    "sens_hi": round(_ci(boot[c]["sens"])[1], 4),
                    "specificity": round(sp0[c], 4),
                    "spec_lo": round(_ci(boot[c]["spec"])[0], 4),
                    "spec_hi": round(_ci(boot[c]["spec"])[1], 4),
                    "precision": round(prec0, 4) if np.isfinite(prec0) else np.nan,
                    "prec_lo": round(_ci(boot[c]["prec"])[0], 4),
                    "prec_hi": round(_ci(boot[c]["prec"])[1], 4),
                    "f1": round(f10, 4) if np.isfinite(f10) else np.nan,
                    "f1_lo": round(_ci(boot[c]["f1"])[0], 4),
                    "f1_hi": round(_ci(boot[c]["f1"])[1], 4),
                })
    return pd.DataFrame(rows)


def _cal(y, conf, correct, prob_full, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    rel = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rel.append((np.nan, np.nan, 0)); continue
        acc = correct[m].mean(); c = conf[m].mean(); gap = abs(acc - c)
        ece += (m.sum() / len(conf)) * gap
        mce = max(mce, gap)
        rel.append((c, acc, int(m.sum())))
    brier = float(np.mean(np.sum((prob_full - np.eye(5)[y]) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(prob_full[np.arange(len(y)), y], 1e-12, 1))))
    return float(ece), float(mce), brier, nll, rel


def calibration(B, seed):
    rng = np.random.default_rng(seed)
    rows, reliab = [], {}
    for m in MODELS:
        for d in DATASETS:
            df = load(m, d)
            if df is None:
                continue
            yt = df["y_true"].to_numpy(int)
            pr = df[PROB].to_numpy(float)
            conf = pr.max(1)
            correct = (pr.argmax(1) == yt).astype(float)
            ece, mce, brier, nll, rel = _cal(yt, conf, correct, pr)
            reliab[f"{m}__{d}"] = rel
            n = len(yt)
            b = [_cal(yt[i], conf[i], correct[i], pr[i])[0]
                 for i in (rng.integers(0, n, n) for _ in range(B))]
            lo, hi = np.percentile(b, [2.5, 97.5])
            rows.append({"model": m, "dataset": d, "n": n,
                         "ECE": round(ece, 4), "ECE_lo": round(lo, 4),
                         "ECE_hi": round(hi, 4), "MCE": round(mce, 4),
                         "Brier": round(brier, 4), "NLL": round(nll, 4),
                         "split": "internal" if d == "APTOS-internal" else "external"})
    return pd.DataFrame(rows), reliab


def referable_roc_pr() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            df = load(m, d)
            if df is None:
                continue
            y = (df["y_true"].to_numpy(int) >= 2).astype(int)
            s = df[["prob_2", "prob_3", "prob_4"]].sum(1).to_numpy()
            if y.sum() in (0, len(y)):
                continue
            fpr, tpr, _ = roc_curve(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            auc = float(roc_auc_score(y, s))
            for a, bb in zip(fpr, tpr):
                rows.append({"model": m, "dataset": d, "curve": "roc",
                             "x": round(float(a), 4), "y": round(float(bb), 4),
                             "auc": round(auc, 4)})
            for p, r in zip(prec, rec):
                rows.append({"model": m, "dataset": d, "curve": "pr",
                             "x": round(float(r), 4), "y": round(float(p), 4),
                             "auc": round(auc, 4)})
    return pd.DataFrame(rows)


def multiplicity() -> pd.DataFrame:
    out = []
    dl = P9 / "delong.csv"
    if dl.is_file():
        d = pd.read_csv(dl)
        fam = d[d.target == "referable_DR(grade>=2)"].copy()
        p = fam["delong_p"].to_numpy(float)
        order = np.argsort(p)
        adj = np.empty_like(p)
        run_max = 0.0
        k = len(p)
        for rank, i in enumerate(order):
            run_max = max(run_max, min((k - rank) * p[i], 1.0))
            adj[i] = run_max
        fam["p_holm"] = np.round(adj, 4)
        for _, r in fam.iterrows():
            out.append({"family": "DeLong referable-DR (vs EfficientNet-B3), k=%d" % k,
                        "comparison": f"{r['dataset']} : {r['model_other']}",
                        "p_raw": round(float(r["delong_p"]), 4),
                        "p_holm": float(r["p_holm"]),
                        "sig_holm_0.05": bool(r["p_holm"] < 0.05)})
    return pd.DataFrame(out)


def training_accounting() -> pd.DataFrame:
    rows = []
    man = SPLIT / "aptos_manifest.csv"
    if man.is_file():
        df = pd.read_csv(man)
        df = df[df["processed_path"].notna()]
        for sp in ["train", "val", "test"]:
            s = df[df["split"] == sp]
            c = s["diagnosis"].value_counts().reindex(range(5), fill_value=0)
            rows.append({"split": f"APTOS {sp}", "n": len(s),
                         **{f"grade_{i}": int(c[i]) for i in range(5)}})
        c = df["diagnosis"].value_counts().reindex(range(5), fill_value=0)
        rows.append({"split": "APTOS total (labelled)", "n": len(df),
                     **{f"grade_{i}": int(c[i]) for i in range(5)}})
    for d, m2 in [("Messidor-2", SPLIT / "messidor2_manifest.csv"),
                  ("IDRiD", SPLIT / "idrid_manifest.csv")]:
        if m2.is_file():
            df = pd.read_csv(m2)
            df = df[df["processed_path"].notna()]
            c = df["diagnosis"].value_counts().reindex(range(5), fill_value=0)
            rows.append({"split": f"{d} (external)", "n": len(df),
                         **{f"grade_{i}": int(c[i]) for i in range(5)}})
    return pd.DataFrame(rows)


DOMAIN_SHIFT_MD = """\
# Domain-shift characterisation: APTOS 2019 vs Messidor-2 vs IDRiD

_Compiled from the dataset publications and `DATA_PROVENANCE.md`. "Not published"
means the axis is not reported in the public releases used here._

| Axis | APTOS 2019 (development) | Messidor-2 (external) | IDRiD (external) |
|---|---|---|---|
| Country / population | India (Aravind Eye Hospital); telescreening | France (Messidor programme partners) | India (Kasturba Medical College, Manipal) |
| Camera / device | Multiple, not individually published (multi-site) | Topcon TRC NW6 non-mydriatic | Kowa VX-10 alpha / Nidek AFC-210 |
| Native resolution | Heterogeneous; ~1.5-3 MP after vendor processing | 1440x960 to 2304x1536 | 4288x2848 |
| Field of view | ~45 deg, macula-centred | ~45 deg, macula-centred | 50 deg, macula-centred |
| Illumination / colour | Variable across sites; partly pre-processed by the Kaggle release | more uniform (single programme) | high-resolution, well-exposed; least colour variation |
| Mydriasis | mixed / not published | non-mydriatic | mydriatic |
| Grading scale | ICDR 0-4 | ICDR 0-4 | ICDR 0-4 |
| Reference standard | single grader (Kaggle labels) | **adjudicated consensus** (Krause et al. 2018) | expert consensus (release) |
| Grade prevalence 0/1/2/3/4 (%) | 49.3 / 10.1 / 27.3 / 5.3 / 8.1 | 58.3 / 15.5 / 19.9 / 4.3 / 2.0 | 28.4 / 4.8 / 34.3 / 18.5 / 14.1 |
| Referable DR (grade >= 2), % | 40.7 | 26.2 | 67.6 |
| Image quality | variable; some low-quality frames retained | curated; adjudicated-gradable subset only | high and uniform |
| Preprocessing here | single Phase 3 pipeline (crop / pad / 224 / ImageNet norm) | same, all 1744 | same, all 455 |

**Interpretation.** Messidor-2 differs from APTOS most on the reference standard
(adjudicated consensus vs single grader) and acquisition programme, and least on
FOV/scale; its referable-DR prevalence is *lower* (26% vs 41%). IDRiD differs most
on resolution, mydriasis, and prevalence (referable DR 68%), and least on colour
uniformity. This matches the observed failure modes: on Messidor-2 (stricter
labels, similar low-severity skew) QWK collapses by an ordinal-structure
mechanism; on IDRiD (very different base rate) accuracy falls more than QWK.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/7] verification ...")
    v = verification()
    v.to_csv(OUT / "verification.csv", index=False)
    nbad = int((~v["MATCH"]).sum())

    print("[2/7] per-class CIs ...")
    per_class_ci(args.boot, args.seed).to_csv(OUT / "per_class_ci.csv", index=False)

    print("[3/7] calibration ...")
    cal, reliab = calibration(args.boot, args.seed)
    cal.to_csv(OUT / "calibration.csv", index=False)
    for key, rel in reliab.items():
        pd.DataFrame(rel, columns=["bin_conf", "bin_acc", "bin_count"]).to_csv(
            OUT / f"reliability_{key}.csv", index=False)

    print("[4/7] referable-DR ROC/PR ...")
    referable_roc_pr().to_csv(OUT / "referable_roc_pr.csv", index=False)

    print("[5/7] multiplicity ...")
    multiplicity().to_csv(OUT / "multiplicity.csv", index=False)

    print("[6/7] training accounting ...")
    training_accounting().to_csv(OUT / "training_accounting.csv", index=False)

    print("[7/7] domain-shift table ...")
    (OUT / "domain_shift.md").write_text(DOMAIN_SHIFT_MD, encoding="utf-8")

    L = ["# Phase 11 - review-response analyses", "",
         f"Verification: {len(v)} metric cells recomputed from prediction files; "
         f"**{nbad} mismatch(es)** vs `master_results.csv` (tolerance 5e-4).", ""]
    if nbad:
        for _, r in v[~v["MATCH"]].iterrows():
            L.append(f"- {r['model']}/{r['dataset']}/{r['metric']}: "
                     f"recomputed {r['recomputed']} vs csv {r['in_master_csv']}")
        L.append("")
    L += ["## Calibration (ECE / MCE / Brier / NLL)", "",
          "| Model | Dataset | ECE (95% CI) | MCE | Brier | NLL |",
          "|---|---|---|---|---|---|"]
    for _, r in cal.iterrows():
        L.append(f"| {r['model']} | {r['dataset']} | "
                 f"{r['ECE']:.3f} ({r['ECE_lo']:.3f}-{r['ECE_hi']:.3f}) | "
                 f"{r['MCE']:.3f} | {r['Brier']:.3f} | {r['NLL']:.3f} |")
    L += ["", "Files: `per_class_ci.csv`, `referable_roc_pr.csv`, "
          "`multiplicity.csv`, `training_accounting.csv`, `domain_shift.md`.", ""]
    (OUT / "phase11_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nwrote {OUT}/  ({nbad} verification mismatches)")
    print(cal.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
