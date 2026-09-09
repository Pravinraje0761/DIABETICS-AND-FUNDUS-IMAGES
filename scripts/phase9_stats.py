"""
Phase 9 - statistical analysis.

Everything here is computed from the Phase 6 per-image prediction CSVs
(`outputs/phase6/<model>__<dataset>_preds.csv`); no model re-inference.

Produces:
  outputs/phase9/bootstrap_ci.csv        95% CI for accuracy / QWK / macro-F1 /
                                         balanced-accuracy / macro-AUC, every
                                         model x dataset (percentile bootstrap)
  outputs/phase9/drop_ci.csv             internal->external drop per metric, per
                                         model, per external dataset, with CI
  outputs/phase9/drop_difference_ci.csv  difference in QWK drop, RETFound vs the
                                         EfficientNet-B3 baseline (paired
                                         bootstrap on shared image IDs)
  outputs/phase9/delong.csv              DeLong paired-AUC test, baseline vs each
                                         other model, per dataset, for the
                                         clinically standard "referable DR"
                                         (grade >= 2) binarisation and per ICDR
                                         grade (one-vs-rest)
  outputs/phase9/phase9_summary.md       human-readable digest for the manuscript
  outputs/phase9/stats_methods_paragraph.md  drop-in Methods text

Run:
  .venv/Scripts/python.exe scripts/phase9_stats.py            # B=2000, seed=42
  .venv/Scripts/python.exe scripts/phase9_stats.py --boot 5000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import GRADE_NAMES, compute_metrics  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
PHASE6_DIR = PROJECT_DIR / "outputs" / "phase6"
OUT_DIR = PROJECT_DIR / "outputs" / "phase9"

MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
          "retfound_linprobe", "retfound_full"]
DATASETS = ["APTOS-internal", "Messidor-2", "IDRiD"]
EXTERNALS = ["Messidor-2", "IDRiD"]
BASELINE = "baseline_efficientnet_b3"          # reference model for comparisons
METRICS = ["accuracy", "quadratic_weighted_kappa", "macro_f1",
           "balanced_accuracy", "auc_ovr_macro"]
PROB_COLS = [f"prob_{c}" for c in range(5)]
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_preds(model: str, dataset: str) -> pd.DataFrame | None:
    p = PHASE6_DIR / f"{model}__{dataset}_preds.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["id_code"] = df["id_code"].astype(str)
    return df


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def metric_vector(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    m = compute_metrics(y_true, y_pred, y_prob, num_classes=5)
    return {k: m.get(k, np.nan) for k in METRICS}


def boot_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def bootstrap_metrics(df: pd.DataFrame, B: int, rng: np.random.Generator) -> dict:
    """Percentile bootstrap CI for every metric in METRICS on one preds frame."""
    y_true = df["y_true"].to_numpy(int)
    y_pred = df["y_pred"].to_numpy(int)
    y_prob = df[PROB_COLS].to_numpy(float)
    point = metric_vector(y_true, y_pred, y_prob)

    n = len(df)
    draws = {k: np.full(B, np.nan) for k in METRICS}
    for b in range(B):
        idx = boot_indices(n, rng)
        yt, yp, pr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        mv = metric_vector(yt, yp, pr)
        for k in METRICS:
            draws[k][b] = mv[k]

    out = {}
    for k in METRICS:
        d = draws[k][~np.isnan(draws[k])]
        if d.size == 0:
            out[k] = (point[k], np.nan, np.nan, np.nan)
        else:
            lo, hi = np.percentile(d, [2.5, 97.5])
            out[k] = (point[k], lo, hi, d.std(ddof=1))
    return out


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                                   weights="quadratic"))


# --------------------------------------------------------------------------- #
# DeLong  (fast midrank implementation, Sun & Xu 2014)
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray,
                    score_b: np.ndarray) -> tuple[float, float, float, float]:
    """Two-sided DeLong test for a paired difference in AUC.

    Returns (auc_a, auc_b, z, p). y_true must be binary {0,1}.
    """
    y_true = np.asarray(y_true, dtype=int)
    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    preds = np.vstack([score_a, score_b])          # (2, N)
    pos_scores = preds[:, pos]                      # (2, m)
    neg_scores = preds[:, neg]                      # (2, n)

    tx = np.array([_compute_midrank(pos_scores[k]) for k in range(2)])   # (2, m)
    ty = np.array([_compute_midrank(neg_scores[k]) for k in range(2)])   # (2, n)
    tz = np.array([_compute_midrank(np.concatenate([pos_scores[k], neg_scores[k]]))
                   for k in range(2)])             # (2, m+n)

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    s = np.atleast_2d(s)

    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        z = np.nan
        p = np.nan if diff != 0 else 1.0
    else:
        z = diff / np.sqrt(var)
        p = 2.0 * sps.norm.sf(abs(z))
    return (float(aucs[0]), float(aucs[1]), float(z), float(p))


# --------------------------------------------------------------------------- #
# main analyses
# --------------------------------------------------------------------------- #
def run_bootstrap_ci(B: int, seed: int) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            df = load_preds(model, ds)
            if df is None:
                continue
            rng = np.random.default_rng(seed)          # same resamples per cell
            res = bootstrap_metrics(df, B, rng)
            for k in METRICS:
                pt, lo, hi, sd = res[k]
                rows.append({"model": model, "dataset": ds, "metric": k,
                             "n": len(df), "point": pt, "ci_lo": lo,
                             "ci_hi": hi, "boot_sd": sd})
    return pd.DataFrame(rows)


def run_drop_ci(B: int, seed: int) -> pd.DataFrame:
    """Internal->external drop per metric, per model, per external dataset, with
    a bootstrap CI (internal and external sets resampled independently)."""
    rows = []
    for model in MODELS:
        di = load_preds(model, "APTOS-internal")
        if di is None:
            continue
        yi_t, yi_p = di["y_true"].to_numpy(int), di["y_pred"].to_numpy(int)
        pi = di[PROB_COLS].to_numpy(float)
        for ds in EXTERNALS:
            de = load_preds(model, ds)
            if de is None:
                continue
            ye_t, ye_p = de["y_true"].to_numpy(int), de["y_pred"].to_numpy(int)
            pe = de[PROB_COLS].to_numpy(float)

            pt_int = metric_vector(yi_t, yi_p, pi)
            pt_ext = metric_vector(ye_t, ye_p, pe)

            rng = np.random.default_rng(seed)
            ni, ne = len(di), len(de)
            draws = {k: np.full(B, np.nan) for k in METRICS}
            for b in range(B):
                ii, ie = boot_indices(ni, rng), boot_indices(ne, rng)
                if len(np.unique(yi_t[ii])) < 2 or len(np.unique(ye_t[ie])) < 2:
                    continue
                mi = metric_vector(yi_t[ii], yi_p[ii], pi[ii])
                me = metric_vector(ye_t[ie], ye_p[ie], pe[ie])
                for k in METRICS:
                    draws[k][b] = mi[k] - me[k]

            for k in METRICS:
                d = draws[k][~np.isnan(draws[k])]
                lo, hi = (np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
                rows.append({
                    "model": model, "external_dataset": ds, "metric": k,
                    "internal": pt_int[k], "external": pt_ext[k],
                    "drop": pt_int[k] - pt_ext[k],
                    "drop_ci_lo": lo, "drop_ci_hi": hi,
                })
    return pd.DataFrame(rows)


def run_drop_difference_ci(B: int, seed: int) -> pd.DataFrame:
    """Paired bootstrap CI on (drop_model - drop_baseline) in QWK, for each
    external dataset. The same resampled image indices are scored for both
    models (internal and external drawn independently of each other but shared
    across models)."""
    rows = []
    base_int = load_preds(BASELINE, "APTOS-internal")
    for model in MODELS:
        if model == BASELINE:
            continue
        m_int = load_preds(model, "APTOS-internal")
        if base_int is None or m_int is None:
            continue
        bi = base_int.set_index("id_code")
        mi = m_int.set_index("id_code")
        ids_int = [i for i in bi.index if i in mi.index]
        bi, mi = bi.loc[ids_int], mi.loc[ids_int]
        yi_t = bi["y_true"].to_numpy(int)
        bi_p, mi_p = bi["y_pred"].to_numpy(int), mi["y_pred"].to_numpy(int)

        for ds in EXTERNALS:
            base_ext = load_preds(BASELINE, ds)
            m_ext = load_preds(model, ds)
            if base_ext is None or m_ext is None:
                continue
            be = base_ext.set_index("id_code")
            me = m_ext.set_index("id_code")
            ids_ext = [i for i in be.index if i in me.index]
            be, me = be.loc[ids_ext], me.loc[ids_ext]
            ye_t = be["y_true"].to_numpy(int)
            be_p, me_p = be["y_pred"].to_numpy(int), me["y_pred"].to_numpy(int)

            d_base = qwk(yi_t, bi_p) - qwk(ye_t, be_p)
            d_model = qwk(yi_t, mi_p) - qwk(ye_t, me_p)
            point = d_model - d_base

            rng = np.random.default_rng(seed)
            ni, ne = len(ids_int), len(ids_ext)
            draws = np.full(B, np.nan)
            for b in range(B):
                ii, ie = boot_indices(ni, rng), boot_indices(ne, rng)
                if len(np.unique(yi_t[ii])) < 2 or len(np.unique(ye_t[ie])) < 2:
                    continue
                db = qwk(yi_t[ii], bi_p[ii]) - qwk(ye_t[ie], be_p[ie])
                dm = qwk(yi_t[ii], mi_p[ii]) - qwk(ye_t[ie], me_p[ie])
                draws[b] = dm - db
            d = draws[~np.isnan(draws)]
            lo, hi = (np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
            rows.append({
                "model": model, "vs": BASELINE, "external_dataset": ds,
                "qwk_drop_model": d_model, "qwk_drop_baseline": d_base,
                "drop_difference": point, "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": bool((lo > 0) or (hi < 0)),
            })
    return pd.DataFrame(rows)


def run_delong() -> pd.DataFrame:
    rows = []
    targets = {"referable_DR(grade>=2)": ("ge2", None)}
    targets.update({f"grade_{c}_{GRADE_NAMES[c]}": ("cls", c) for c in range(5)})

    for ds in DATASETS:
        base = load_preds(BASELINE, ds)
        if base is None:
            continue
        base = base.set_index("id_code")
        for model in MODELS:
            if model == BASELINE:
                continue
            other = load_preds(model, ds)
            if other is None:
                continue
            other = other.set_index("id_code")
            ids = [i for i in base.index if i in other.index]
            b, o = base.loc[ids], other.loc[ids]
            yt = b["y_true"].to_numpy(int)
            for tname, (kind, c) in targets.items():
                if kind == "ge2":
                    y = (yt >= 2).astype(int)
                    sa = b[["prob_2", "prob_3", "prob_4"]].sum(axis=1).to_numpy()
                    sb = o[["prob_2", "prob_3", "prob_4"]].sum(axis=1).to_numpy()
                else:
                    y = (yt == c).astype(int)
                    sa = b[f"prob_{c}"].to_numpy()
                    sb = o[f"prob_{c}"].to_numpy()
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                auc_b, auc_o, z, p = delong_roc_test(y, sa, sb)
                rows.append({
                    "dataset": ds, "target": tname, "n": len(y),
                    "n_pos": int(y.sum()),
                    "model_baseline": BASELINE, "auc_baseline": auc_b,
                    "model_other": model, "auc_other": auc_o,
                    "auc_diff_other_minus_baseline": (auc_o - auc_b)
                    if np.isfinite(auc_o) and np.isfinite(auc_b) else np.nan,
                    "delong_z": z, "delong_p": p,
                    "significant_at_0_05": bool(np.isfinite(p) and p < ALPHA),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def fmt_ci(pt, lo, hi, prec=3):
    if not np.isfinite(pt):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{pt:.{prec}f}"
    return f"{pt:.{prec}f} ({lo:.{prec}f}–{hi:.{prec}f})"


def write_summary(ci: pd.DataFrame, drop: pd.DataFrame,
                  ddiff: pd.DataFrame, delong: pd.DataFrame,
                  B: int, seed: int) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = ["# Phase 9 - Statistical analysis", "",
         f"_Generated {ts}. Percentile bootstrap, B={B}, seed={seed}, "
         f"two-sided alpha={ALPHA}._", ""]

    L += ["## 1. Headline metrics with 95% CI", "",
          "| Model | Dataset | Accuracy | QWK | Macro-F1 | Bal-Acc | AUC(OvR) |",
          "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for ds in DATASETS:
            sub = ci[(ci.model == model) & (ci.dataset == ds)]
            if sub.empty:
                continue
            g = {r.metric: (r.point, r.ci_lo, r.ci_hi) for r in sub.itertuples()}
            L.append(
                f"| {model} | {ds} | "
                + " | ".join(fmt_ci(*g[k]) for k in METRICS) + " |")

    L += ["", "## 2. Internal->external drop with 95% CI (QWK)", "",
          "| Model | External set | QWK internal | QWK external | Drop (95% CI) |",
          "|---|---|--:|--:|---|"]
    for r in drop[drop.metric == "quadratic_weighted_kappa"].itertuples():
        L.append(f"| {r.model} | {r.external_dataset} | {r.internal:.3f} | "
                 f"{r.external:.3f} | {fmt_ci(r.drop, r.drop_ci_lo, r.drop_ci_hi)} |")

    L += ["", "## 3. Difference in QWK drop vs the EfficientNet-B3 baseline", "",
          "Positive = the model lost *more* ordinal agreement than the baseline "
          "on that external set. CI excluding 0 = a statistically distinguishable "
          "difference in robustness.", "",
          "| Model | External set | Drop (model) | Drop (baseline) | "
          "Difference (95% CI) | Excludes 0 |",
          "|---|---|--:|--:|---|:--:|"]
    for r in ddiff.itertuples():
        L.append(f"| {r.model} | {r.external_dataset} | {r.qwk_drop_model:.3f} | "
                 f"{r.qwk_drop_baseline:.3f} | "
                 f"{fmt_ci(r.drop_difference, r.ci_lo, r.ci_hi)} | "
                 f"{'yes' if r.excludes_zero else 'no'} |")

    L += ["", "## 4. DeLong paired-AUC test (vs EfficientNet-B3 baseline)", "",
          "Referable-DR (ICDR grade >= 2) binarisation; score = P(grade>=2).", "",
          "| Dataset | Model | AUC baseline | AUC model | dAUC | z | p | sig. |",
          "|---|---|--:|--:|--:|--:|--:|:--:|"]
    for r in delong[delong.target == "referable_DR(grade>=2)"].itertuples():
        L.append(f"| {r.dataset} | {r.model_other} | {r.auc_baseline:.3f} | "
                 f"{r.auc_other:.3f} | {r.auc_diff_other_minus_baseline:+.3f} | "
                 f"{r.delong_z:.2f} | {r.delong_p:.3f} | "
                 f"{'*' if r.significant_at_0_05 else 'ns'} |")

    L += ["", "Full per-grade one-vs-rest DeLong results: `outputs/phase9/delong.csv`.", ""]
    (OUT_DIR / "phase9_summary.md").write_text("\n".join(L), encoding="utf-8")


def write_methods_paragraph(B: int, seed: int) -> None:
    txt = f"""\
### Statistical analysis (drop-in Methods text)

All uncertainty estimates were obtained by non-parametric bootstrap resampling of
the relevant test set(s) with replacement ({B} iterations; NumPy PCG64 generator,
seed {seed}), recomputing each metric on every resample and taking the 2.5th and
97.5th percentiles as the 95% confidence interval. For the internal-to-external
performance drop, the internal APTOS test set and the external test set were
resampled independently within each iteration; for the between-model difference
in that drop, the same resampled image indices were scored for both models so the
comparison is paired at the image level. Areas under the one-vs-rest receiver
operating characteristic curve were compared between models on the same test set
with the two-sided DeLong test (fast midrank algorithm of Sun and Xu), applied to
the clinically standard referable-diabetic-retinopathy binarisation (International
Clinical Diabetic Retinopathy grade >= 2; score = summed probability of grades
2-4) and, secondarily, to each grade one-vs-rest. The significance threshold was
alpha = 0.05, two-sided, with no adjustment for multiple comparisons given the
small pre-specified family of confirmatory contrasts (foundation model vs.
EfficientNet-B3 baseline on each external dataset); secondary per-grade tests are
reported as exploratory. Analyses used Python 3.14, NumPy, pandas, SciPy and
scikit-learn; the bootstrap and DeLong code is released with the study
(`scripts/phase9_stats.py`).
"""
    (OUT_DIR / "stats_methods_paragraph.md").write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Phase 9: bootstrap B={args.boot}, seed={args.seed}")

    print("  [1/4] bootstrap CIs for the master table ...")
    ci = run_bootstrap_ci(args.boot, args.seed)
    ci.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)

    print("  [2/4] internal->external drop CIs ...")
    drop = run_drop_ci(args.boot, args.seed)
    drop.to_csv(OUT_DIR / "drop_ci.csv", index=False)

    print("  [3/4] difference-in-drop CIs (paired) ...")
    ddiff = run_drop_difference_ci(args.boot, args.seed)
    ddiff.to_csv(OUT_DIR / "drop_difference_ci.csv", index=False)

    print("  [4/4] DeLong paired-AUC tests ...")
    delong = run_delong()
    delong.to_csv(OUT_DIR / "delong.csv", index=False)

    write_summary(ci, drop, ddiff, delong, args.boot, args.seed)
    write_methods_paragraph(args.boot, args.seed)

    print(f"\nwrote {OUT_DIR}/bootstrap_ci.csv")
    print(f"wrote {OUT_DIR}/drop_ci.csv")
    print(f"wrote {OUT_DIR}/drop_difference_ci.csv")
    print(f"wrote {OUT_DIR}/delong.csv")
    print(f"wrote {OUT_DIR}/phase9_summary.md")
    print(f"wrote {OUT_DIR}/stats_methods_paragraph.md")

    print("\n=== QWK drop difference vs EfficientNet-B3 (95% CI) ===")
    for r in ddiff.itertuples():
        print(f"  {r.model:24s} {r.external_dataset:12s} "
              f"diff={r.drop_difference:+.3f}  CI[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  "
              f"{'EXCLUDES 0' if r.excludes_zero else 'includes 0'}")
    print("\n=== DeLong referable-DR AUC (vs EfficientNet-B3) ===")
    for r in delong[delong.target == 'referable_DR(grade>=2)'].itertuples():
        print(f"  {r.dataset:14s} {r.model_other:24s} "
              f"AUC {r.auc_baseline:.3f} vs {r.auc_other:.3f}  "
              f"p={r.delong_p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
