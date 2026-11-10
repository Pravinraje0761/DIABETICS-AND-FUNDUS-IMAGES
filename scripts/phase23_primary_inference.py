"""
Phase 23 - primary-outcome inference package (reviewer revision, 2026-10).

QWK is the single PRIMARY outcome. This script makes the primary inference
explicit and adds the checks the reviewer asked for. Everything is computed
from the Phase 6 per-image prediction CSVs; no re-inference, no torch, no
sklearn (numpy + pandas only), so it runs anywhere in seconds.

Outputs (outputs/phase23/):
  qwk_pairwise_diff.csv     paired bootstrap CI for QWK(model) - QWK(EffNet-B3)
                            on each dataset (internal, Messidor-2, IDRiD)
  qwk_drop.csv              internal->external QWK drop per model, with CI
  confirmatory_family.csv   PRIMARY CONFIRMATORY FAMILY: difference in QWK drop,
                            {RETFound full FT, RETFound linear probe} vs
                            EfficientNet-B3, on {Messidor-2, IDRiD} = 4 contrasts.
                            95% CI, Bonferroni-adjusted 98.75% CI, bootstrap
                            two-sided p, Holm-adjusted p.
  secondary_family.csv      same for ResNet-50 vs EfficientNet-B3 (secondary,
                            not part of the confirmatory family).
  cluster_sensitivity.csv   worst-case non-independence check: m-out-of-n
                            bootstrap with m = n/2 on every set (equivalent to
                            assuming every image has a perfectly correlated
                            partner image, e.g. the fellow eye of the same
                            patient; Messidor-2 is 2 images x 872 exams).
  idrid_reweighted.csv      IDRiD results re-weighted by grade so the 455-image
                            mirror matches the grade mix of the official 516-image
                            Disease-Grading release (missingness sensitivity).
  phase23_summary.md        digest.

Run:
  python scripts/phase23_primary_inference.py            # B=2000, seed=42
  python scripts/phase23_primary_inference.py --preds-dir outputs/phase8/mc --tag mc
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
          "retfound_linprobe", "retfound_full"]
LABEL = {"baseline_efficientnet_b3": "EfficientNet-B3",
         "baseline_resnet50": "ResNet-50",
         "retfound_linprobe": "RETFound linear probe",
         "retfound_full": "RETFound full fine-tune"}
BASELINE = "baseline_efficientnet_b3"
CONFIRMATORY = ["retfound_full", "retfound_linprobe"]
SECONDARY = ["baseline_resnet50"]
EXTERNALS = ["Messidor-2", "IDRiD"]
K = 5
# Official IDRiD Disease-Grading release (413 train + 103 test = 516), per grade
# 0..4. Train 134/20/136/74/49, test 34/5/32/19/13 (IEEE DataPort release, as
# mirrored with identical split sizes on Hugging Face amin-nejad/idrid-disease-grading;
# label statistics retrieved 2026-10-21).
IDRID_OFFICIAL = np.array([168, 25, 168, 93, 62], float)

_W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2


def qwk(y, p, w=None) -> float:
    """Quadratic-weighted kappa from a (optionally sample-weighted) confusion
    matrix. Identical to sklearn cohen_kappa_score(weights='quadratic') when
    w is None."""
    if w is None:
        w = np.ones(len(y))
    O = np.bincount(y * K + p, weights=w, minlength=K * K).reshape(K, K)
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    den = (_W * E).sum()
    return float(1.0 - (_W * O).sum() / den) if den > 0 else np.nan


def load(preds_dir: Path, model: str, ds: str) -> pd.DataFrame:
    f = preds_dir / f"{model}__{ds}_preds.csv"
    return pd.read_csv(f).set_index("id_code")


def align(a: pd.DataFrame, b: pd.DataFrame):
    ids = [i for i in a.index if i in b.index]
    a, b = a.loc[ids], b.loc[ids]
    assert (a["y_true"].to_numpy() == b["y_true"].to_numpy()).all()
    return (a["y_true"].to_numpy(int), a["y_pred"].to_numpy(int),
            b["y_pred"].to_numpy(int))


def pct(d, a=0.05):
    d = d[~np.isnan(d)]
    return np.percentile(d, [100 * a / 2, 100 * (1 - a / 2)])


def boot_p(d) -> float:
    """Two-sided bootstrap p for H0: parameter = 0 (percentile-inversion)."""
    d = d[~np.isnan(d)]
    B = d.size
    lo = (np.sum(d <= 0) + 1) / (B + 1)
    hi = (np.sum(d >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def ddrop_draws(yi, bi, mi, ye, be, me, B, seed, m_frac=1.0, wi=None, we=None):
    """Bootstrap draws of drop_model - drop_baseline. Same resampling scheme as
    phase9_stats.run_drop_difference_ci (internal and external resampled
    independently, indices shared across the two models); m_frac<1 gives an
    m-out-of-n bootstrap (variance inflation by 1/m_frac)."""
    rng = np.random.default_rng(seed)
    ni, ne = len(yi), len(ye)
    mi_n, me_n = max(2, int(round(ni * m_frac))), max(2, int(round(ne * m_frac)))
    out = np.full(B, np.nan)
    for b in range(B):
        ii = rng.integers(0, ni, size=mi_n)
        ie = rng.integers(0, ne, size=me_n)
        if len(np.unique(yi[ii])) < 2 or len(np.unique(ye[ie])) < 2:
            continue
        wii = None if wi is None else wi[ii]
        wee = None if we is None else we[ie]
        db = qwk(yi[ii], bi[ii], wii) - qwk(ye[ie], be[ie], wee)
        dm = qwk(yi[ii], mi[ii], wii) - qwk(ye[ie], me[ie], wee)
        out[b] = dm - db
    # m-out-of-n: percentiles of the statistic on m=n*m_frac images are used
    # directly, i.e. the CI is that of an estimator with half the effective n.
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preds-dir", default="outputs/phase6")
    ap.add_argument("--tag", default="singlepass")
    ap.add_argument("--out-dir", default="outputs/phase23")
    args = ap.parse_args()
    B, seed = args.boot, args.seed
    pdir = (PROJECT_DIR / args.preds_dir).resolve()
    odir = (PROJECT_DIR / args.out_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.tag in ("", "singlepass") else f"_{args.tag}"

    P = {(m, d): load(pdir, m, d) for m in MODELS
         for d in ["APTOS-internal"] + EXTERNALS}

    # 1. paired QWK differences per dataset ---------------------------------
    rows = []
    for m in MODELS:
        if m == BASELINE:
            continue
        for ds in ["APTOS-internal"] + EXTERNALS:
            y, pb, pm = align(P[(BASELINE, ds)], P[(m, ds)])
            pt = qwk(y, pm) - qwk(y, pb)
            rng = np.random.default_rng(seed)
            d = np.full(B, np.nan)
            for b in range(B):
                i = rng.integers(0, len(y), size=len(y))
                d[b] = qwk(y[i], pm[i]) - qwk(y[i], pb[i])
            lo, hi = pct(d)
            rows.append({"model": m, "label": LABEL[m], "dataset": ds,
                         "qwk_model": qwk(y, pm), "qwk_baseline": qwk(y, pb),
                         "qwk_diff": pt, "ci_lo": lo, "ci_hi": hi,
                         "p_boot": boot_p(d), "role": "secondary (descriptive)"})
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(odir / f"qwk_pairwise_diff{sfx}.csv", index=False)

    # 2. drops with CI -------------------------------------------------------
    rows = []
    for m in MODELS:
        di = P[(m, "APTOS-internal")]
        yi, pi = di["y_true"].to_numpy(int), di["y_pred"].to_numpy(int)
        for ds in EXTERNALS:
            de = P[(m, ds)]
            ye, pe = de["y_true"].to_numpy(int), de["y_pred"].to_numpy(int)
            rng = np.random.default_rng(seed)
            d = np.full(B, np.nan)
            for b in range(B):
                ii = rng.integers(0, len(yi), size=len(yi))
                ie = rng.integers(0, len(ye), size=len(ye))
                d[b] = qwk(yi[ii], pi[ii]) - qwk(ye[ie], pe[ie])
            lo, hi = pct(d)
            rows.append({"model": m, "label": LABEL[m], "external": ds,
                         "qwk_internal": qwk(yi, pi), "qwk_external": qwk(ye, pe),
                         "drop": qwk(yi, pi) - qwk(ye, pe), "ci_lo": lo, "ci_hi": hi})
    drops = pd.DataFrame(rows)
    drops.to_csv(odir / f"qwk_drop{sfx}.csv", index=False)

    # 3. confirmatory + secondary families (difference in drop) --------------
    def family(models, m_frac=1.0):
        rows = []
        for m in models:
            yi, bi, mi = align(P[(BASELINE, "APTOS-internal")], P[(m, "APTOS-internal")])
            for ds in EXTERNALS:
                ye, be, me = align(P[(BASELINE, ds)], P[(m, ds)])
                pt = (qwk(yi, mi) - qwk(ye, me)) - (qwk(yi, bi) - qwk(ye, be))
                d = ddrop_draws(yi, bi, mi, ye, be, me, B, seed, m_frac)
                lo, hi = pct(d)
                blo, bhi = pct(d, 0.05 / 4)
                rows.append({"model": m, "label": LABEL[m], "external": ds,
                             "delta_drop": pt, "ci95_lo": lo, "ci95_hi": hi,
                             "ci_bonf_lo": blo, "ci_bonf_hi": bhi,
                             "p_boot": boot_p(d)})
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p_boot"]) if len(df) else []
        return df

    conf = family(CONFIRMATORY)
    conf["family"] = "confirmatory (primary)"
    conf.to_csv(odir / f"confirmatory_family{sfx}.csv", index=False)
    sec = family(SECONDARY)
    sec["family"] = "secondary"
    sec.to_csv(odir / f"secondary_family{sfx}.csv", index=False)

    # 4. worst-case non-independence (m = n/2) -------------------------------
    clus = family(CONFIRMATORY + SECONDARY, m_frac=0.5)
    clus["scenario"] = "m-out-of-n bootstrap, m=n/2 (worst-case paired images)"
    clus.to_csv(odir / f"cluster_sensitivity{sfx}.csv", index=False)

    # 5. IDRiD grade reweighting to the official 516-image mix --------------
    rows = []
    for m in MODELS:
        yi, bi, mi = align(P[(BASELINE, "APTOS-internal")], P[(m, "APTOS-internal")])
        ye, be, me = align(P[(BASELINE, "IDRiD")], P[(m, "IDRiD")])
        mirror = np.bincount(ye, minlength=K).astype(float)
        wg = (IDRID_OFFICIAL / IDRID_OFFICIAL.sum()) / (mirror / mirror.sum())
        we = wg[ye]
        q_raw, q_w = qwk(ye, me), qwk(ye, me, we)
        row = {"model": m, "label": LABEL[m],
               "qwk_idrid_mirror": q_raw, "qwk_idrid_reweighted": q_w,
               "drop_mirror": qwk(yi, mi) - q_raw,
               "drop_reweighted": qwk(yi, mi) - q_w}
        if m != BASELINE:
            d = ddrop_draws(yi, bi, mi, ye, be, me, B, seed, we=we)
            lo, hi = pct(d)
            row.update({"delta_drop_reweighted":
                        (qwk(yi, mi) - q_w) - (qwk(yi, bi) - qwk(ye, be, we)),
                        "ci95_lo": lo, "ci95_hi": hi})
        rows.append(row)
    rew = pd.DataFrame(rows)
    rew.to_csv(odir / f"idrid_reweighted{sfx}.csv", index=False)

    # 6. summary --------------------------------------------------------------
    f3 = lambda x: f"{x:.3f}"
    L = [f"# Phase 23 - primary-outcome inference ({args.tag})", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}; "
         f"B={B}, seed={seed}, preds={args.preds_dir}", "",
         "## Confirmatory family (primary): difference in QWK drop vs EfficientNet-B3", "",
         "| Model | External | dDrop | 95% CI | Bonferroni 98.75% CI | p (boot) | p (Holm) |",
         "|---|---|---|---|---|---|---|"]
    for r in conf.itertuples():
        L.append(f"| {r.label} | {r.external} | {r.delta_drop:+.3f} | "
                 f"[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | [{r.ci_bonf_lo:.3f}, {r.ci_bonf_hi:.3f}] | "
                 f"{r.p_boot:.4f} | {r.p_holm:.4f} |")
    L += ["", "## Secondary: ResNet-50 vs EfficientNet-B3", "",
          "| External | dDrop | 95% CI | p (boot) |", "|---|---|---|---|"]
    for r in sec.itertuples():
        L.append(f"| {r.external} | {r.delta_drop:+.3f} | [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {r.p_boot:.4f} |")
    L += ["", "## Paired QWK difference (model - EfficientNet-B3)", "",
          "| Model | Dataset | QWK model | QWK base | diff | 95% CI |", "|---|---|---|---|---|---|"]
    for r in pairwise.itertuples():
        L.append(f"| {r.label} | {r.dataset} | {f3(r.qwk_model)} | {f3(r.qwk_baseline)} | "
                 f"{r.qwk_diff:+.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |")
    L += ["", "## Worst-case non-independence (m = n/2)", "",
          "| Model | External | dDrop | 95% CI | excludes 0 |", "|---|---|---|---|---|"]
    for r in clus.itertuples():
        L.append(f"| {r.label} | {r.external} | {r.delta_drop:+.3f} | "
                 f"[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {(r.ci95_lo > 0) or (r.ci95_hi < 0)} |")
    L += ["", "## IDRiD reweighted to the official 516-image grade mix", "",
          "| Model | QWK mirror | QWK reweighted | dDrop reweighted (95% CI) |", "|---|---|---|---|"]
    for r in rew.itertuples():
        dd = ("---" if r.model == BASELINE else
              f"{r.delta_drop_reweighted:+.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]")
        L.append(f"| {r.label} | {f3(r.qwk_idrid_mirror)} | {f3(r.qwk_idrid_reweighted)} | {dd} |")
    (odir / f"phase23_summary{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
