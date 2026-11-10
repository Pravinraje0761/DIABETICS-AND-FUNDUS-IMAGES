"""
Phase 25 - separating dataset shift from site generalisation.

Reviewer request (round 2): "distinguish dataset shift from genuine independent
clinical-site generalisation. Explain differences in cameras, grading,
populations and disease prevalence."

The internal->external QWK drop mixes several things. This script estimates how
much of it each one explains, using only the frozen predictions (no retraining):

 A. Prevalence (label-distribution) shift.  Importance-reweight every external
    image by w_k = p_APTOS-test(grade k) / p_external(grade k), so the external
    set has the development grade mix. QWK/accuracy under that weighting remove
    the part of the drop due to a different case mix. The remainder is shift in
    p(image | grade) (camera, population, acquisition) plus reference-standard
    shift.
 B. Prevalence-invariant performance. Per-grade recall and referable-DR
    sensitivity / specificity, internal vs external. These do not depend on
    case mix, so a drop here is conditional (image or label) shift.
 C. Reference-standard shift at the mild boundary. Messidor-2 grades are an
    adjudicated 3-specialist consensus; APTOS grades are single-clinician. If
    disagreement about grade 1 drives the drop, merging grades 0 and 1
    (4-level scale) should recover most of it on Messidor-2 but not on IDRiD.
 D. All three combined: prevalence-reweighted QWK on the 4-level scale.

Bootstrap: B=2000 image-level resamples (seed 42); paired across models for
the difference in drop vs EfficientNet-B3.

Outputs (outputs/phase25/):
  shift_decomposition.csv   QWK/acc: raw, reweighted, 4-level, both; drops + CIs
  per_grade_recall.csv      per-grade recall with Wilson CIs, internal/external
  referable_invariant.csv   referable sens/spec (prevalence-invariant) + PPV/NPV
                            re-expressed at APTOS prevalence
  confusion_mild.csv        where true grade-1 images go, per model x dataset
  phase25_summary.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase25"
MODELS = {
    "baseline_efficientnet_b3": "EfficientNet-B3",
    "baseline_resnet50": "ResNet-50",
    "retfound_linprobe": "RETFound linear probe",
    "retfound_full": "RETFound full fine-tune",
}
EXTERNAL = ["Messidor-2", "IDRiD"]
K = 5


def wqwk(y, yhat, w=None, k=K):
    if w is None:
        w = np.ones(len(y))
    O = np.zeros((k, k))
    np.add.at(O, (y, yhat), w)
    n = O.sum()
    W = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / n
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def wacc(y, yhat, w=None):
    if w is None:
        w = np.ones(len(y))
    return float((w * (y == yhat)).sum() / w.sum())


def collapse(g):          # 0,1 -> 0 ; 2->1 ; 3->2 ; 4->3
    return np.maximum(g - 1, 0)


def wilson(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


PREDS = ROOT / "outputs" / "phase6"


def load(m, d):
    df = pd.read_csv(PREDS / f"{m}__{d}_preds.csv")
    return df["y_true"].to_numpy(int), df["y_pred"].to_numpy(int)


def metrics(y, yhat, ref_prev):
    """All variants for one (sub)sample. ref_prev = APTOS grade distribution."""
    ext_prev = np.bincount(y, minlength=K) / len(y)
    w = np.where(ext_prev[y] > 0, ref_prev[y] / np.maximum(ext_prev[y], 1e-12), 0)
    y4, yh4 = collapse(y), collapse(yhat)
    return {
        "qwk_raw": wqwk(y, yhat),
        "qwk_reweighted": wqwk(y, yhat, w),
        "qwk_4level": wqwk(y4, yh4, k=4),
        "qwk_4level_reweighted": wqwk(y4, yh4, w, k=4),
        "acc_raw": wacc(y, yhat),
        "acc_reweighted": wacc(y, yhat, w),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    global PREDS, OUT
    if a.preds_dir:
        PREDS = ROOT / a.preds_dir
    if a.out_dir:
        OUT = ROOT / a.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    yA, _ = load("baseline_efficientnet_b3", "APTOS-internal")
    ref_prev = np.bincount(yA, minlength=K) / len(yA)
    VARS = ["qwk_raw", "qwk_reweighted", "qwk_4level", "qwk_4level_reweighted",
            "acc_raw", "acc_reweighted"]

    cache = {m: {d: load(m, d) for d in ["APTOS-internal"] + EXTERNAL} for m in MODELS}
    # internal metrics (reweighting is the identity internally)
    intern = {m: metrics(*cache[m]["APTOS-internal"], ref_prev) for m in MODELS}
    nA = len(yA)
    bA = rng.integers(0, nA, size=(a.boot, nA))
    int_boot = {m: {v: [] for v in VARS} for m in MODELS}
    for b in bA:
        for m in MODELS:
            y, yh = cache[m]["APTOS-internal"]
            r = metrics(y[b], yh[b], ref_prev)
            for v in VARS:
                int_boot[m][v].append(r[v])

    rows = []
    for d in EXTERNAL:
        n = len(cache["baseline_efficientnet_b3"][d][0])
        bE = rng.integers(0, n, size=(a.boot, n))
        ext_boot = {m: {v: [] for v in VARS} for m in MODELS}
        for b in bE:
            for m in MODELS:
                y, yh = cache[m][d]
                r = metrics(y[b], yh[b], ref_prev)
                for v in VARS:
                    ext_boot[m][v].append(r[v])
        for m, ml in MODELS.items():
            y, yh = cache[m][d]
            ext = metrics(y, yh, ref_prev)
            for v in VARS:
                drop = intern[m][v] - ext[v]
                db = np.array(int_boot[m][v]) - np.array(ext_boot[m][v])
                base = "baseline_efficientnet_b3"
                dbb = np.array(int_boot[base][v]) - np.array(ext_boot[base][v])
                dd = db - dbb
                rows.append({
                    "model": m, "model_label": ml, "external": d, "metric": v,
                    "internal": intern[m][v], "external_value": ext[v],
                    "drop": drop,
                    "drop_lo": float(np.nanpercentile(db, 2.5)),
                    "drop_hi": float(np.nanpercentile(db, 97.5)),
                    "ddrop_vs_effb3": drop - (intern[base][v] - metrics(*cache[base][d], ref_prev)[v]) if m != base else np.nan,
                    "ddrop_lo": float(np.nanpercentile(dd, 2.5)) if m != base else np.nan,
                    "ddrop_hi": float(np.nanpercentile(dd, 97.5)) if m != base else np.nan,
                })
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT / "shift_decomposition.csv", index=False)

    # ---- B. per-grade recall + referable invariant metrics
    pg, rv, cm = [], [], []
    prevA = (yA >= 2).mean()
    for m, ml in MODELS.items():
        for d in ["APTOS-internal"] + EXTERNAL:
            y, yh = cache[m][d]
            for k in range(K):
                sel = y == k
                x = int((yh[sel] == k).sum())
                lo, hi = wilson(x, int(sel.sum()))
                pg.append({"model": m, "model_label": ml, "dataset": d, "grade": k,
                           "n": int(sel.sum()), "recall": x / sel.sum() if sel.sum() else np.nan,
                           "lo": lo, "hi": hi,
                           "mean_abs_grade_error": float(np.abs(yh[sel] - k).mean()) if sel.sum() else np.nan})
            t, p = y >= 2, yh >= 2
            se, sp = (p & t).sum() / t.sum(), (~p & ~t).sum() / (~t).sum()
            sl, sh = wilson((p & t).sum(), t.sum())
            pl, ph = wilson((~p & ~t).sum(), (~t).sum())
            def ppv(prev):
                return se * prev / (se * prev + (1 - sp) * (1 - prev))
            def npv(prev):
                return sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
            rv.append({"model": m, "model_label": ml, "dataset": d,
                       "prevalence": t.mean(), "sens": se, "sens_lo": sl, "sens_hi": sh,
                       "spec": sp, "spec_lo": pl, "spec_hi": ph,
                       "ppv_own_prev": ppv(t.mean()), "npv_own_prev": npv(t.mean()),
                       "ppv_at_aptos_prev": ppv(prevA), "npv_at_aptos_prev": npv(prevA)})
            sel = y == 1
            cnt = np.bincount(yh[sel], minlength=K)
            cm.append({"model": m, "model_label": ml, "dataset": d, "n_true_mild": int(sel.sum()),
                       **{f"pred_{k}": int(cnt[k]) for k in range(K)}})
    pd.DataFrame(pg).to_csv(OUT / "per_grade_recall.csv", index=False)
    pd.DataFrame(rv).to_csv(OUT / "referable_invariant.csv", index=False)
    pd.DataFrame(cm).to_csv(OUT / "confusion_mild.csv", index=False)

    prev = {d: np.bincount(cache["baseline_efficientnet_b3"][d][0], minlength=K) /
            len(cache["baseline_efficientnet_b3"][d][0]) for d in ["APTOS-internal"] + EXTERNAL}
    write_summary(dec, pd.DataFrame(pg), pd.DataFrame(rv), pd.DataFrame(cm), prev, a)
    return 0


def write_summary(dec, pg, rv, cm, prev, a):
    L = ["# Phase 25 - dataset shift vs. site generalisation\n",
         f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}; B={a.boot}, seed={a.seed}; predictions from `{PREDS.relative_to(ROOT)}` (frozen checkpoints)._\n",
         "## Grade mix (%)\n", "| Dataset | 0 | 1 | 2 | 3 | 4 | referable |", "|---|--:|--:|--:|--:|--:|--:|"]
    for d, p in prev.items():
        L.append(f"| {d} | " + " | ".join(f"{100*x:.1f}" for x in p) + f" | {100*p[2:].sum():.1f} |")
    L += ["", "## A/C/D. QWK drop (internal − external) under each correction, 95% CI\n",
          "Reweighted = external images importance-weighted to the APTOS-test grade mix "
          "(removes prevalence shift). 4-level = grades 0 and 1 merged (removes the "
          "mild-boundary reference-standard disagreement).\n",
          "| Model | External | raw | reweighted | 4-level | 4-level + reweighted |", "|---|---|---|---|---|---|"]
    for m, ml in MODELS.items():
        for d in EXTERNAL:
            r = dec[(dec.model == m) & (dec.external == d)].set_index("metric")
            cell = lambda v: f"{r.loc[v,'drop']:.3f} [{r.loc[v,'drop_lo']:.3f}, {r.loc[v,'drop_hi']:.3f}]"
            L.append(f"| {ml} | {d} | " + " | ".join(cell(v) for v in
                     ["qwk_raw", "qwk_reweighted", "qwk_4level", "qwk_4level_reweighted"]) + " |")
    L += ["", "### Share of the raw QWK drop removed by each correction\n",
          "| Model | External | by reweighting | by 4-level | by both | residual drop |", "|---|---|--:|--:|--:|--:|"]
    for m, ml in MODELS.items():
        for d in EXTERNAL:
            r = dec[(dec.model == m) & (dec.external == d)].set_index("metric")["drop"]
            raw = r["qwk_raw"]
            f = lambda v: f"{100*(raw - r[v])/raw:.0f}%"
            L.append(f"| {ml} | {d} | {f('qwk_reweighted')} | {f('qwk_4level')} | {f('qwk_4level_reweighted')} | {r['qwk_4level_reweighted']:.3f} |")
    L += ["", "### Does the primary comparison survive the corrections? Difference in drop vs EfficientNet-B3\n",
          "| Model | External | raw | reweighted | 4-level + reweighted |", "|---|---|---|---|---|"]
    for m, ml in MODELS.items():
        if m == "baseline_efficientnet_b3":
            continue
        for d in EXTERNAL:
            r = dec[(dec.model == m) & (dec.external == d)].set_index("metric")
            cell = lambda v: f"{r.loc[v,'ddrop_vs_effb3']:+.3f} [{r.loc[v,'ddrop_lo']:+.3f}, {r.loc[v,'ddrop_hi']:+.3f}]"
            L.append(f"| {ml} | {d} | {cell('qwk_raw')} | {cell('qwk_reweighted')} | {cell('qwk_4level_reweighted')} |")
    L += ["", "## B. Prevalence-invariant performance\n", "### Per-grade recall\n",
          "| Model | Dataset | 0 | 1 | 2 | 3 | 4 |", "|---|---|--:|--:|--:|--:|--:|"]
    for m, ml in MODELS.items():
        for d in ["APTOS-internal"] + EXTERNAL:
            r = pg[(pg.model == m) & (pg.dataset == d)].sort_values("grade")
            L.append(f"| {ml} | {d} | " + " | ".join(f"{x:.2f}" for x in r.recall) + " |")
    L += ["", "### Referable DR: sensitivity / specificity (prevalence-invariant), and PPV at own vs APTOS prevalence\n",
          "| Model | Dataset | prev. | sens [CI] | spec [CI] | PPV own | PPV @APTOS prev |", "|---|---|--:|---|---|--:|--:|"]
    for _, r in rv.iterrows():
        L.append(f"| {r.model_label} | {r.dataset} | {r.prevalence:.2f} | {r.sens:.2f} [{r.sens_lo:.2f}, {r.sens_hi:.2f}] | "
                 f"{r.spec:.2f} [{r.spec_lo:.2f}, {r.spec_hi:.2f}] | {r.ppv_own_prev:.2f} | {r.ppv_at_aptos_prev:.2f} |")
    L += ["", "### Where reference-standard grade-1 (mild) images are predicted\n",
          "| Model | Dataset | n | →0 | →1 | →2 | →3 | →4 |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for _, r in cm.iterrows():
        L.append(f"| {r.model_label} | {r.dataset} | {r.n_true_mild} | " +
                 " | ".join(str(r[f'pred_{k}']) for k in range(K)) + " |")
    (OUT / "phase25_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
