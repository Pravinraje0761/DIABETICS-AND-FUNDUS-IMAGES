"""
Phase 24 - uncertainty validity: does uncertainty identify errors, and does
selective prediction (referring the most uncertain cases) improve performance?
Every uncertainty score is compared against simple confidence baselines.

Reviewer request (round 2): "Demonstrate whether uncertainty identifies errors
and whether selective prediction improves performance at different referral
rates. Compare against simple confidence baselines."

Inputs: per-image prediction CSVs (id_code, y_true, y_pred, prob_0..prob_4,
entropy) from
  * outputs/phase6/                      single forward pass (primary)
  * outputs/phase8/mc_lp_deterministic/  T=20 stochastic passes (mean softmax)

Uncertainty scores (higher = less trustworthy):
  msp        1 - max softmax probability           [simple confidence baseline]
  margin     -(p_top1 - p_top2)                    [simple confidence baseline]
  entropy    predictive entropy, single pass        [score used in the paper]
  mc_entropy predictive entropy of the T=20 mean softmax
  grade_sd   SD of the predicted grade distribution, sqrt(sum_k p_k (k-E[k])^2)
             (ordinal-aware: ignores mass spread over adjacent grades less)
  random     random ranking                         [no-information baseline]

Error definitions:
  grade error      argmax grade != reference grade      (primary)
  referable error  (pred >= 2) != (reference >= 2)      (clinically relevant)

Outputs (outputs/phase24/):
  error_detection.csv       AUROC / AUPRC per model x dataset x score x error type,
                            95% bootstrap CI, paired AUROC difference vs MSP
  selective_prediction.csv  accuracy, QWK, referable sens/spec, error capture at
                            referral rates 0-50%, with 95% CIs for gains
  aurc.csv                  area under the risk-coverage curve (+ E-AURC) per score
  risk_coverage_curves.csv  full curves (1% steps)
  fig_risk_coverage.png     risk-coverage curves, external sets
  fig_referral_gain.png     accuracy / QWK gain vs referral rate
  phase24_summary.md        digest

Bootstrap: B=2000 image-level resamples, seed 42, paired across scores
(the same resample indices for every score of a model x dataset).

Run:  python scripts/phase24_uncertainty_validity.py [--boot 2000]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase24"

MODELS = {
    "baseline_efficientnet_b3": "EfficientNet-B3",
    "baseline_resnet50": "ResNet-50",
    "retfound_linprobe": "RETFound linear probe",
    "retfound_full": "RETFound full fine-tune",
}
DATASETS = ["APTOS-internal", "Messidor-2", "IDRiD"]
SCORES = ["msp", "margin", "entropy", "mc_entropy", "grade_sd", "random"]
SCORE_LABEL = {"msp": "1 - max prob (MSP)", "margin": "top-2 margin",
               "entropy": "entropy (1 pass)", "mc_entropy": "entropy (T=20)",
               "grade_sd": "grade SD", "random": "random"}
REFERRAL = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
K = 5


# ----------------------------------------------------------------- metrics
def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    n1 = pos.sum()
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    # average ranks over ties
    _, idx_start, counts = np.unique(s, return_index=True, return_counts=True)
    avg = np.repeat(idx_start + (counts + 1) / 2.0, counts)
    ranks[order] = avg
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score: np.ndarray, pos: np.ndarray) -> float:
    """Average precision (step-wise), positives = errors."""
    n1 = pos.sum()
    if n1 == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    p = pos[order]
    tp = np.cumsum(p)
    prec = tp / np.arange(1, len(p) + 1)
    return float((prec * p).sum() / n1)


def qwk(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    O = np.zeros((K, K))
    np.add.at(O, (y, yhat), 1)
    n = O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / n
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def sens_spec(y: np.ndarray, yhat: np.ndarray):
    t = y >= 2
    p = yhat >= 2
    sens = (p & t).sum() / t.sum() if t.sum() else np.nan
    spec = (~p & ~t).sum() / (~t).sum() if (~t).sum() else np.nan
    return float(sens), float(spec)


# ----------------------------------------------------------------- scores
def compute_scores(df: pd.DataFrame, mc: pd.DataFrame | None, rng) -> dict:
    P = df[[f"prob_{k}" for k in range(K)]].to_numpy(float)
    Ps = np.sort(P, axis=1)
    E = (P * np.arange(K)).sum(1)
    out = {
        "msp": 1 - Ps[:, -1],
        "margin": -(Ps[:, -1] - Ps[:, -2]),
        "entropy": df["entropy"].to_numpy(float),
        "grade_sd": np.sqrt((P * (np.arange(K)[None, :] - E[:, None]) ** 2).sum(1)),
        "random": rng.random(len(df)),
    }
    if mc is not None:
        m = mc.set_index("id_code").reindex(df["id_code"])
        Pm = m[[f"prob_{k}" for k in range(K)]].to_numpy(float)
        out["mc_entropy"] = -(Pm * np.log(np.clip(Pm, 1e-12, 1))).sum(1)
    else:
        out["mc_entropy"] = np.full(len(df), np.nan)
    return out


# ----------------------------------------------------------------- selective
def selective_at(order, y, yhat, rate):
    n = len(y)
    k = n - int(round(rate * n))
    keep = order[:k]
    ref = order[k:]
    err = y != yhat
    tot = err.sum()
    sens, spec = sens_spec(y[keep], yhat[keep])
    return {
        "n_kept": k,
        "accuracy": float((~err[keep]).mean()),
        "qwk": qwk(y[keep], yhat[keep]),
        "ref_sens": sens, "ref_spec": spec,
        "error_capture": float(err[ref].sum() / tot) if tot else np.nan,
        "ref_err_rate_kept": float(((y[keep] >= 2) != (yhat[keep] >= 2)).mean()),
        # composition of what is kept vs referred (selective prediction can raise
        # accuracy simply by referring the diseased, harder cases)
        "referable_prev_kept": float((y[keep] >= 2).mean()),
        "referable_prev_referred": float((y[ref] >= 2).mean()) if len(ref) else np.nan,
        # human-in-the-loop view: referred cases go to a grader; a referable case is
        # missed only if it is kept AND called non-referable by the model
        "missed_referable_frac": float(((y[keep] >= 2) & (yhat[keep] < 2)).sum() / max(1, (y >= 2).sum())),
    }


def aurc(score, err):
    """Area under risk(=error rate)-coverage curve, accepting most confident first."""
    order = np.argsort(score, kind="mergesort")
    e = err[order].astype(float)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    a = risk.mean()
    r = e.mean()
    opt = r + (1 - r) * np.log(1 - r) if 0 < r < 1 else 0.0   # optimal AURC
    return float(a), float(a - opt)


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sp-dir", default="outputs/phase6")
    ap.add_argument("--mc-dir", default="outputs/phase8/mc_lp_deterministic")
    ap.add_argument("--model", default=None,
                    help="compute one model only and save to outputs/phase24/parts/ "
                         "(lets the four models run as separate shorter jobs)")
    ap.add_argument("--merge", action="store_true",
                    help="merge outputs/phase24/parts/ and write tables/figures")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = OUT / "parts"
    names = ["error_detection", "selective_prediction", "aurc", "risk_coverage_curves"]
    if a.merge:
        dfs = [pd.concat([pd.read_csv(parts / f"{n}__{m}.csv") for m in MODELS])
               for n in names]
        for n, df in zip(names, dfs):
            df.to_csv(OUT / f"{n}.csv", index=False)
        make_figures(dfs[3], dfs[1])
        write_summary(dfs[0], dfs[1], dfs[2], a)
        return 0

    det_rows, sel_rows, aurc_rows, curve_rows = [], [], [], []
    for m, mlabel in MODELS.items():
        if a.model and m != a.model:
            continue
        rng_master = np.random.default_rng(a.seed + list(MODELS).index(m))
        for d in DATASETS:
            sp = pd.read_csv(ROOT / a.sp_dir / f"{m}__{d}_preds.csv")
            mcp = ROOT / a.mc_dir / f"{m}__{d}_preds.csv"
            mc = pd.read_csv(mcp) if mcp.exists() else None
            y = sp["y_true"].to_numpy(int)
            yhat = sp["y_pred"].to_numpy(int)
            n = len(y)
            rng = np.random.default_rng(a.seed)          # same random score per cell
            S = compute_scores(sp, mc, rng)
            errs = {"grade": y != yhat, "referable": (y >= 2) != (yhat >= 2)}
            boot_idx = rng_master.integers(0, n, size=(a.boot, n))

            # ---- error detection
            for et, err in errs.items():
                base_b = np.array([auroc(S["msp"][b], err[b]) for b in boot_idx])
                for s in SCORES:
                    sc = S[s]
                    if np.isnan(sc).all():
                        continue
                    au = auroc(sc, err)
                    ap_ = auprc(sc, err)
                    bb = np.array([auroc(sc[b], err[b]) for b in boot_idx])
                    diff = bb - base_b
                    d0 = au - auroc(S["msp"], err)
                    p_two = 2 * min((diff <= 0).mean(), (diff >= 0).mean()) if s != "msp" else np.nan
                    det_rows.append({
                        "model": m, "model_label": mlabel, "dataset": d,
                        "error_type": et, "score": s, "n": n,
                        "n_errors": int(err.sum()), "error_rate": float(err.mean()),
                        "auroc": au,
                        "auroc_lo": float(np.nanpercentile(bb, 2.5)),
                        "auroc_hi": float(np.nanpercentile(bb, 97.5)),
                        "auprc": ap_, "auprc_chance": float(err.mean()),
                        "auroc_minus_msp": d0 if s != "msp" else np.nan,
                        "diff_lo": float(np.nanpercentile(diff, 2.5)) if s != "msp" else np.nan,
                        "diff_hi": float(np.nanpercentile(diff, 97.5)) if s != "msp" else np.nan,
                        "p_vs_msp": min(1.0, p_two) if s != "msp" else np.nan,
                    })

            # ---- AURC + curves
            err = errs["grade"]
            for s in SCORES:
                sc = S[s]
                if np.isnan(sc).all():
                    continue
                A, EA = aurc(sc, err)
                bb = np.array([aurc(sc[b], err[b])[0] for b in boot_idx[: min(a.boot, 1000)]])
                aurc_rows.append({"model": m, "model_label": mlabel, "dataset": d,
                                  "score": s, "aurc": A, "e_aurc": EA,
                                  "aurc_lo": float(np.percentile(bb, 2.5)),
                                  "aurc_hi": float(np.percentile(bb, 97.5)),
                                  "risk_full_coverage": float(err.mean())})
                order = np.argsort(sc, kind="mergesort")
                for cov in np.round(np.arange(1.0, 0.0, -0.01), 2):
                    k = max(1, int(round(cov * n)))
                    keep = order[:k]
                    curve_rows.append({"model": m, "dataset": d, "score": s,
                                       "coverage": cov,
                                       "risk": float(err[keep].mean()),
                                       "qwk": qwk(y[keep], yhat[keep])})

            # ---- selective prediction at referral rates, with bootstrap CIs on gains
            nb = min(a.boot, 1000)
            for s in SCORES:
                sc = S[s]
                if np.isnan(sc).all():
                    continue
                order = np.argsort(sc, kind="mergesort")
                full = selective_at(order, y, yhat, 0.0)
                for rate in REFERRAL:
                    r = selective_at(order, y, yhat, rate)
                    # bootstrap gains (vs no referral, vs MSP at same rate)
                    g_acc, g_qwk, v_acc, v_qwk = [], [], [], []
                    for b in boot_idx[:nb]:
                        yb, yhb, sb, mb = y[b], yhat[b], sc[b], S["msp"][b]
                        ob = np.argsort(sb, kind="mergesort")
                        omb = np.argsort(mb, kind="mergesort")
                        f = selective_at(ob, yb, yhb, 0.0)
                        rr = selective_at(ob, yb, yhb, rate)
                        rm = selective_at(omb, yb, yhb, rate)
                        g_acc.append(rr["accuracy"] - f["accuracy"])
                        g_qwk.append(rr["qwk"] - f["qwk"])
                        v_acc.append(rr["accuracy"] - rm["accuracy"])
                        v_qwk.append(rr["qwk"] - rm["qwk"])
                    pct = lambda v, q: float(np.nanpercentile(v, q))
                    sel_rows.append({
                        "model": m, "model_label": mlabel, "dataset": d, "score": s,
                        "referral_rate": rate, **r,
                        "acc_gain": r["accuracy"] - full["accuracy"],
                        "acc_gain_lo": pct(g_acc, 2.5), "acc_gain_hi": pct(g_acc, 97.5),
                        "qwk_gain": r["qwk"] - full["qwk"],
                        "qwk_gain_lo": pct(g_qwk, 2.5), "qwk_gain_hi": pct(g_qwk, 97.5),
                        "acc_vs_msp_lo": pct(v_acc, 2.5) if s != "msp" else np.nan,
                        "acc_vs_msp_hi": pct(v_acc, 97.5) if s != "msp" else np.nan,
                        "qwk_vs_msp_lo": pct(v_qwk, 2.5) if s != "msp" else np.nan,
                        "qwk_vs_msp_hi": pct(v_qwk, 97.5) if s != "msp" else np.nan,
                    })
            print(f"done {m} {d}", flush=True)

    det = pd.DataFrame(det_rows)
    sel = pd.DataFrame(sel_rows)
    au = pd.DataFrame(aurc_rows)
    cur = pd.DataFrame(curve_rows)
    if a.model:
        parts.mkdir(exist_ok=True)
        for n, df in zip(names, [det, sel, au, cur]):
            df.to_csv(parts / f"{n}__{a.model}.csv", index=False)
        return 0
    det.to_csv(OUT / "error_detection.csv", index=False)
    sel.to_csv(OUT / "selective_prediction.csv", index=False)
    au.to_csv(OUT / "aurc.csv", index=False)
    cur.to_csv(OUT / "risk_coverage_curves.csv", index=False)
    make_figures(cur, sel)
    write_summary(det, sel, au, a)
    return 0


def make_figures(cur, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"msp": "#4C72B0", "margin": "#55A868", "entropy": "#C44E52",
            "mc_entropy": "#8172B2", "grade_sd": "#CCB974", "random": "#999999"}
    fig, axes = plt.subplots(len(MODELS), len(DATASETS), figsize=(12, 13), sharex=True)
    for i, m in enumerate(MODELS):
        for j, d in enumerate(DATASETS):
            ax = axes[i, j]
            for s in SCORES:
                c = cur[(cur.model == m) & (cur.dataset == d) & (cur.score == s)]
                if len(c):
                    ax.plot(c.coverage, c.risk, color=cols[s], lw=1.4,
                            ls="--" if s == "random" else "-", label=SCORE_LABEL[s])
            ax.set_title(f"{MODELS[m]} | {d}", fontsize=9)
            ax.invert_xaxis()
            if j == 0:
                ax.set_ylabel("error rate on retained cases")
            if i == len(MODELS) - 1:
                ax.set_xlabel("coverage (fraction not referred)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk_coverage.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(12, 7), sharex=True)
    mcols = {"baseline_efficientnet_b3": "#4C72B0", "baseline_resnet50": "#55A868",
             "retfound_linprobe": "#C44E52", "retfound_full": "#8172B2"}
    for j, d in enumerate(DATASETS):
        for m in MODELS:
            for s, ls in [("msp", "-"), ("entropy", ":")]:
                c = sel[(sel.model == m) & (sel.dataset == d) & (sel.score == s)]
                axes[0, j].plot(c.referral_rate * 100, c.acc_gain, ls, color=mcols[m],
                                label=f"{MODELS[m]} ({SCORE_LABEL[s]})")
                axes[1, j].plot(c.referral_rate * 100, c.qwk_gain, ls, color=mcols[m])
        axes[0, j].set_title(d)
        axes[0, j].axhline(0, color="k", lw=0.5)
        axes[1, j].axhline(0, color="k", lw=0.5)
        axes[1, j].set_xlabel("referral rate (%)")
    axes[0, 0].set_ylabel("accuracy gain")
    axes[1, 0].set_ylabel("QWK gain")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_referral_gain.png", dpi=200)
    plt.close(fig)


def write_summary(det, sel, au, a):
    L = [f"# Phase 24 - uncertainty validity (error detection, selective prediction)\n",
         f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}; "
         f"B={a.boot} (selective-prediction CIs B={min(a.boot,1000)}), seed={a.seed}; "
         f"single pass from `{a.sp_dir}`, T=20 from `{a.mc_dir}`._\n",
         "Scores rank cases from most to least trustworthy; the least trustworthy "
         "are referred. MSP = 1 - max softmax probability (simple confidence "
         "baseline). Paired bootstrap differences use identical resamples.\n",
         "## 1. Error detection (grade error), AUROC [95% CI]\n",
         "| Model | Dataset | err. rate | MSP | margin | entropy | entropy T=20 | grade SD |",
         "|---|---|--:|---|---|---|---|---|"]
    g = det[det.error_type == "grade"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            r = g[(g.model == m) & (g.dataset == d)].set_index("score")
            cell = lambda s: (f"{r.loc[s,'auroc']:.3f} [{r.loc[s,'auroc_lo']:.2f}, {r.loc[s,'auroc_hi']:.2f}]"
                              if s in r.index else "—")
            L.append(f"| {ml} | {d} | {r.loc['msp','error_rate']:.2f} | " +
                     " | ".join(cell(s) for s in ["msp", "margin", "entropy", "mc_entropy", "grade_sd"]) + " |")
    L += ["", "## 2. Difference vs MSP baseline (grade error): ΔAUROC [95% CI]\n",
          "| Model | Dataset | margin | entropy | entropy T=20 | grade SD |", "|---|---|---|---|---|---|"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            r = g[(g.model == m) & (g.dataset == d)].set_index("score")
            cell = lambda s: (f"{r.loc[s,'auroc_minus_msp']:+.3f} [{r.loc[s,'diff_lo']:+.3f}, {r.loc[s,'diff_hi']:+.3f}]"
                              if s in r.index else "—")
            L.append(f"| {ml} | {d} | " + " | ".join(cell(s) for s in ["margin", "entropy", "mc_entropy", "grade_sd"]) + " |")
    L += ["", "## 3. Error detection (referable-DR error), AUROC [95% CI]\n",
          "| Model | Dataset | err. rate | MSP | entropy | grade SD |", "|---|---|--:|---|---|---|"]
    rf = det[det.error_type == "referable"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            r = rf[(rf.model == m) & (rf.dataset == d)].set_index("score")
            cell = lambda s: f"{r.loc[s,'auroc']:.3f} [{r.loc[s,'auroc_lo']:.2f}, {r.loc[s,'auroc_hi']:.2f}]"
            L.append(f"| {ml} | {d} | {r.loc['msp','error_rate']:.2f} | " +
                     " | ".join(cell(s) for s in ["msp", "entropy", "grade_sd"]) + " |")
    L += ["", "## 4. Selective prediction with the MSP baseline: accuracy / QWK on retained cases\n",
          "Gain vs. no referral with 95% CI. Random referral has expected gain 0.\n",
          "| Model | Dataset | referral | acc (gain [CI]) | QWK (gain [CI]) | errors captured | ref. sens / spec |",
          "|---|---|--:|---|---|--:|---|"]
    s0 = sel[sel.score == "msp"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            for rate in [0.0, 0.10, 0.20, 0.30]:
                r = s0[(s0.model == m) & (s0.dataset == d) & (np.isclose(s0.referral_rate, rate))].iloc[0]
                if rate == 0:
                    L.append(f"| {ml} | {d} | 0% | {r.accuracy:.3f} | {r.qwk:.3f} | — | {r.ref_sens:.2f} / {r.ref_spec:.2f} |")
                else:
                    L.append(f"| | | {int(rate*100)}% | {r.accuracy:.3f} ({r.acc_gain:+.3f} [{r.acc_gain_lo:+.3f}, {r.acc_gain_hi:+.3f}]) | "
                             f"{r.qwk:.3f} ({r.qwk_gain:+.3f} [{r.qwk_gain_lo:+.3f}, {r.qwk_gain_hi:+.3f}]) | "
                             f"{r.error_capture:.2f} | {r.ref_sens:.2f} / {r.ref_spec:.2f} |")
    L += ["", "## 4b. What gets referred? Human-in-the-loop view (MSP ranking)\n",
          "Referred cases go to a human grader. `missed referable` = referable-DR cases "
          "that are kept AND called non-referable, as a fraction of all referable cases "
          "(the system's miss rate if the grader is correct on referred cases).\n",
          "| Model | Dataset | referral | ref. prev. kept | ref. prev. referred | missed referable |",
          "|---|---|--:|--:|--:|--:|"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            for rate in [0.0, 0.10, 0.20, 0.30]:
                r = s0[(s0.model == m) & (s0.dataset == d) & (np.isclose(s0.referral_rate, rate))].iloc[0]
                L.append(f"| {ml if rate == 0 else ''} | {d if rate == 0 else ''} | {int(rate*100)}% | "
                         f"{r.referable_prev_kept:.2f} | {r.referable_prev_referred:.2f} | {r.missed_referable_frac:.2f} |")
    L += ["", "## 5. Does entropy beat MSP for referral? Accuracy at the same referral rate\n",
          "| Model | Dataset | referral | MSP acc | entropy acc | Δ [95% CI] |", "|---|---|--:|--:|--:|---|"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            for rate in [0.10, 0.20, 0.30]:
                q = sel[(sel.model == m) & (sel.dataset == d) & np.isclose(sel.referral_rate, rate)].set_index("score")
                L.append(f"| {ml} | {d} | {int(rate*100)}% | {q.loc['msp','accuracy']:.3f} | {q.loc['entropy','accuracy']:.3f} | "
                         f"{q.loc['entropy','accuracy']-q.loc['msp','accuracy']:+.3f} [{q.loc['entropy','acc_vs_msp_lo']:+.3f}, {q.loc['entropy','acc_vs_msp_hi']:+.3f}] |")
    L += ["", "## 6. AURC (grade error; lower = better), E-AURC = excess over an oracle ranking\n",
          "| Model | Dataset | risk@100% | MSP | margin | entropy | entropy T=20 | grade SD | random |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for m, ml in MODELS.items():
        for d in DATASETS:
            r = au[(au.model == m) & (au.dataset == d)].set_index("score")
            L.append(f"| {ml} | {d} | {r.loc['msp','risk_full_coverage']:.3f} | " +
                     " | ".join(f"{r.loc[s,'aurc']:.3f}" if s in r.index else "—"
                                for s in ["msp", "margin", "entropy", "mc_entropy", "grade_sd", "random"]) + " |")
    (OUT / "phase24_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
