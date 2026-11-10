"""
Phase 27 - inter-rater agreement for the blinded saliency rating.

Reviewer request (round 2): "...and inter-rater agreement for any subjective
assessment."

Input: two or more completed copies of
outputs/phase7/rating_packet/rating_sheet_BLANK.csv (one per rater, saved as
rater_<initials>.csv, `blind_id` unchanged), plus the withheld KEY.csv.
Optionally a re-rating sheet (same rater, >= 2 weeks later, items re-shuffled)
for intra-rater reliability: pass it with --rerate RATER=path.

For each item `focus_plausible` (0/1) and `lesion_aligned` (0/1/NA):
  * observed agreement, Cohen's kappa (2 raters) or Fleiss' kappa (>= 3),
    Gwet's AC1 (robust to the kappa prevalence paradox when most maps are rated
    the same way), Krippendorff's alpha (nominal; handles NA);
  * 95% CIs by bootstrap over items (B=2000, seed 42);
  * the same per model family and per dataset (do raters agree as well on
    RETFound maps as on CNN maps?);
  * agreement of the rater consensus (majority; ties -> NA) with the automated
    proxy (Phase 7) and, where the item is an IDRiD-Segmentation image, with the
    mask-based localisation (Phase 26 lift > 1);
  * weighted kappa (quadratic) for the 1-5 `confidence` column.
Interpretation bands (Landis & Koch) are reported only alongside the CI.

Outputs (outputs/phase27/): agreement.csv, agreement_by_group.csv,
consensus_vs_proxy.csv, phase27_summary.md

Run:
  python scripts/phase27_rater_agreement.py --sheets outputs/phase7/rating_packet/rater_AB.csv outputs/phase7/rating_packet/rater_CD.csv
  python scripts/phase27_rater_agreement.py --selftest      # synthetic check of the statistics
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "outputs" / "phase7" / "rating_packet"
OUT = ROOT / "outputs" / "phase27"


# ----------------------------------------------------------------- statistics
def _counts(R: np.ndarray, cats) -> np.ndarray:
    """R: items x raters (nan = missing). Returns items x categories counts."""
    return np.stack([(R == c).sum(1) for c in cats], 1).astype(float)


def pct_agree(R):
    ok = ~np.isnan(R).any(1)
    R = R[ok]
    return float((R == R[:, :1]).all(1).mean()) if len(R) else np.nan


def cohen(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return np.nan
    cats = np.unique(np.r_[a, b])
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def fleiss(R, cats):
    ok = ~np.isnan(R).any(1)
    n = _counts(R[ok], cats)
    m = n.sum(1)[0] if len(n) else 0
    if m < 2:
        return np.nan
    P_i = ((n * (n - 1)).sum(1)) / (m * (m - 1))
    p_j = n.sum(0) / n.sum()
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else np.nan


def gwet_ac1(R, cats):
    ok = (~np.isnan(R)).sum(1) >= 2
    n = _counts(R[ok], cats)
    r = n.sum(1)
    if len(n) == 0:
        return np.nan
    Pa = ((n * (n - 1)).sum(1) / (r * (r - 1))).mean()
    pi = (n / r[:, None]).mean(0)
    q = len(cats)
    Pe = (pi * (1 - pi)).sum() / (q - 1)
    return float((Pa - Pe) / (1 - Pe))


def kripp_alpha_nominal(R, cats):
    n = _counts(R, cats)
    r = n.sum(1)
    n = n[r >= 2]
    r = r[r >= 2]
    if len(n) == 0:
        return np.nan
    o = np.zeros((len(cats), len(cats)))
    for row, ri in zip(n, r):
        o += (np.outer(row, row) - np.diag(row)) / (ri - 1)
    nc = o.sum(1)
    N = nc.sum()
    Do = (o.sum() - np.trace(o))
    De = (N * N - (nc ** 2).sum()) / (N - 1)
    return float(1 - Do / De) if De > 0 else np.nan


def wkappa_quadratic(a, b, cats=(1, 2, 3, 4, 5)):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok].astype(int), b[ok].astype(int)
    if len(a) == 0:
        return np.nan
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    W = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


def all_stats(R, cats=(0, 1)):
    s = {"n_items": int((~np.isnan(R)).sum(1).__ge__(2).sum()),
         "pct_agree": pct_agree(R),
         "gwet_ac1": gwet_ac1(R, cats),
         "kripp_alpha": kripp_alpha_nominal(R, cats),
         "prevalence_1": float(np.nanmean(R))}
    if R.shape[1] == 2:
        s["kappa"] = cohen(R[:, 0], R[:, 1])
        s["kappa_type"] = "Cohen"
    else:
        s["kappa"] = fleiss(R, cats)
        s["kappa_type"] = "Fleiss"
    return s


def boot(R, B=2000, seed=42, cats=(0, 1)):
    rng = np.random.default_rng(seed)
    keys = ["pct_agree", "kappa", "gwet_ac1", "kripp_alpha"]
    acc = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, len(R), len(R))
        s = all_stats(R[idx], cats)
        for k in keys:
            acc[k].append(s[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in acc.items()}


def band(k):
    if np.isnan(k):
        return "n/a"
    for t, lab in [(0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= t:
            return lab
    return "almost perfect"


# ----------------------------------------------------------------- io
def load_sheet(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, dtype={"blind_id": str})
    for c in ["focus_plausible", "lesion_aligned", "confidence"]:
        d[c] = pd.to_numeric(d[c].replace({"NA": np.nan, "na": np.nan, "": np.nan}), errors="coerce")
    return d.set_index("blind_id")


def matrix(sheets: dict, col: str, index) -> np.ndarray:
    return np.column_stack([sheets[r].reindex(index)[col].to_numpy(float) for r in sheets])


def run(sheets: dict, key: pd.DataFrame, B: int, rerate: dict | None = None) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    index = key.index
    rows, grp = [], []
    for col in ["focus_plausible", "lesion_aligned"]:
        R = matrix(sheets, col, index)
        s = all_stats(R)
        ci = boot(R, B)
        rows.append({"item": col, **s, **{f"{k}_lo": v[0] for k, v in ci.items()},
                     **{f"{k}_hi": v[1] for k, v in ci.items()}})
        for gcol in [c for c in ["family", "dataset", "correct"] if c in key.columns]:
            for gv, sub in key.groupby(gcol):
                Rg = matrix(sheets, col, sub.index)
                sg = all_stats(Rg)
                cig = boot(Rg, max(200, B // 4))
                grp.append({"item": col, "group": gcol, "value": gv, **sg,
                            "kappa_lo": cig["kappa"][0], "kappa_hi": cig["kappa"][1],
                            "ac1_lo": cig["gwet_ac1"][0], "ac1_hi": cig["gwet_ac1"][1]})
    raters = list(sheets)
    conf = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a = sheets[raters[i]].reindex(index)["confidence"].to_numpy(float)
            b = sheets[raters[j]].reindex(index)["confidence"].to_numpy(float)
            conf.append((raters[i], raters[j], wkappa_quadratic(a, b)))
    intra = []
    for r, p in (rerate or {}).items():
        s2 = load_sheet(Path(p))
        for col in ["focus_plausible", "lesion_aligned"]:
            a = sheets[r].reindex(index)[col].to_numpy(float)
            b = s2.reindex(index)[col].to_numpy(float)
            intra.append((r, col, cohen(a, b), int((~np.isnan(a) & ~np.isnan(b)).sum())))

    # consensus vs automated proxy
    cons_rows = []
    Rf = matrix(sheets, "focus_plausible", index)
    m = np.nanmean(Rf, 1)
    cons = np.where(np.isnan(m) | (m == 0.5), np.nan, (m > 0.5).astype(float))
    key = key.assign(consensus=cons)
    if "proxy_plausible" in key.columns:
        px = key["proxy_plausible"].astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}).to_numpy(float)
        cons_rows.append({"comparison": "consensus vs automated proxy", "n": int((~np.isnan(cons) & ~np.isnan(px)).sum()),
                          "pct_agree": pct_agree(np.c_[cons, px]), "kappa": cohen(cons, px),
                          "ac1": gwet_ac1(np.c_[cons, px], (0, 1))})
    pd.DataFrame(rows).to_csv(OUT / "agreement.csv", index=False)
    pd.DataFrame(grp).to_csv(OUT / "agreement_by_group.csv", index=False)
    pd.DataFrame(cons_rows).to_csv(OUT / "consensus_vs_proxy.csv", index=False)

    L = ["# Phase 27 - inter-rater agreement, blinded saliency rating\n",
         f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}; raters: {', '.join(raters)}; "
         f"items: {len(index)}; bootstrap over items B={B}._\n",
         "| Item | n | % agree | κ (type) [95% CI] | Gwet AC1 [CI] | Krippendorff α [CI] | rated 1 |",
         "|---|--:|--:|---|---|---|--:|"]
    for r in rows:
        L.append(f"| {r['item']} | {r['n_items']} | {100*r['pct_agree']:.1f} | {r['kappa']:.2f} ({r['kappa_type']}) "
                 f"[{r['kappa_lo']:.2f}, {r['kappa_hi']:.2f}] — {band(r['kappa'])} | {r['gwet_ac1']:.2f} [{r['gwet_ac1_lo']:.2f}, {r['gwet_ac1_hi']:.2f}] | "
                 f"{r['kripp_alpha']:.2f} [{r['kripp_alpha_lo']:.2f}, {r['kripp_alpha_hi']:.2f}] | {100*r['prevalence_1']:.0f}% |")
    if grp:
        L += ["", "## By subgroup\n", "| Item | group | value | n | % agree | κ [CI] | AC1 [CI] |", "|---|---|---|--:|--:|---|---|"]
        for g in grp:
            L.append(f"| {g['item']} | {g['group']} | {g['value']} | {g['n_items']} | {100*g['pct_agree']:.1f} | "
                     f"{g['kappa']:.2f} [{g['kappa_lo']:.2f}, {g['kappa_hi']:.2f}] | {g['gwet_ac1']:.2f} [{g['ac1_lo']:.2f}, {g['ac1_hi']:.2f}] |")
    if conf:
        L += ["", "Confidence (1-5), quadratic-weighted κ: " + "; ".join(f"{a}–{b} {k:.2f}" for a, b, k in conf)]
    if intra:
        L += ["", "Intra-rater (re-rating) Cohen κ: " + "; ".join(f"{r} {c} {k:.2f} (n={n})" for r, c, k, n in intra)]
    for c in cons_rows:
        L += ["", f"Rater consensus vs automated proxy: n={c['n']}, agree {100*c['pct_agree']:.1f}%, κ {c['kappa']:.2f}, AC1 {c['ac1']:.2f}"]
    txt = "\n".join(L) + "\n"
    (OUT / "phase27_summary.md").write_text(txt, encoding="utf-8")
    return txt


def selftest():
    """Check estimators against textbook values and simulated raters."""
    # Fleiss (1971) style check: perfect agreement -> 1; independent raters -> ~0
    rng = np.random.default_rng(0)
    truth = rng.random(400) < 0.7
    def rater(acc):
        flip = rng.random(400) > acc
        return np.where(flip, ~truth, truth).astype(float)
    R_perfect = np.c_[truth, truth].astype(float)
    R_indep = np.c_[rng.random(400) < 0.7, rng.random(400) < 0.7].astype(float)
    R_good = np.c_[rater(0.9), rater(0.9), rater(0.9)]
    out = {"perfect": all_stats(R_perfect), "independent": all_stats(R_indep), "3 raters acc .9": all_stats(R_good)}
    # Cohen kappa vs sklearn
    try:
        from sklearn.metrics import cohen_kappa_score
        a, b = R_good[:, 0], R_good[:, 1]
        assert abs(cohen(a, b) - cohen_kappa_score(a, b)) < 1e-9
        out["cohen_matches_sklearn"] = True
    except ImportError:
        pass
    # Krippendorff alpha ~ Fleiss for complete nominal data with many items
    out["alpha_minus_fleiss_3r"] = all_stats(R_good)["kripp_alpha"] - all_stats(R_good)["kappa"]
    for k, v in out.items():
        print(k, v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", nargs="+")
    ap.add_argument("--key", default=str(PACKET / "KEY.csv"))
    ap.add_argument("--rerate", nargs="*", default=[], help="RATER=path for intra-rater re-rating sheets")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default outputs/phase27)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    if a.selftest:
        selftest()
        return 0
    if not a.sheets or len(a.sheets) < 2:
        print("need >= 2 rater sheets (--sheets a.csv b.csv)")
        return 1
    key = pd.read_csv(a.key, dtype={"blind_id": str}).set_index("blind_id")
    if "model" in key.columns:
        key["family"] = np.where(key["model"].str.contains("retfound"), "RETFound", "CNN")
    sheets = {Path(p).stem.replace("rater_", ""): load_sheet(Path(p)) for p in a.sheets}
    rerate = dict(x.split("=", 1) for x in a.rerate)
    print(run(sheets, key, a.boot, rerate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
