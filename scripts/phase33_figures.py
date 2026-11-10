"""
Phase 33 - manuscript figures drawn from the Phase 31 analysis (deduplicated internal set,
examination-level Messidor-2 resampling, seed-averaged primary contrast), so that figures,
tables and text share one source. Fig 1 (design) and Fig 5 (lesion masks) are unchanged
from Phase 29 and re-exported here; Figs 2-4 are redrawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase29_figures as F  # noqa: E402  (style, colours, helpers)
from phase29_figures import plt, W, MM, MODELS, LAB, COL, MK, LS, INK2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P31 = ROOT / "outputs" / "phase31"
OUT = ROOT / "outputs" / "phase33" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
F.OUT = OUT
CI = pd.read_csv(P31 / "ci.csv")
CON = pd.read_csv(P31 / "contrasts.csv")
SC = pd.read_csv(P31 / "seed_contrasts.csv")
GRP = {"effb3": "baseline_efficientnet_b3", "retfound_linprobe": "retfound_linprobe", "retfound_full": "retfound_full"}
EXT = ["Messidor-2", "IDRiD"]


def cival(m, s, k):
    r = CI[(CI.model == m) & (CI.set == s) & (CI.metric == k)].iloc[0]
    return r.value, r.lo, r.hi


def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 72 * MM), gridspec_kw={"width_ratios": [1, 1.1]})
    sc = SC[(SC.internal == "APTOS-internal") & (SC.metric == "qwk")]
    fams = ["effb3", "retfound_full", "retfound_linprobe"]
    for e_i, ext in enumerate(EXT):
        for j, g in enumerate(fams + ["baseline_resnet50"]):
            m = GRP.get(g, g)
            x = e_i + (j - 1.5) * 0.2
            if g in GRP:
                r = sc[(sc.group == g) & (sc.external == ext)].iloc[0]
                per = [float(v) for v in str(r.per_seed_drops).split(";")]
                a.scatter(np.full(len(per), x - 0.05), per, s=9, marker=MK[m],
                          facecolor="white", edgecolor=COL[m], lw=0.8, zorder=3)
                mean = r["drop"]
            else:
                c = CON[(CON.internal == "APTOS-internal") & (CON.model == m) & (CON.external == ext) & (CON.metric == "qwk")].iloc[0]
                mean = c["drop"]
            a.scatter([x + 0.03], [mean], s=34, marker=MK[m], color=COL[m], edgecolor="white", lw=0.6, zorder=4,
                      label=LAB[m] if e_i == 0 else None)
    a.set_xticks(range(2)); a.set_xticklabels(EXT)
    a.set_ylabel("Internal-to-external QWK drop")
    a.yaxis.grid(True); a.set_axisbelow(True)
    a.text(0.03, 0.42, "open: individual training runs\nfilled: mean over runs", transform=a.transAxes,
           fontsize=6.3, color=INK2, va="center", ha="left")
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08), handletextpad=0.3, columnspacing=1.2)
    a.text(-0.16, 1.02, "a", transform=a.transAxes, fontweight="bold", fontsize=10)
    # b: excess drop vs EfficientNet-B3
    rows = []
    for g in ["retfound_full", "retfound_linprobe"]:
        for ext in EXT:
            r = sc[(sc.group == g) & (sc.external == ext)].iloc[0]
            rows.append((GRP[g], ext, r.ddrop, r.h_lo, r.h_hi, r.h_p_holm, int(r.n_seeds)))
    for ext in EXT:
        c = CON[(CON.internal == "APTOS-internal") & (CON.model == "baseline_resnet50") & (CON.external == ext) & (CON.metric == "qwk")].iloc[0]
        rows.append(("baseline_resnet50", ext, c.ddrop, c.ddrop_lo, c.ddrop_hi, np.nan, 1))
    lo_all = min(r[3] for r in rows); hi_all = max(r[4] for r in rows)
    xmax = hi_all + 0.02
    for k, (m, d, est, lo, hi, ph, n) in enumerate(rows):
        yk = len(rows) - 1 - k
        b.errorbar(est, yk, xerr=[[est - lo], [hi - est]], fmt=MK[m], color=COL[m], ms=4.5,
                   elinewidth=1.0, capsize=0, mec="white", mew=0.5)
        b.text(min(lo_all, -0.05) - 0.012, yk, f"{LAB[m]}{' (single run)' if n == 1 and m == 'baseline_resnet50' else ''}, {d}", ha="right", va="center", fontsize=6.8)
        if not np.isnan(ph):
            b.text(xmax + 0.005, yk, f"Holm p = {ph:.3f}" if ph >= 0.001 else "Holm p < 0.001",
                   ha="left", va="center", fontsize=6.3, color=INK2)
    b.axvline(0, color="#5f5e5a", lw=0.7)
    b.set_xlim(min(lo_all, -0.05) - 0.005, xmax); b.set_yticks([]); b.spines["left"].set_visible(False)
    b.set_xlabel("Excess QWK drop vs EfficientNet-B3 (95% CI)\n← smaller drop        larger drop →")
    b.xaxis.grid(True); b.set_axisbelow(True)
    b.text(-0.95, 1.02, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=1.05)
    F.save(fig, "Fig2")


def fig3():
    fig = plt.figure(figsize=(W, 78 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.32)
    a = fig.add_subplot(gs[0])
    metrics = [("qwk", "Observed"), ("qwk_rw", "Case mix\nmatched"), ("qwk4", "Grades 0+1\nmerged"), ("qwk4_rw", "Both")]
    xs = np.arange(len(metrics))
    con = CON[CON.internal == "APTOS-internal"]
    for e_i, ext in enumerate(EXT):
        for j, m in enumerate(["baseline_efficientnet_b3", "retfound_full"]):
            sub = con[(con.model == m) & (con.external == ext)].set_index("metric")
            v = np.array([sub.loc[k, "drop"] for k, _ in metrics])
            lo = v - np.array([sub.loc[k, "drop_lo"] for k, _ in metrics])
            hi = np.array([sub.loc[k, "drop_hi"] for k, _ in metrics]) - v
            off = (e_i * 2 + j - 1.5) * 0.19
            a.errorbar(xs + off, v, yerr=[lo, hi], fmt=MK[m], color=COL[m], ms=4.2,
                       mfc=COL[m] if ext == "Messidor-2" else "white", mec=COL[m], mew=1.0,
                       elinewidth=0.9, capsize=0, ls="none", label=f"{LAB[m]}, {ext}")
    a.axhline(0, color="#5f5e5a", lw=0.7)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in metrics])
    a.set_ylabel("Internal-to-external QWK drop (95% CI)")
    a.yaxis.grid(True); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.3)
    a.text(-0.13, 1.02, "a", transform=a.transAxes, fontweight="bold", fontsize=10)
    b = fig.add_subplot(gs[1])
    shades = ["#e8f1fb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
    hatches = ["", "....", "", "////", ""]
    bars = []
    for ext in EXT:
        p = F.preds("baseline_efficientnet_b3", ext)
        for g in [0, 1, 2]:
            c = np.bincount(p.y_pred[p.y_true == g], minlength=5)
            bars.append((f"{ext}\ngrade {g}", c / c.sum(), int(c.sum())))
    y = np.arange(len(bars))[::-1]
    for yi, (lab, frac, n) in zip(y, bars):
        left = 0
        for k in range(5):
            b.barh(yi, frac[k], left=left, color=shades[k], edgecolor="white", lw=0.8,
                   hatch=hatches[k], height=0.72, label=f"graded {k}" if yi == y[0] else None)
            left += frac[k]
        b.text(1.02, yi, f"n={n}", va="center", fontsize=6.5, color=INK2)
    b.set_yticks(y); b.set_yticklabels([l for l, _, _ in bars], fontsize=7)
    b.set_xlim(0, 1); b.set_xlabel("Share of images (EfficientNet-B3)")
    b.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2),
             handlelength=1.2, columnspacing=0.8)
    b.text(-0.42, 1.02, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
    F.save(fig, "Fig3")


def fig4():
    fig, axs = plt.subplots(1, 3, figsize=(W, 62 * MM), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    rates = [0, 10, 20, 30]
    for k, ext in enumerate(EXT):
        ax = axs[k]
        for m in MODELS:
            v = [cival(m, ext, "missed_0" if r == 0 else f"missed_msp_{r}")[0] * 100 for r in rates]
            ax.plot(rates, v, LS[m], color=COL[m], marker=MK[m], ms=3.4, mec="white", mew=0.4, label=LAB[m])
        v = [cival("baseline_efficientnet_b3", ext, "missed_0" if r == 0 else f"missed_random_{r}")[0] * 100 for r in rates]
        ax.plot(rates, v, color="#a8a79f", lw=1.0, ls=(0, (1, 1.5)), label="Random referral (EfficientNet-B3)")
        ax.set_xticks(rates); ax.set_xlabel("Images referred to a grader (%)")
        if k == 0:
            ax.set_ylabel("Referable DR missed by the system (%)")
        ax.set_ylim(0, 70); ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.text(0.5, 0.97, ext, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2)
        ax.text(-0.22, 1.09, "ab"[k], transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.1), fontsize=6.5, handlelength=2.6, columnspacing=1.0)
    c = axs[2]
    for j, m in enumerate(MODELS):
        for i, d in enumerate(["APTOS-internal"] + EXT):
            for sc_, fill in [("msp", True), ("entropy", False)]:
                v, lo, hi = cival(m, d, f"err_auc_{sc_}")
                x = i + (j - 1.5) * 0.19 + (0.05 if sc_ == "entropy" else -0.05)
                c.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt=MK[m], color=COL[m], mfc=COL[m] if fill else "white",
                           mec=COL[m], ms=3.6, mew=0.9, elinewidth=0.8, capsize=0)
    c.axhline(0.5, color="#5f5e5a", lw=0.7, ls=(0, (2, 2)))
    c.set_xticks(range(3)); c.set_xticklabels(["APTOS\n(internal)", "Messidor-2", "IDRiD"])
    c.set_ylabel("Error-detection AUC (95% CI)"); c.set_ylim(0.35, 0.95)
    c.yaxis.grid(True); c.set_axisbelow(True)
    from matplotlib.lines import Line2D
    c.legend(handles=[Line2D([], [], marker="o", color="#5f5e5a", ls="none", ms=4, label="Max. probability (filled)"),
                      Line2D([], [], marker="o", color="#5f5e5a", mfc="white", ls="none", ms=4, label="Entropy (open)")],
             frameon=False, fontsize=6.3, loc="upper right")
    c.text(-0.22, 1.09, "c", transform=c.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=0.38, bottom=0.2)
    F.save(fig, "Fig4")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": F.fig1, "2": fig2, "3": fig3, "4": fig4, "5": F.fig5}[w]()
        print("Fig", w, "done", flush=True)
