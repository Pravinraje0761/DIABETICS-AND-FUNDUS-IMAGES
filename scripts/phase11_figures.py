"""
Phase 11 figures for the IEEEtran build. Reads outputs/phase11/.

  outputs/figures/fig_reliability.{png,pdf}   reliability diagrams
  outputs/figures/fig_roc_pr.{png,pdf}        referable-DR ROC + PR curves
  outputs/figures/fig_perclass_ci.{png,pdf}   per-class sensitivity with 95% CI

Run after scripts/phase11_review_response.py:
  .venv/Scripts/python.exe scripts/phase11_figures.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
P11 = PROJECT / "outputs" / "phase11"
OUT = PROJECT / "outputs" / "figures"
DS = ["APTOS-internal", "Messidor-2", "IDRiD"]
MLAB = {"baseline_efficientnet_b3": "EfficientNet-B3", "baseline_resnet50": "ResNet-50",
        "retfound_linprobe": "RETFound lin.probe", "retfound_full": "RETFound full FT"}


def reliability():
    show = ["baseline_efficientnet_b3", "retfound_full"]
    fig, axes = plt.subplots(len(show), len(DS), figsize=(10.5, 6.6))
    for r, m in enumerate(show):
        for c, d in enumerate(DS):
            ax = axes[r][c]
            f = P11 / f"reliability_{m}__{d}.csv"
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            if f.is_file():
                t = pd.read_csv(f).dropna()
                if len(t):
                    ax.bar(t["bin_conf"], t["bin_acc"], width=0.06, alpha=0.8,
                           color="#4C72B0", edgecolor="k", lw=0.3)
                    gaps = t["bin_conf"] - t["bin_acc"]
                    ax.bar(t["bin_conf"], gaps, bottom=t["bin_acc"], width=0.06,
                           alpha=0.35, color="#C44E52")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(d, fontsize=10)
            if c == 0:
                ax.set_ylabel(MLAB[m] + "\naccuracy", fontsize=9)
            if r == len(show) - 1:
                ax.set_xlabel("confidence", fontsize=9)
    fig.suptitle("Reliability diagrams (blue = accuracy per confidence bin; "
                 "red = calibration gap)", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_reliability.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_reliability")


def roc_pr():
    f = P11 / "referable_roc_pr.csv"
    if not f.is_file():
        print("skip roc_pr"); return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(2, len(DS), figsize=(11, 6.6))
    for c, d in enumerate(DS):
        for k, curve in enumerate(("roc", "pr")):
            ax = axes[k][c]
            for m in MLAB:
                s = df[(df.model == m) & (df.dataset == d) & (df.curve == curve)]
                if s.empty:
                    continue
                s = s.sort_values("x")
                ax.plot(s["x"], s["y"], lw=1.4,
                        label=f"{MLAB[m]} ({s['auc'].iloc[0]:.2f})")
            if curve == "roc":
                ax.plot([0, 1], [0, 1], "k--", lw=0.7)
                ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
            else:
                ax.set_xlabel("recall"); ax.set_ylabel("precision")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            if k == 0:
                ax.set_title(d, fontsize=10)
            ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle(r"Referable DR (ICDR grade $\geq$ 2): ROC (top) and "
                 "precision-recall (bottom); AUROC in legend", fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_roc_pr.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_roc_pr")


def perclass_ci():
    f = P11 / "per_class_ci.csv"
    if not f.is_file():
        print("skip perclass_ci"); return
    df = pd.read_csv(f)
    show = ["baseline_efficientnet_b3", "retfound_full"]
    grades = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    fig, axes = plt.subplots(1, len(DS), figsize=(11, 3.8), sharex=True)
    for c, d in enumerate(DS):
        ax = axes[c]
        y = 0
        yt, yl = [], []
        for g in grades:
            for m, col in zip(show, ("#4C72B0", "#C44E52")):
                r = df[(df.model == m) & (df.dataset == d) & (df.grade == g)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r["sensitivity"], y,
                                xerr=[[max(r["sensitivity"] - r["sens_lo"], 0)],
                                      [max(r["sens_hi"] - r["sensitivity"], 0)]],
                                fmt="o", ms=4, capsize=2, color=col)
                yt.append(y); yl.append(f"{g[:4]} / {MLAB[m].split()[0]}")
                y += 1
            y += 0.6
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6)
        ax.set_xlim(0, 1); ax.invert_yaxis()
        ax.set_title(d, fontsize=10); ax.set_xlabel("per-class sensitivity (95% CI)")
    fig.suptitle("Per-class sensitivity with 95% bootstrap CI: EfficientNet-B3 "
                 "(blue) vs RETFound full FT (red)", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_perclass_ci.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_perclass_ci")


if __name__ == "__main__":
    reliability()
    roc_pr()
    perclass_ci()
