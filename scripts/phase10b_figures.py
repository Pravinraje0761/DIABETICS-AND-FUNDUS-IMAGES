"""
Phase 10b - extra manuscript figures for the IEEEtran build.

  outputs/figures/fig3_perclass_heatmap.{png,pdf}   per-class sensitivity grid
  outputs/figures/fig4_confusion.{png,pdf}          row-normalised confusion matrices
  outputs/figures/fig5_grade_distribution.{png,pdf} ICDR grade share per dataset

Run after Phase 6:
  .venv/Scripts/python.exe scripts/phase10b_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
P6 = PROJECT_DIR / "outputs" / "phase6"
OUT = PROJECT_DIR / "outputs" / "figures"
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Prolif."]
MODEL_LABEL = {
    "baseline_efficientnet_b3": "EfficientNet-B3",
    "baseline_resnet50": "ResNet-50",
    "retfound_linprobe": "RETFound (lin. probe)",
    "retfound_full": "RETFound (full FT)",
}
MODELS = list(MODEL_LABEL)
DATASETS = ["APTOS-internal", "Messidor-2", "IDRiD"]


def fig_perclass_heatmap() -> None:
    csv = P6 / "master_results_per_class.csv"
    if not csv.is_file():
        print("skip heatmap: master_results_per_class.csv missing")
        return
    df = pd.read_csv(csv)
    df["grade"] = df["grade"].astype(str)
    order = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 3.6), sharey=True)
    im = None
    for ax, ds in zip(axes, DATASETS):
        M = np.full((len(MODELS), 5), np.nan)
        for i, m in enumerate(MODELS):
            sub = df[(df.model == m) & (df.dataset == ds)]
            for _, r in sub.iterrows():
                if r["grade"] in order:
                    M[i, order.index(r["grade"])] = r["sensitivity"]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(ds, fontsize=10)
        ax.set_xticks(range(5))
        ax.set_xticklabels(GRADES, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=8)
        for i in range(len(MODELS)):
            for j in range(5):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if M[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="per-class sensitivity")
    fig.suptitle("Per-class (ICDR grade) sensitivity across models and datasets",
                 fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_perclass_heatmap.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig3_perclass_heatmap.png/.pdf")


def fig_confusion() -> None:
    show = ["baseline_efficientnet_b3", "retfound_full"]
    fig, axes = plt.subplots(len(show), len(DATASETS), figsize=(10.5, 7))
    im = None
    for r, m in enumerate(show):
        for c, ds in enumerate(DATASETS):
            ax = axes[r][c]
            mj = P6 / f"{m}__{ds}_metrics.json"
            if not mj.is_file():
                ax.axis("off")
                continue
            cm = np.array(json.loads(mj.read_text())["confusion_matrix"], float)
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xticklabels(GRADES, rotation=40, ha="right", fontsize=7)
            ax.set_yticklabels(GRADES, fontsize=7)
            if r == 0:
                ax.set_title(ds, fontsize=10)
            if c == 0:
                ax.set_ylabel(MODEL_LABEL[m] + "\ntrue grade", fontsize=9)
            if r == len(show) - 1:
                ax.set_xlabel("predicted grade", fontsize=8)
    fig.suptitle("Row-normalised confusion matrices (internal vs external)", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="fraction of true class")
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_confusion.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig4_confusion.png/.pdf")


def fig_grade_distribution() -> None:
    share = {
        "APTOS 2019": [49.3, 10.1, 27.3, 5.3, 8.1],
        "Messidor-2": [58.3, 15.5, 19.9, 4.3, 2.0],
        "IDRiD": [28.4, 4.8, 34.3, 18.5, 14.1],
    }
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(5)
    w = 0.26
    for k, (name, v) in enumerate(share.items()):
        ax.bar(x + (k - 1) * w, v, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["0 No DR", "1 Mild", "2 Moderate", "3 Severe",
                        "4 Proliferative"])
    ax.set_ylabel("share of gradable images (%)")
    ax.set_title("ICDR grade distribution by dataset")
    ax.legend()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_grade_distribution.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig5_grade_distribution.png/.pdf")


if __name__ == "__main__":
    fig_perclass_heatmap()
    fig_confusion()
    fig_grade_distribution()
