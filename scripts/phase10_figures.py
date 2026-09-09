"""
Phase 10 - manuscript figures that are not already emitted by the analysis
scripts (Phase 7 -> fig_explainability.png, Phase 8 -> fig_uncertainty_*.png).

Produces:
  outputs/figures/fig1_study_design.png / .pdf   pipeline schematic
  outputs/figures/fig2_performance_forest.png / .pdf
        QWK / accuracy / macro-AUC for every model across the internal and two
        external datasets, with 95% bootstrap CI error bars (from
        outputs/phase9/bootstrap_ci.csv)

Run after Phase 9:
  .venv/Scripts/python.exe scripts/phase10_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
PHASE9 = PROJECT_DIR / "outputs" / "phase9" / "bootstrap_ci.csv"
OUT = PROJECT_DIR / "outputs" / "figures"

MODEL_LABEL = {
    "baseline_efficientnet_b3": "EfficientNet-B3 (baseline)",
    "baseline_resnet50": "ResNet-50 (baseline)",
    "retfound_linprobe": "RETFound (linear probe)",
    "retfound_full": "RETFound (full fine-tune)",
}
MODEL_ORDER = list(MODEL_LABEL)
DATASET_ORDER = ["APTOS-internal", "Messidor-2", "IDRiD"]
DS_COLOR = {"APTOS-internal": "#4C72B0", "Messidor-2": "#C44E52", "IDRiD": "#55A868"}


def fig1_study_design() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EEF3FA", ec="#33517A"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.2, color="#555"))

    box(1, 39, 20, 9, "APTOS 2019\n(n=3662)\nDEVELOPMENT ONLY", fc="#E7F0E7")
    box(1, 27, 20, 9, "Messidor-2\n(n=1744)\nEXTERNAL TEST", fc="#FBECEC")
    box(1, 15, 20, 9, "IDRiD\n(n=455)\nEXTERNAL TEST", fc="#FBECEC")

    box(27, 27, 16, 21, "Single\npreprocessing\npipeline\n(crop / pad /\n224x224 / norm)")

    box(49, 39, 20, 9, "APTOS split\n70 / 10 / 20\nstratified, seed 42")
    box(49, 22, 22, 12, "Fine-tune on APTOS train\n- EfficientNet-B3 / ResNet-50\n"
                        "  (ImageNet)\n- RETFound ViT-L\n  (linear probe + full)")

    box(76, 33, 22, 12, "Internal test\n(APTOS, n=733)\nsame code path")
    box(76, 17, 22, 12, "EXTERNAL VALIDATION\nMessidor-2 + IDRiD\nmodels unchanged, one run")

    box(49, 3, 49, 10, "Explainability (Grad-CAM++ CNN / ViT Grad-CAM) + "
                       "predictive uncertainty\n(entropy, selective prediction) + "
                       "bootstrap 95% CIs, DeLong test")

    arrow(21, 43, 27, 40)
    arrow(21, 31, 27, 36)
    arrow(21, 19, 27, 32)
    arrow(43, 40, 49, 43)
    arrow(43, 34, 49, 30)
    arrow(59, 39, 60, 34)
    arrow(71, 28, 76, 24)
    arrow(69, 43, 76, 40)
    arrow(60, 22, 60, 13)
    ax.text(50, 50, "Study design: train/tune on APTOS only; Messidor-2 and IDRiD "
                    "opened once, for all models, at external validation",
            ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_study_design.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_forest() -> None:
    if not PHASE9.is_file():
        print(f"skip fig2: {PHASE9} not found (run scripts/phase9_stats.py first)")
        return
    ci = pd.read_csv(PHASE9)
    metrics = [("quadratic_weighted_kappa", "Quadratic-weighted kappa"),
               ("accuracy", "Accuracy"),
               ("auc_ovr_macro", "Macro one-vs-rest AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True)

    yticks, ylabels = [], []
    row = 0
    gap = 1.0
    for mkey in MODEL_ORDER:
        for di, dkey in enumerate(DATASET_ORDER):
            yticks.append(row)
            ylabels.append(f"{MODEL_LABEL[mkey]} - {dkey}" if di == 1
                           else f"{'':30s}{dkey}")
            row += 1
        row += gap

    xlims = {"quadratic_weighted_kappa": (0.2, 1.0), "accuracy": (0.2, 1.0),
             "auc_ovr_macro": (0.5, 1.0)}
    for ax, (mcol, mtitle) in zip(axes, metrics):
        row = 0
        for mkey in MODEL_ORDER:
            for dkey in DATASET_ORDER:
                sub = ci[(ci.model == mkey) & (ci.dataset == dkey)
                         & (ci.metric == mcol)]
                if len(sub):
                    r = sub.iloc[0]
                    ax.errorbar(r["point"], row,
                                xerr=[[max(r["point"] - r["ci_lo"], 0)],
                                      [max(r["ci_hi"] - r["point"], 0)]],
                                fmt="o", ms=6, capsize=3, lw=1.4,
                                color=DS_COLOR[dkey])
                row += 1
            row += gap
        ax.set_title(mtitle, fontsize=10)
        ax.set_xlim(*xlims[mcol])
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ylabels, fontsize=7)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=DS_COLOR[d], label=d)
               for d in DATASET_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               title="dataset", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model performance across the internal and two external datasets "
                 "(points = estimate, bars = 95% bootstrap CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_performance_forest.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    fig1_study_design()
    print(f"wrote {OUT}/fig1_study_design.png/.pdf")
    fig2_forest()
    if (OUT / "fig2_performance_forest.png").is_file():
        print(f"wrote {OUT}/fig2_performance_forest.png/.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
