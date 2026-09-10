"""
Phase 8 - predictive-uncertainty analysis.

Consumes per-image prediction CSVs that carry an `entropy` column (predictive
entropy of the mean softmax). Two sources are supported:

  * the Phase 6 single-forward-pass files  (outputs/phase6/, default), and
  * the Monte-Carlo-dropout files            (outputs/phase8/mc/, produced by
    scripts/phase8_mc_dropout.py with T stochastic passes).

Pass --preds-dir / --tag to pick. For every model x external dataset it computes:

  * Mann-Whitney U test, predictive entropy of correctly vs incorrectly
    classified images (one-sided: incorrect > correct), with rank-biserial
    effect size;
  * AUROC of predictive entropy as an error detector;
  * selective-prediction curve (accuracy and QWK vs coverage as the most
    uncertain predictions are referred out) and the error-capture fraction at
    90% and 80% retained coverage.

Outputs (outputs/phase8/):
  uncertainty_<tag>.csv                per model x dataset summary
  uncertainty_by_class_<tag>.csv       per model x dataset x grade breakdown
                                       (grouped by true grade and by predicted
                                       grade): n, accuracy, mean/median entropy,
                                       mean entropy of correct vs incorrect
  selective_prediction_<tag>.csv       coverage / accuracy / qwk / err-capture
  fig_uncertainty_<tag>.png            distributions + selective-prediction curve
  fig_uncertainty_byclass_<tag>.png    mean predictive entropy by true grade
  phase8_summary_<tag>.md              manuscript digest
  uncertainty_methods_paragraph.md     drop-in Methods text

Run (after Phase 6; MC files optional):
  .venv/Scripts/python.exe scripts/phase8_uncertainty.py
  .venv/Scripts/python.exe scripts/phase8_uncertainty.py --preds-dir outputs/phase8/mc --tag mc
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "outputs" / "phase8"

MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
          "retfound_linprobe", "retfound_full"]
DATASETS = ["Messidor-2", "IDRiD", "APTOS-internal"]
COVERAGES = np.round(np.arange(1.00, 0.09, -0.05), 2)
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def qwk(y_true, y_pred) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                                   weights="quadratic"))


def selective_curve(entropy: np.ndarray, correct: np.ndarray,
                    y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Accept the least-uncertain fraction `cov`; report accuracy / QWK on the
    retained set and how many of all errors land in the referred-out remainder."""
    order = np.argsort(entropy, kind="stable")      # ascending: most confident first
    n = len(entropy)
    total_err = int((~correct).sum())
    rows = []
    for cov in COVERAGES:
        k = max(1, int(round(cov * n)))
        keep = order[:k]
        referred = order[k:]
        rows.append({
            "coverage": float(cov),
            "n_kept": k,
            "accuracy_kept": float(correct[keep].mean()),
            "qwk_kept": qwk(y_true[keep], y_pred[keep]),
            "errors_referred": int((~correct[referred]).sum()),
            "error_capture_frac": (int((~correct[referred]).sum()) / total_err)
            if total_err else np.nan,
        })
    return pd.DataFrame(rows)


def analyse(preds_dir: Path):
    summ_rows, sel_rows = [], []
    for model in MODELS:
        for ds in DATASETS:
            p = preds_dir / f"{model}__{ds}_preds.csv"
            if not p.is_file():
                continue
            df = pd.read_csv(p)
            if "entropy" not in df.columns:
                continue
            ent = df["entropy"].to_numpy(float)
            y_true = df["y_true"].to_numpy(int)
            y_pred = df["y_pred"].to_numpy(int)
            correct = (y_true == y_pred)

            ent_ok, ent_bad = ent[correct], ent[~correct]
            if len(ent_ok) >= 2 and len(ent_bad) >= 2:
                u, p_one = sps.mannwhitneyu(ent_bad, ent_ok, alternative="greater")
                rb = (2.0 * u) / (len(ent_ok) * len(ent_bad)) - 1.0  # +ve: incorrect more uncertain
                auroc_err = (roc_auc_score((~correct).astype(int), ent)
                             if len(np.unique(correct)) == 2 else np.nan)
            else:
                u, p_one, rb, auroc_err = (np.nan, np.nan, np.nan, np.nan)

            sc = selective_curve(ent, correct, y_true, y_pred)
            sc.insert(0, "dataset", ds)
            sc.insert(0, "model", model)
            sel_rows.append(sc)

            def at(cov):
                r = sc[np.isclose(sc["coverage"], cov)]
                return r.iloc[0] if len(r) else None
            r90, r80 = at(0.90), at(0.80)

            summ_rows.append({
                "model": model, "dataset": ds, "n": len(df),
                "acc_all": float(correct.mean()),
                "qwk_all": qwk(y_true, y_pred),
                "mean_entropy_correct": float(np.mean(ent_ok)) if len(ent_ok) else np.nan,
                "mean_entropy_incorrect": float(np.mean(ent_bad)) if len(ent_bad) else np.nan,
                "mannwhitney_U": float(u), "mannwhitney_p_one_sided": float(p_one),
                "rank_biserial": float(rb),
                "auroc_entropy_error_detection": float(auroc_err),
                "acc_at_90cov": float(r90["accuracy_kept"]) if r90 is not None else np.nan,
                "err_capture_at_90cov": float(r90["error_capture_frac"]) if r90 is not None else np.nan,
                "acc_at_80cov": float(r80["accuracy_kept"]) if r80 is not None else np.nan,
                "err_capture_at_80cov": float(r80["error_capture_frac"]) if r80 is not None else np.nan,
            })
    return (pd.DataFrame(summ_rows),
            pd.concat(sel_rows, ignore_index=True) if sel_rows else pd.DataFrame())


def per_class_uncertainty(preds_dir: Path) -> pd.DataFrame:
    """Break predictive uncertainty down by ICDR grade, both by the true grade
    (does the model know it is unsure on the rarer, sight-threatening grades?)
    and by the predicted grade (are the mass-produced grade-0 calls on the
    external sets confidently wrong?). One row per model x dataset x grouping x
    grade."""
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            p = preds_dir / f"{model}__{ds}_preds.csv"
            if not p.is_file():
                continue
            df = pd.read_csv(p)
            if "entropy" not in df.columns:
                continue
            correct = (df["y_true"].to_numpy(int) == df["y_pred"].to_numpy(int))
            df = df.assign(_correct=correct)
            n_tot = len(df)
            for grouping, col in (("true_grade", "y_true"), ("pred_grade", "y_pred")):
                for g in range(5):
                    sub = df[df[col].to_numpy(int) == g]
                    ent = sub["entropy"].to_numpy(float)
                    cok = sub["_correct"].to_numpy(bool)
                    rows.append({
                        "model": model, "dataset": ds, "group_by": grouping,
                        "grade": g, "grade_name": GRADE_NAMES[g],
                        "n": int(len(sub)),
                        "frac_of_split": round(len(sub) / n_tot, 4) if n_tot else np.nan,
                        "n_correct": int(cok.sum()),
                        "accuracy": float(cok.mean()) if len(sub) else np.nan,
                        "mean_entropy": float(ent.mean()) if len(sub) else np.nan,
                        "median_entropy": float(np.median(ent)) if len(sub) else np.nan,
                        "mean_entropy_correct": float(ent[cok].mean()) if cok.any() else np.nan,
                        "mean_entropy_incorrect": float(ent[~cok].mean()) if (~cok).any() else np.nan,
                    })
    return pd.DataFrame(rows)


def make_byclass_figure(bycls: pd.DataFrame, tag: str) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [d for d in ("Messidor-2", "IDRiD")
           if d in bycls["dataset"].unique()]
    if not ext:
        return None
    d = bycls[(bycls["group_by"] == "true_grade") & (bycls["dataset"].isin(ext))]
    fig, axes = plt.subplots(1, len(ext), figsize=(5.4 * len(ext), 4.2),
                             squeeze=False)
    width = 0.2
    for j, ds in enumerate(ext):
        ax = axes[0][j]
        for k, m in enumerate(MODELS):
            s = d[(d["model"] == m) & (d["dataset"] == ds)].sort_values("grade")
            if s.empty:
                continue
            x = s["grade"].to_numpy() + (k - 1.5) * width
            ax.bar(x, s["mean_entropy"].to_numpy(), width,
                   label=m.replace("baseline_", "").replace("retfound_", "RF-"))
            for xi, ni in zip(x, s["n"].to_numpy()):
                ax.text(xi, 0.01, str(ni), ha="center", va="bottom",
                        fontsize=5, rotation=90)
        ax.set_title(f"Mean predictive entropy by true grade - {ds}")
        ax.set_xlabel("true ICDR grade")
        ax.set_ylabel("mean entropy (nats)")
        ax.set_xticks(range(5))
        ax.set_xticklabels(GRADE_NAMES, rotation=20, fontsize=7)
        ax.set_ylim(0, float(np.log(5)) * 1.02)   # max possible entropy for 5 classes
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle(f"Phase 8 - per-class predictive uncertainty ({tag})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / f"fig_uncertainty_byclass_{tag}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_figure(summ: pd.DataFrame, sel: pd.DataFrame, preds_dir: Path, tag: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [d for d in ("Messidor-2", "IDRiD") if d in summ["dataset"].unique()]
    ncol = len(ext) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 7.2), squeeze=False)

    show_models = ([m for m in ("baseline_efficientnet_b3", "retfound_full")
                    if ext and (preds_dir / f"{m}__{ext[0]}_preds.csv").is_file()])
    for j, ds in enumerate(ext):
        ax = axes[0][j]
        for m, c in zip(show_models, ("#4C72B0", "#C44E52")):
            df = pd.read_csv(preds_dir / f"{m}__{ds}_preds.csv")
            ok = df["y_true"] == df["y_pred"]
            lbl = m.split("_")[-1]
            ax.hist(df.loc[ok, "entropy"], bins=30, alpha=0.5, density=True,
                    color=c, label=f"{lbl} correct")
            ax.hist(df.loc[~ok, "entropy"], bins=30, alpha=0.9, density=True,
                    color=c, histtype="step", linewidth=1.8, label=f"{lbl} incorrect")
        ax.set_title(f"Predictive entropy - {ds}")
        ax.set_xlabel("entropy (nats)"); ax.set_ylabel("density")
        ax.legend(fontsize=7)

    ax = axes[0][-1]
    piv = summ[summ["dataset"].isin(ext)].pivot(
        index="model", columns="dataset", values="auroc_entropy_error_detection")
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_title("AUROC: entropy as error detector"); ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.0); ax.tick_params(axis="x", labelrotation=30)

    for j, ds in enumerate(ext):
        ax = axes[1][j]
        for m in MODELS:
            s = sel[(sel["model"] == m) & (sel["dataset"] == ds)]
            if s.empty:
                continue
            ax.plot(s["coverage"], s["accuracy_kept"], marker="o", ms=3,
                    label=m.replace("baseline_", "").replace("retfound_", "RF-"))
        ax.set_title(f"Selective prediction - {ds}")
        ax.set_xlabel("coverage (fraction retained)")
        ax.set_ylabel("accuracy on retained")
        ax.invert_xaxis(); ax.legend(fontsize=7)

    ax = axes[1][-1]
    for ds in ext:
        for m in MODELS:
            s = sel[(sel["model"] == m) & (sel["dataset"] == ds)]
            if s.empty:
                continue
            ax.plot(s["coverage"], s["error_capture_frac"], marker="s", ms=3,
                    label=f"{m.split('_')[-1]}/{ds}")
    ax.set_title("Error capture vs coverage")
    ax.set_xlabel("coverage (fraction retained)")
    ax.set_ylabel("fraction of errors referred out")
    ax.invert_xaxis(); ax.legend(fontsize=6)

    fig.suptitle(f"Phase 8 - predictive uncertainty ({tag})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / f"fig_uncertainty_{tag}.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def write_summary(summ: pd.DataFrame, tag: str, preds_dir: Path,
                  bycls: pd.DataFrame | None = None) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = [f"# Phase 8 - Predictive uncertainty ({tag})", "",
         f"_Generated {ts}. Source: `{preds_dir.relative_to(PROJECT_DIR)}`._", "",
         "Uncertainty score = predictive entropy of the mean softmax. "
         "Mann-Whitney U is one-sided (incorrect > correct); rank-biserial "
         "positive = incorrect predictions are more uncertain.", "",
         "| Model | Dataset | n | Acc | mean H (correct) | mean H (incorrect) | "
         "MWU p | rank-biserial | AUROC(err) | Acc@90%cov | err-capture@90% |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in summ.itertuples():
        L.append(f"| {r.model} | {r.dataset} | {r.n} | {r.acc_all:.3f} | "
                 f"{r.mean_entropy_correct:.3f} | {r.mean_entropy_incorrect:.3f} | "
                 f"{r.mannwhitney_p_one_sided:.2e} | {r.rank_biserial:+.3f} | "
                 f"{r.auroc_entropy_error_detection:.3f} | {r.acc_at_90cov:.3f} | "
                 f"{r.err_capture_at_90cov:.3f} |")
    L += ["", "Selective-prediction curve (all coverages): "
          f"`outputs/phase8/selective_prediction_{tag}.csv`; "
          f"figure: `outputs/phase8/fig_uncertainty_{tag}.png`.", ""]

    if bycls is not None and not bycls.empty:
        L += ["## Per-class predictive uncertainty", "",
              "Mean predictive entropy (nats) and accuracy per ICDR grade on the "
              "external sets. `H` = mean entropy; accuracy in brackets; grouped "
              "by **true** grade. Full breakdown (incl. grouping by predicted "
              f"grade, median entropy, correct/incorrect split): "
              f"`outputs/phase8/uncertainty_by_class_{tag}.csv`; figure: "
              f"`outputs/phase8/fig_uncertainty_byclass_{tag}.png`.", ""]
        for ds in [d for d in ("Messidor-2", "IDRiD")
                   if d in bycls["dataset"].unique()]:
            L += [f"### {ds} (by true grade)", "",
                  "| Model | " + " | ".join(GRADE_NAMES) + " |",
                  "|---|" + "---|" * 5]
            d = bycls[(bycls.group_by == "true_grade") & (bycls.dataset == ds)]
            for m in MODELS:
                s = d[d.model == m].sort_values("grade")
                if s.empty:
                    continue
                cells = []
                for r in s.itertuples():
                    if r.n == 0:
                        cells.append("--")
                    else:
                        cells.append(f"H={r.mean_entropy:.3f} "
                                     f"({r.accuracy:.2f}, n={r.n})")
                L.append(f"| {m} | " + " | ".join(cells) + " |")
            L.append("")

    (OUT_DIR / f"phase8_summary_{tag}.md").write_text("\n".join(L), encoding="utf-8")


def write_methods_paragraph() -> None:
    txt = """\
### Uncertainty quantification (drop-in Methods text)

Predictive uncertainty was quantified as the Shannon entropy of the predictive
softmax distribution. In the primary analysis this was the entropy of a single
forward pass; in a sensitivity analysis it was the entropy of the mean softmax
over T = 30 stochastic forward passes with dropout (and, for the Vision
Transformer, stochastic-depth) layers held active at inference (Monte-Carlo
dropout; Gal and Ghahramani). For each model and external dataset we tested
whether misclassified images carried higher entropy than correctly classified
images with a one-sided Mann-Whitney U test, reporting the rank-biserial
correlation as the effect size, and we computed the area under the ROC curve for
predictive entropy used directly as a binary error detector. To translate this
into an operating characteristic we plotted selective-prediction curves: images
were ranked by entropy and the most uncertain fraction referred out, and the
accuracy, quadratic-weighted kappa and fraction of all errors captured among the
referred images were recorded as a function of retained coverage. Predictive
entropy was additionally broken down by International Clinical Diabetic
Retinopathy grade, stratified both by the reference grade and by the predicted
grade, to check whether the models expressed greater uncertainty on the rarer
sight-threatening grades and whether the large number of grade-0 predictions on
the external sets were confidently made. Analyses used SciPy and scikit-learn;
code is released as scripts/phase8_uncertainty.py and
scripts/phase8_mc_dropout.py.
"""
    (OUT_DIR / "uncertainty_methods_paragraph.md").write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds-dir", default="outputs/phase6")
    ap.add_argument("--tag", default="singlepass",
                    help="label for output files (e.g. singlepass, mc)")
    args = ap.parse_args()

    preds_dir = (PROJECT_DIR / args.preds_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Phase 8: source={preds_dir}  tag={args.tag}")

    summ, sel = analyse(preds_dir)
    if summ.empty:
        print("no usable preds found")
        return 1
    bycls = per_class_uncertainty(preds_dir)
    summ.to_csv(OUT_DIR / f"uncertainty_{args.tag}.csv", index=False)
    sel.to_csv(OUT_DIR / f"selective_prediction_{args.tag}.csv", index=False)
    bycls.to_csv(OUT_DIR / f"uncertainty_by_class_{args.tag}.csv", index=False)
    fig = make_figure(summ, sel, preds_dir, args.tag)
    fig_bc = make_byclass_figure(bycls, args.tag)
    write_summary(summ, args.tag, preds_dir, bycls)
    write_methods_paragraph()

    print(f"wrote {OUT_DIR}/uncertainty_{args.tag}.csv")
    print(f"wrote {OUT_DIR}/uncertainty_by_class_{args.tag}.csv")
    print(f"wrote {OUT_DIR}/selective_prediction_{args.tag}.csv")
    print(f"wrote {fig}")
    if fig_bc:
        print(f"wrote {fig_bc}")
    print(f"wrote {OUT_DIR}/phase8_summary_{args.tag}.md\n")
    cols = ["model", "dataset", "acc_all", "mean_entropy_correct",
            "mean_entropy_incorrect", "mannwhitney_p_one_sided",
            "auroc_entropy_error_detection", "err_capture_at_90cov"]
    print(summ[cols].to_string(index=False))

    print("\n=== per-class mean entropy (by true grade, external sets) ===")
    piv = (bycls[(bycls.group_by == "true_grade")
                 & (bycls.dataset.isin(["Messidor-2", "IDRiD"]))]
           .pivot_table(index=["dataset", "model"], columns="grade_name",
                        values="mean_entropy")
           .reindex(columns=GRADE_NAMES))
    print(piv.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
