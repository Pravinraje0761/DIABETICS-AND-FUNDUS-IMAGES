"""
Shared evaluation metrics for 5-class DR grading (Phases 4-9).

Primary metric = quadratic-weighted kappa (ordinal, imbalance-robust). Accuracy
is reported but is not the headline - a grade-0-only predictor scores ~0.5 on
APTOS/Messidor-2. Per-class sensitivity/specificity and one-vs-rest AUC round
out the picture the methodology asks for.

Pure numpy + scikit-learn; no torch. Feed it plain arrays:
    y_true : (N,) int in [0, num_classes)
    y_pred : (N,) int
    y_prob : (N, num_classes) float, optional (needed for AUC)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def per_class_sens_spec(cm: np.ndarray) -> tuple[list[float], list[float]]:
    """From a confusion matrix (rows=true, cols=pred) -> per-class sens, spec."""
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    sens, spec = [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else float("nan"))
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else float("nan"))
    return sens, spec


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    num_classes: int = 5,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sens, spec = per_class_sens_spec(cm)

    out: dict = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "balanced_accuracy": float(np.nanmean(sens)),
        "per_class": {
            GRADE_NAMES[c]: {
                "support": int(cm[c, :].sum()),
                "sensitivity": float(sens[c]),
                "specificity": float(spec[c]),
            }
            for c in range(num_classes)
        },
        "confusion_matrix": cm.astype(int).tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        present = np.unique(y_true)
        try:
            if len(present) < 2:
                raise ValueError("need >=2 classes present for AUC")
            macro_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=labels
            )
            out["auc_ovr_macro"] = float(macro_auc)
            per_class_auc = {}
            for c in range(num_classes):
                if c in present and (y_true != c).any():
                    per_class_auc[GRADE_NAMES[c]] = float(
                        roc_auc_score((y_true == c).astype(int), y_prob[:, c])
                    )
                else:
                    per_class_auc[GRADE_NAMES[c]] = float("nan")
            out["auc_ovr_per_class"] = per_class_auc
        except (ValueError, IndexError) as exc:
            out["auc_ovr_macro"] = float("nan")
            out["auc_error"] = str(exc)

    return out


def format_metrics(d: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"== {title} ==")
    lines.append(f"n={d['n']}  acc={d['accuracy']:.4f}  "
                 f"QWK={d['quadratic_weighted_kappa']:.4f}  "
                 f"macroF1={d['macro_f1']:.4f}  balAcc={d['balanced_accuracy']:.4f}")
    if "auc_ovr_macro" in d:
        lines.append(f"AUC(ovr macro)={d['auc_ovr_macro']:.4f}")
    lines.append("per-class  grade            support   sens    spec    auc")
    for c, name in enumerate(GRADE_NAMES):
        pc = d["per_class"][name]
        auc = d.get("auc_ovr_per_class", {}).get(name, float("nan"))
        lines.append(f"           {c} {name:<14} {pc['support']:>6}  "
                     f"{pc['sensitivity']:.3f}  {pc['specificity']:.3f}  {auc:.3f}")
    lines.append("confusion (rows=true, cols=pred):")
    for row in d["confusion_matrix"]:
        lines.append("  " + " ".join(f"{v:5d}" for v in row))
    return "\n".join(lines)
