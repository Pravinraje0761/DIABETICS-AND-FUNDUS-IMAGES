"""Recompute the paper's headline results from the committed per-image predictions.

If any of these fail, a number in the manuscript no longer matches the prediction
files it was computed from.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metrics import compute_metrics
from phase9_stats import delong_roc_test

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "outputs" / "phase6"
PROB_COLS = [f"prob_{c}" for c in range(5)]

# Manuscript Table 2 (master results): accuracy, QWK, macro one-vs-rest AUC.
TABLE_2 = {
    ("baseline_efficientnet_b3", "APTOS-internal"): (733, 0.804, 0.876, 0.928),
    ("baseline_efficientnet_b3", "Messidor-2"): (1744, 0.626, 0.499, 0.775),
    ("baseline_efficientnet_b3", "IDRiD"): (455, 0.497, 0.760, 0.830),
    ("baseline_resnet50", "APTOS-internal"): (733, 0.789, 0.875, 0.924),
    ("baseline_resnet50", "Messidor-2"): (1744, 0.614, 0.435, 0.754),
    ("baseline_resnet50", "IDRiD"): (455, 0.448, 0.707, 0.841),
    ("retfound_linprobe", "APTOS-internal"): (733, 0.759, 0.847, 0.908),
    ("retfound_linprobe", "Messidor-2"): (1744, 0.589, 0.394, 0.758),
    ("retfound_linprobe", "IDRiD"): (455, 0.336, 0.626, 0.778),
    ("retfound_full", "APTOS-internal"): (733, 0.769, 0.869, 0.915),
    ("retfound_full", "Messidor-2"): (1744, 0.602, 0.368, 0.758),
    ("retfound_full", "IDRiD"): (455, 0.420, 0.720, 0.821),
}


def _load(model, dataset):
    return pd.read_csv(PREDS / f"{model}__{dataset}_preds.csv")


@pytest.mark.parametrize("model,dataset", sorted(TABLE_2))
def test_table2_row_recomputes_from_predictions(model, dataset):
    n, acc, qwk, auc = TABLE_2[(model, dataset)]
    df = _load(model, dataset)
    m = compute_metrics(df.y_true.values, df.y_pred.values, df[PROB_COLS].values)
    assert len(df) == n
    assert round(m["accuracy"], 3) == acc
    assert round(m["quadratic_weighted_kappa"], 3) == qwk
    assert round(m["auc_ovr_macro"], 3) == auc


def test_predictions_are_valid_distributions():
    for model, dataset in TABLE_2:
        probs = _load(model, dataset)[PROB_COLS].values
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
        assert (probs >= 0).all()


@pytest.mark.parametrize("dataset,other,auc_base,auc_other,significant", [
    ("Messidor-2", "retfound_full", 0.842, 0.789, True),
    ("Messidor-2", "retfound_linprobe", 0.842, 0.781, True),
    ("IDRiD", "retfound_full", 0.962, 0.873, True),
    ("IDRiD", "retfound_linprobe", 0.962, 0.903, True),
    ("APTOS-internal", "retfound_full", 0.980, 0.978, False),
])
def test_referable_dr_delong(dataset, other, auc_base, auc_other, significant):
    """Referable DR (ICDR >= 2): baseline vs RETFound, paired DeLong test."""
    a = _load("baseline_efficientnet_b3", dataset)
    b = _load(other, dataset)
    assert (a.id_code.values == b.id_code.values).all()  # paired on the same images
    y = (a.y_true.values >= 2).astype(int)
    score = lambda d: d[["prob_2", "prob_3", "prob_4"]].sum(axis=1).values  # noqa: E731
    auc_a, auc_b, _, p = delong_roc_test(y, score(a), score(b))
    assert round(auc_a, 3) == auc_base
    assert round(auc_b, 3) == auc_other
    assert (p < 0.05) == significant
