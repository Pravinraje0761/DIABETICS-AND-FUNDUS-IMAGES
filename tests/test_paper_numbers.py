"""Recompute headline results of the current manuscript from the committed per-image predictions.

The manuscript (paper/submission_io_final, Table 3) reports, for the seed-42 run of each model,
external quadratic weighted kappa (QWK) and referable-DR (grade >= 2) sensitivity, specificity
and AUC on Messidor-2 (1,744 images) and the official IDRiD release (516 images). Every number in
the manuscript is generated into paper/submission_io/generated/numbers.json by
scripts/phase32_tex_numbers.py; these tests recompute a subset independently from the prediction
files and check that the generated values agree. If any fail, a number in the manuscript no
longer matches the prediction files it was computed from.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / "outputs" / "phase6"      # seed-42 runs: APTOS internal and Messidor-2
P28 = ROOT / "outputs" / "phase28"    # official IDRiD release (all runs) and later runs
NUMBERS = ROOT / "paper" / "submission_io" / "generated" / "numbers.json"
PROB_COLS = [f"prob_{c}" for c in range(5)]
ALIAS = {"baseline_efficientnet_b3": "eff", "baseline_resnet50": "res",
         "retfound_linprobe": "lp", "retfound_full": "full"}


def _load(model, dataset):
    if dataset == "IDRiD":
        return pd.read_csv(P28 / f"{model}__IDRiD-official_preds.csv")
    return pd.read_csv(P6 / f"{model}__{dataset}_preds.csv")


def _external(model, dataset):
    d = _load(model, dataset)
    y, h = d.y_true.values, d.y_pred.values
    ref, call = y >= 2, h >= 2
    return dict(
        n=len(d),
        qwk=cohen_kappa_score(y, h, weights="quadratic"),
        sens=(call & ref).sum() / ref.sum(),
        spec=(~call & ~ref).sum() / (~ref).sum(),
        auc=roc_auc_score(ref, d[["prob_2", "prob_3", "prob_4"]].sum(axis=1)),
    )


# Manuscript Table 3, external rows (seed-42 run of each model).
TABLE_3 = {
    ("baseline_efficientnet_b3", "Messidor-2"): (1744, 0.499, 39, 97, 0.842),
    ("baseline_resnet50", "Messidor-2"): (1744, 0.435, 36, 96, 0.789),
    ("retfound_linprobe", "Messidor-2"): (1744, 0.394, 43, 90, 0.781),
    ("retfound_full", "Messidor-2"): (1744, 0.487, 39, 98, 0.817),
    ("baseline_efficientnet_b3", "IDRiD"): (516, 0.726, 83, 91, 0.927),
    ("baseline_resnet50", "IDRiD"): (516, 0.662, 72, 95, 0.912),
    ("retfound_linprobe", "IDRiD"): (516, 0.590, 83, 66, 0.882),
    ("retfound_full", "IDRiD"): (516, 0.703, 73, 93, 0.876),
}


@pytest.mark.parametrize("model,dataset", sorted(TABLE_3))
def test_table3_external_row_recomputes_from_predictions(model, dataset):
    n, qwk, sens, spec, auc = TABLE_3[(model, dataset)]
    m = _external(model, dataset)
    assert m["n"] == n
    assert round(m["qwk"], 3) == qwk
    assert round(100 * m["sens"]) == sens
    assert round(100 * m["spec"]) == spec
    assert round(m["auc"], 3) == auc


@pytest.mark.parametrize("model,dataset", sorted(TABLE_3))
def test_generated_manuscript_numbers_match_predictions(model, dataset):
    """The macros typeset in the manuscript agree with an independent recomputation."""
    nv = json.loads(NUMBERS.read_text())
    key = ALIAS[model] + {"Messidor-2": "mes", "IDRiD": "idr"}[dataset]
    m = _external(model, dataset)
    assert nv[key + "qwk"] == f"{m['qwk']:.3f}"
    assert nv[key + "refauc"] == f"{m['auc']:.3f}"


def test_no_dr_mild_boundary_counts():
    """Results: EfficientNet-B3 grades mild DR as no DR on Messidor-2 and no DR as mild on IDRiD."""
    m = _load("baseline_efficientnet_b3", "Messidor-2")
    i = _load("baseline_efficientnet_b3", "IDRiD")
    assert ((m.y_true == 1) & (m.y_pred == 0)).sum() == 259 and (m.y_true == 1).sum() == 270
    assert ((m.y_true == 2) & (m.y_pred == 0)).sum() == 249 and (m.y_true == 2).sum() == 347
    assert ((i.y_true == 0) & (i.y_pred == 1)).sum() == 82 and (i.y_true == 0).sum() == 168


def test_predictions_are_valid_distributions():
    for model, dataset in TABLE_3:
        probs = _load(model, dataset)[PROB_COLS].values
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
        assert (probs >= 0).all()
