"""Unit checks for the shared metric, statistics and preprocessing code."""
import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

from metrics import compute_metrics
from phase9_stats import delong_roc_test
from preprocessing import crop_borders, pad_to_square, preprocess_fundus


def test_qwk_perfect_and_ordinal():
    y = np.array([0, 1, 2, 3, 4] * 20)
    assert compute_metrics(y, y)["quadratic_weighted_kappa"] == pytest.approx(1.0)
    # Off-by-one errors must score better than off-by-four errors (ordinal credit).
    near = compute_metrics(y, np.clip(y + 1, 0, 4))["quadratic_weighted_kappa"]
    far = compute_metrics(y, 4 - y)["quadratic_weighted_kappa"]
    assert near > far


def test_delong_auc_matches_sklearn_and_identical_scores_give_p1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(0, 1.0, 400)
    auc_a, auc_b, _, p = delong_roc_test(y, s, s)
    assert auc_a == pytest.approx(roc_auc_score(y, s))
    assert auc_a == auc_b
    assert p == 1.0


def test_delong_detects_a_clearly_better_score():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    good = y + rng.normal(0, 0.5, 1000)
    bad = y + rng.normal(0, 3.0, 1000)
    auc_good, auc_bad, _, p = delong_roc_test(y, good, bad)
    assert auc_good > auc_bad
    assert p < 1e-6


def _synthetic_fundus(h=300, w=400):
    """A bright disc on a black frame, off-centre and non-square, like a raw photo."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - 150) ** 2 + (xx - 220) ** 2 < 120 ** 2
    arr[disc] = (180, 90, 40)
    return arr


def test_crop_removes_black_frame():
    cropped = crop_borders(_synthetic_fundus())
    assert cropped.shape[0] <= 241 and cropped.shape[1] <= 241
    assert cropped.mean() > _synthetic_fundus().mean()


def test_pad_to_square_preserves_content():
    arr = np.ones((100, 160, 3), dtype=np.uint8) * 200
    sq = pad_to_square(arr)
    assert sq.shape[0] == sq.shape[1] == 160
    assert (sq > 0).sum() == (arr > 0).sum()


def test_full_pipeline_outputs_224_rgb():
    out = preprocess_fundus(Image.fromarray(_synthetic_fundus()))
    assert out.size == (224, 224)
    assert out.mode == "RGB"
