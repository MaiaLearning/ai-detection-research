"""Unit tests for src.stats_utils.bootstrap_paired_delta_ci — the statistic
Experiment 6's verdict hinges on (M1-M0 delta, not two separate CIs
subtracted, since M0/M1 predictions are correlated: same folds, same rows).
"""
import numpy as np
import pytest

from src.stats_utils import bootstrap_paired_delta_ci


def test_paired_delta_ci_is_zero_when_predictions_identical():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred_a = y + rng.normal(scale=0.5, size=200)
    pred_b = pred_a.copy()  # identical predictions -> delta must be exactly 0
    point, lo, hi = bootstrap_paired_delta_ci(
        y, pred_a, pred_b, metric_fn=lambda yt, yp: float(np.corrcoef(yt, yp)[0, 1]),
        n_boot=500, seed=1,
    )
    assert point == pytest.approx(0.0, abs=1e-9)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.0, abs=1e-9)


def test_paired_delta_ci_detects_clear_improvement():
    rng = np.random.default_rng(2)
    y = rng.normal(size=500)
    pred_a = rng.normal(size=500)  # uncorrelated with y: a useless model
    pred_b = y + rng.normal(scale=0.1, size=500)  # tightly tracks y: a good model
    point, lo, hi = bootstrap_paired_delta_ci(
        y, pred_a, pred_b, metric_fn=lambda yt, yp: float(np.corrcoef(yt, yp)[0, 1]),
        n_boot=500, seed=3,
    )
    assert point > 0.5
    assert lo > 0.0  # CI should exclude zero -- a real, detectable improvement


def test_paired_delta_ci_is_reproducible_given_same_seed():
    rng = np.random.default_rng(4)
    y = rng.normal(size=100)
    pred_a = y + rng.normal(scale=1.0, size=100)
    pred_b = y + rng.normal(scale=0.3, size=100)
    metric = lambda yt, yp: float(np.corrcoef(yt, yp)[0, 1])
    result_1 = bootstrap_paired_delta_ci(y, pred_a, pred_b, metric_fn=metric, n_boot=300, seed=9)
    result_2 = bootstrap_paired_delta_ci(y, pred_a, pred_b, metric_fn=metric, n_boot=300, seed=9)
    assert result_1 == result_2
