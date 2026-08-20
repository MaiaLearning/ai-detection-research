"""Unit tests for src.stats_utils — AUC and correlation point estimates with
bootstrap CIs, used to report results with confidence intervals rather than
bare point estimates (CLAUDE.md)."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from src.stats_utils import (
    bootstrap_auc_ci,
    bootstrap_corr_ci,
    bootstrap_stat_ci,
    compute_auc,
    partial_pearson,
    partial_spearman,
    rate_at_threshold,
    threshold_at_fpr,
)


def test_compute_auc_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    scores = [1, 2, 3, 10, 11, 12]
    assert compute_auc(y, scores) == pytest.approx(1.0)


def test_compute_auc_no_signal_when_score_distributions_identical():
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [1, 1, 2, 2, 3, 3, 4, 4]
    assert compute_auc(y, scores) == pytest.approx(0.5)


def test_compute_auc_inverted_relationship_is_below_half():
    # higher score -> more likely class 0, so AUC (for predicting class 1) < 0.5
    y = [0, 0, 0, 1, 1, 1]
    scores = [12, 11, 10, 3, 2, 1]
    assert compute_auc(y, scores) == pytest.approx(0.0)


def test_bootstrap_auc_ci_contains_point_estimate():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    scores = y + rng.normal(0, 1.5, size=200)  # moderate, noisy separation
    point, lo, hi = bootstrap_auc_ci(y, scores, n_boot=500, seed=0)
    assert lo <= point <= hi


def test_bootstrap_auc_ci_is_narrow_and_high_for_perfect_separation():
    y = [0] * 20 + [1] * 20
    scores = list(range(20)) + list(range(100, 120))
    point, lo, hi = bootstrap_auc_ci(y, scores, n_boot=500, seed=0)
    assert point == pytest.approx(1.0)
    assert lo > 0.9


def test_bootstrap_auc_ci_is_reproducible_given_same_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=100)
    scores = rng.normal(0, 1, size=100)
    result_a = bootstrap_auc_ci(y, scores, n_boot=300, seed=7)
    result_b = bootstrap_auc_ci(y, scores, n_boot=300, seed=7)
    assert result_a == result_b


def test_partial_spearman_close_to_raw_when_control_is_independent():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=0.5, size=n)
    z = rng.normal(size=n)  # independent of both x and y
    raw = spearmanr(x, y).statistic
    partial = partial_spearman(x, y, z)
    assert partial == pytest.approx(raw, abs=0.05)


def test_partial_spearman_removes_correlation_induced_by_shared_confound():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.1, size=n)  # x driven mostly by z
    y = z + rng.normal(scale=0.1, size=n)  # y driven mostly by z, independent noise
    raw = spearmanr(x, y).statistic
    partial = partial_spearman(x, y, z)
    assert raw > 0.8  # strong spurious correlation via shared confound
    assert abs(partial) < 0.2  # mostly explained away once z is controlled


def test_partial_pearson_close_to_raw_when_control_is_independent():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=0.5, size=n)
    z = rng.normal(size=n)  # independent of both x and y
    raw = np.corrcoef(x, y)[0, 1]
    partial = partial_pearson(x, y, z)
    assert partial == pytest.approx(raw, abs=0.05)


def test_partial_pearson_removes_correlation_induced_by_shared_confound():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.1, size=n)  # x driven mostly by z
    y = z + rng.normal(scale=0.1, size=n)  # y driven mostly by z, independent noise
    raw = np.corrcoef(x, y)[0, 1]
    partial = partial_pearson(x, y, z)
    assert raw > 0.8  # strong spurious correlation via shared confound
    assert abs(partial) < 0.2  # mostly explained away once z is controlled


def test_partial_pearson_matches_partial_spearman_on_rank_data():
    # partial_pearson on already-ranked data should equal partial_spearman
    # on the raw data (Spearman IS Pearson on ranks) -- a cross-check that
    # the two formulas are the same computation applied to different inputs.
    rng = np.random.default_rng(2)
    n = 500
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.3, size=n)
    y = -z + rng.normal(scale=0.3, size=n)
    from scipy.stats import rankdata
    ranked = partial_pearson(rankdata(x), rankdata(y), rankdata(z))
    assert ranked == pytest.approx(partial_spearman(x, y, z), abs=1e-9)


def test_bootstrap_corr_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    y = x + rng.normal(scale=0.5, size=300)
    point, lo, hi = bootstrap_corr_ci(x, y, lambda a, b: spearmanr(a, b).statistic, n_boot=500, seed=0)
    assert lo <= point <= hi


def test_bootstrap_corr_ci_is_reproducible_given_same_seed():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    corr_fn = lambda a, b: spearmanr(a, b).statistic
    result_a = bootstrap_corr_ci(x, y, corr_fn, n_boot=300, seed=9)
    result_b = bootstrap_corr_ci(x, y, corr_fn, n_boot=300, seed=9)
    assert result_a == result_b


def test_bootstrap_stat_ci_resamples_all_arrays_in_correspondence():
    # a and c are elementwise identical; if the bootstrap resamples every
    # array with the SAME draw of indices (as it must, to keep rows
    # corresponding), then a[idx] == c[idx] on every single resample.
    # If any array were left un-resampled (or resampled independently),
    # this would fail almost every draw for a large n.
    n = 50
    a = np.arange(n, dtype=float)
    b = np.arange(n, dtype=float) * 2
    c = np.arange(n, dtype=float)
    agreement = lambda aa, bb, cc: float(np.mean(aa == cc))
    point, lo, hi = bootstrap_stat_ci([a, b, c], agreement, n_boot=200, seed=0)
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_threshold_at_fpr_matches_manual_quantile():
    negative_scores = np.arange(1, 101)  # 1..100
    threshold = threshold_at_fpr(negative_scores, target_fpr=0.1)
    assert threshold == pytest.approx(90.1)
    # exactly 10 of the 100 negative scores (91..100) are >= threshold
    assert rate_at_threshold(negative_scores, threshold) == pytest.approx(0.1)


def test_rate_at_threshold_all_above():
    scores = [5, 6, 7, 8]
    assert rate_at_threshold(scores, 5.0) == pytest.approx(1.0)


def test_rate_at_threshold_none_above():
    scores = [1, 2, 3]
    assert rate_at_threshold(scores, 10.0) == pytest.approx(0.0)


def test_bootstrap_stat_ci_supports_partial_correlation_with_control_resampled():
    # regression test for a bug where the control array (z) in a partial
    # correlation wasn't resampled alongside x/y, decoupling the bootstrap
    # distribution from the reported point estimate.
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.1, size=n)
    y = z + rng.normal(scale=0.1, size=n)
    point, lo, hi = bootstrap_stat_ci(
        [x, y, z], partial_spearman, n_boot=300, seed=1,
    )
    assert lo <= point <= hi
    assert abs(point) < 0.2
