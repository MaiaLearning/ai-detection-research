"""Unit tests for src.dispersion — across-document dispersion and centroid
statistics used by Experiment 7 to distinguish within-document polish (H1)
from across-document homogenization (H2) as drivers of detectability."""
import numpy as np
import pytest

from src.dispersion import (
    centroid,
    cosine_similarity,
    covariance_determinant,
    covariance_trace,
    mean_distance_to_centroid,
    project_onto_vector,
    standardize_by_reference,
)


def test_standardize_by_reference_uses_reference_mean_and_std_not_own():
    reference = np.array([[0.0], [2.0], [4.0]])  # mean=2, std=2
    target = np.array([[2.0], [4.0]])
    result = standardize_by_reference(target, reference)
    # (2-2)/2=0, (4-2)/2=1 -- uses REFERENCE's mean/std, not target's own
    assert result[:, 0] == pytest.approx([0.0, 1.0])


def test_centroid_is_the_mean_point():
    X = np.array([[0.0, 0.0], [2.0, 4.0]])
    assert centroid(X) == pytest.approx([1.0, 2.0])


def test_mean_distance_to_centroid_is_zero_for_identical_points():
    X = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
    assert mean_distance_to_centroid(X) == pytest.approx(0.0)


def test_mean_distance_to_centroid_matches_hand_computation():
    # centroid at (0,0); two points at distance 1 and 3 respectively
    X = np.array([[1.0, 0.0], [-3.0, 0.0]])  # centroid = (-1, 0)
    # distances: |1-(-1)|=2, |-3-(-1)|=2 -> mean = 2
    assert mean_distance_to_centroid(X) == pytest.approx(2.0)


def test_covariance_trace_is_zero_for_no_spread():
    X = np.tile([1.0, 2.0, 3.0], (10, 1))
    assert covariance_trace(X) == pytest.approx(0.0, abs=1e-10)


def test_covariance_trace_sums_per_feature_variance():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0, 1, 5000), rng.normal(0, 2, 5000)])
    # var ~ 1 and ~4 -> trace ~ 5
    assert covariance_trace(X) == pytest.approx(5.0, abs=0.3)


def test_covariance_determinant_is_zero_for_perfectly_correlated_features():
    X = np.column_stack([np.arange(20.0), np.arange(20.0) * 2])  # second is 2x the first
    assert covariance_determinant(X) == pytest.approx(0.0, abs=1e-6)


def test_project_onto_vector_along_axis():
    origin = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])  # unit vector along x
    point = np.array([3.0, 5.0])
    # projection onto x-axis direction, from origin, is just the x-coordinate
    assert project_onto_vector(point, origin, direction) == pytest.approx(3.0)


def test_project_onto_vector_normalizes_direction():
    origin = np.array([0.0, 0.0])
    direction = np.array([2.0, 0.0])  # non-unit vector, same direction as above
    point = np.array([3.0, 5.0])
    assert project_onto_vector(point, origin, direction) == pytest.approx(3.0)


def test_cosine_similarity_identical_direction_is_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])  # same direction, different magnitude
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_direction_is_negative_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, -2.0, -3.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)
