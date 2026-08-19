"""Unit tests for src.matching — exact-stratum + nearest-neighbor caliper
matching, used to pair ELL and non-ELL essays on grade/prompt/length/quality
before computing feature AUCs (RESEARCH_PLAN.md, Experiment 1)."""
import pandas as pd
import pytest

from src.matching import nearest_neighbor_match


def test_matches_each_treated_row_to_nearest_control_within_group():
    df = pd.DataFrame([
        {"id": "t1", "grp": "A", "treat": 1, "x": 0.0},
        {"id": "t2", "grp": "A", "treat": 1, "x": 5.0},
        {"id": "c1", "grp": "A", "treat": 0, "x": 0.1},
        {"id": "c2", "grp": "A", "treat": 0, "x": 4.9},
        {"id": "c3", "grp": "A", "treat": 0, "x": 10.0},
    ])
    result = nearest_neighbor_match(
        df, id_col="id", treatment_col="treat", treatment_value=1,
        control_value=0, exact_cols=["grp"], distance_cols=["x"], caliper=1.0,
    )
    pairs = dict(zip(result["id_treated"], result["id_control"]))
    assert pairs == {"t1": "c1", "t2": "c2"}


def test_does_not_match_across_exact_groups():
    df = pd.DataFrame([
        {"id": "t1", "grp": "A", "treat": 1, "x": 0.0},
        {"id": "c1", "grp": "B", "treat": 0, "x": 0.0},  # perfect distance, wrong group
    ])
    result = nearest_neighbor_match(
        df, id_col="id", treatment_col="treat", treatment_value=1,
        control_value=0, exact_cols=["grp"], distance_cols=["x"], caliper=1.0,
    )
    assert result.empty


def test_drops_treated_row_with_no_control_within_caliper():
    df = pd.DataFrame([
        {"id": "t1", "grp": "A", "treat": 1, "x": 0.0},
        {"id": "c1", "grp": "A", "treat": 0, "x": 5.0},  # distance 5.0 > caliper
    ])
    result = nearest_neighbor_match(
        df, id_col="id", treatment_col="treat", treatment_value=1,
        control_value=0, exact_cols=["grp"], distance_cols=["x"], caliper=1.0,
    )
    assert result.empty


def test_greedy_matching_is_first_come_first_served_by_id_order():
    # both t1 and t2 are closest to c1; t1 sorts first and should claim it,
    # leaving t2 unmatched since it's the only control in the group.
    df = pd.DataFrame([
        {"id": "t2", "grp": "A", "treat": 1, "x": 0.2},
        {"id": "t1", "grp": "A", "treat": 1, "x": 0.0},
        {"id": "c1", "grp": "A", "treat": 0, "x": 0.1},
    ])
    result = nearest_neighbor_match(
        df, id_col="id", treatment_col="treat", treatment_value=1,
        control_value=0, exact_cols=["grp"], distance_cols=["x"], caliper=1.0,
    )
    assert len(result) == 1
    assert result.iloc[0]["id_treated"] == "t1"
    assert result.iloc[0]["id_control"] == "c1"


def test_matches_on_multiple_distance_columns_via_euclidean_distance():
    df = pd.DataFrame([
        {"id": "t1", "grp": "A", "treat": 1, "x": 0.0, "y": 0.0},
        {"id": "c1", "grp": "A", "treat": 0, "x": 3.0, "y": 4.0},  # dist = 5.0
        {"id": "c2", "grp": "A", "treat": 0, "x": 1.0, "y": 1.0},  # dist = sqrt(2)
    ])
    result = nearest_neighbor_match(
        df, id_col="id", treatment_col="treat", treatment_value=1,
        control_value=0, exact_cols=["grp"], distance_cols=["x", "y"], caliper=10.0,
    )
    assert result.iloc[0]["id_control"] == "c2"
    assert result.iloc[0]["distance"] == pytest.approx(2 ** 0.5)


def test_no_exact_cols_matches_across_whole_frame():
    df = pd.DataFrame([
        {"id": "t1", "treat": 1, "x": 0.0},
        {"id": "c1", "treat": 0, "x": 0.1},
    ])
    result = nearest_neighbor_match(
        df, id_col="id", treatment_col="treat", treatment_value=1,
        control_value=0, exact_cols=[], distance_cols=["x"], caliper=1.0,
    )
    assert len(result) == 1
    assert result.iloc[0]["id_control"] == "c1"
