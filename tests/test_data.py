"""Unit tests for src.data — shared PERSUADE loading/cleaning used by every
experiment script, so cleaning criteria can't silently drift between them."""
import numpy as np
import pandas as pd
import pytest

from src.data import add_standardized_covariates, load_and_clean


def _write_csv(tmp_path, rows):
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_and_clean_keeps_only_yes_no_ell_status(tmp_path):
    rows = [
        {"ell_status": "Yes", "grade_level": 8, "holistic_essay_score": 3, "word_count": 200, "prompt_name": "P1", "full_text": "a"},
        {"ell_status": "No", "grade_level": 8, "holistic_essay_score": 4, "word_count": 250, "prompt_name": "P1", "full_text": "b"},
        {"ell_status": "", "grade_level": 8, "holistic_essay_score": 3, "word_count": 200, "prompt_name": "P1", "full_text": "c"},
        {"ell_status": np.nan, "grade_level": 8, "holistic_essay_score": 3, "word_count": 200, "prompt_name": "P1", "full_text": "d"},
    ]
    path = _write_csv(tmp_path, rows)
    clean, manifest = load_and_clean(path)
    assert set(clean["ell_clean"]) == {"Yes", "No"}
    assert len(clean) == 2
    assert manifest["n_total_rows"] == 4
    assert manifest["n_after_filter"] == 2
    assert manifest["n_ell"] == 1
    assert manifest["n_non_ell"] == 1


def test_load_and_clean_drops_rows_missing_required_fields(tmp_path):
    rows = [
        {"ell_status": "Yes", "grade_level": 8, "holistic_essay_score": 3, "word_count": 200, "prompt_name": "P1", "full_text": "a"},
        {"ell_status": "No", "grade_level": np.nan, "holistic_essay_score": 4, "word_count": 250, "prompt_name": "P1", "full_text": "b"},
        {"ell_status": "No", "grade_level": 8, "holistic_essay_score": np.nan, "word_count": 250, "prompt_name": "P1", "full_text": "c"},
    ]
    path = _write_csv(tmp_path, rows)
    clean, manifest = load_and_clean(path)
    assert len(clean) == 1
    assert manifest["n_dropped_missing_fields"] == 2


def test_add_standardized_covariates_are_zero_mean_unit_variance():
    df = pd.DataFrame({
        "word_count": [100, 200, 300, 400, 500],
        "holistic_essay_score": [1, 2, 3, 4, 5],
    })
    out = add_standardized_covariates(df)
    assert out["z_logwc"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["z_logwc"].std(ddof=1) == pytest.approx(1.0)
    assert out["z_holistic"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["z_holistic"].std(ddof=1) == pytest.approx(1.0)


def test_add_standardized_covariates_preserves_row_order():
    df = pd.DataFrame({
        "word_count": [100, 500],
        "holistic_essay_score": [1, 5],
    })
    out = add_standardized_covariates(df)
    assert out["z_logwc"].iloc[0] < out["z_logwc"].iloc[1]
    assert out["z_holistic"].iloc[0] < out["z_holistic"].iloc[1]
