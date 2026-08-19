"""Unit tests for src.feedback_verification — checking whether a review
feedback claim about an essay's writing (e.g. "your sentences are
monotonous") is actually supported by the measured Tier 1 features,
independent of whether those features predict quality (Experiment 6,
second deliverable)."""
import numpy as np
import pytest

from src.feedback_verification import CLAIM_FEATURE_MAP, percentile_rank, verify_claim


def test_percentile_rank_of_minimum_is_zero():
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(1.0, reference) == pytest.approx(0.0)


def test_percentile_rank_of_maximum_is_near_one():
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(5.0, reference) == pytest.approx(0.8)  # 4 of 5 reference values are below it


def test_percentile_rank_of_median():
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(3.0, reference) == pytest.approx(0.4)  # 2 of 5 below it


def test_verify_claim_supported_when_value_in_expected_high_tail():
    text = "Short. This one is much longer with many more words in it to pad the length out considerably."
    reference_values = {"sentence_length_std": np.linspace(0, 1, 100)}  # essay's std will be high
    result = verify_claim(text, "sentence_variety_high", reference_values)
    assert result["feature"] == "sentence_length_std"
    assert result["supported"] is True


def test_verify_claim_unsupported_when_value_in_middle():
    text = "One two three. Four five six. Seven eight nine."  # uniform sentence lengths -> std = 0
    reference_values = {"sentence_length_std": np.linspace(0, 10, 100)}
    result = verify_claim(text, "sentence_variety_high", reference_values)
    assert result["percentile"] == pytest.approx(0.0)
    assert result["supported"] is False


def test_verify_claim_low_direction_flips_expected_tail():
    text = "Short. This one is much longer with many more words in it to pad the length out considerably."
    reference_values = {"sentence_length_std": np.linspace(0, 1, 100)}
    result = verify_claim(text, "sentence_variety_low", reference_values)
    # same essay as the "high" test, but claim direction is "low" -> should NOT be supported
    assert result["supported"] is False


def test_all_claim_types_reference_a_real_tier1_feature():
    from src.features import TIER1_FEATURES
    for claim_type, (feature_name, direction) in CLAIM_FEATURE_MAP.items():
        assert feature_name in TIER1_FEATURES
        assert direction in ("high", "low")


def test_verify_claim_raises_on_unknown_claim_type():
    with pytest.raises(KeyError):
        verify_claim("some text", "not_a_real_claim", {})
