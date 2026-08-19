"""Verify whether a review-feedback claim about an essay's writing is
supported by the measured Tier 1 features — independent of whether those
features predict quality (Experiment 6, second deliverable).

Only claims that map cleanly onto a measured feature are covered here.
Claims about specificity, personal detail, or argument quality aren't
measurable this way and are deliberately left out (EXPERIMENT_6.md).
"""
import numpy as np

from src.features import TIER1_FEATURES

# (feature_name, direction) -- "high" means the claim asserts the essay is
# unusually high on this feature relative to the reference distribution;
# "low" means unusually low.
CLAIM_FEATURE_MAP = {
    "sentence_variety_high": ("sentence_length_std", "high"),
    "sentence_variety_low": ("sentence_length_std", "low"),
    "transition_density_high": ("transition_phrase_rate", "high"),
    "transition_density_low": ("transition_phrase_rate", "low"),
    "lexical_repetition_high": ("mtld", "low"),
    "lexical_diversity_high": ("mtld", "high"),
    "paragraph_consistency_high": ("paragraph_length_variance", "low"),
    "paragraph_consistency_low": ("paragraph_length_variance", "high"),
}

LOW_TAIL_THRESHOLD = 0.33
HIGH_TAIL_THRESHOLD = 0.67


def percentile_rank(value: float, reference_values: np.ndarray) -> float:
    reference_values = np.asarray(reference_values, dtype=float)
    return float(np.mean(reference_values < value))


def verify_claim(text: str, claim_type: str, reference_values_by_feature: dict) -> dict:
    feature_name, direction = CLAIM_FEATURE_MAP[claim_type]
    value = TIER1_FEATURES[feature_name](text)
    pct = percentile_rank(value, reference_values_by_feature[feature_name])
    supported = pct >= HIGH_TAIL_THRESHOLD if direction == "high" else pct <= LOW_TAIL_THRESHOLD
    return {
        "claim_type": claim_type, "feature": feature_name, "direction": direction,
        "value": value, "percentile": pct, "supported": supported,
    }
