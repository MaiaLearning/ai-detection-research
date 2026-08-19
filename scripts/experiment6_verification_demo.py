"""Experiment 6, second deliverable: feedback verification harness demo.

Independent of whether the Tier 1 features predict quality (they showed no
incremental value above word count -- see experiment6_quality_composite.py),
they do measure what they claim to measure. This demonstrates checking a
review-feedback claim about an essay's writing against the actual measured
feature value and its percentile in a reference distribution (here, the
full cleaned PERSUADE corpus).

This is a demonstration harness, not a benchmark: it shows the mechanism on
a few real essays with hand-written example claims (a mix of claims that
should verify as supported and ones that should verify as unsupported),
rather than evaluating real production review feedback (which doesn't
exist in this offline study).

Usage: uv run python scripts/experiment6_verification_demo.py
Output: results/experiment6_verification_demo.csv
"""
import json
from pathlib import Path

import pandas as pd

from src.data import load_and_clean
from src.feedback_verification import CLAIM_FEATURE_MAP, verify_claim
from src.features import TIER1_FEATURES

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
RESULTS_DIR = Path("results")
N_DEMO_ESSAYS = 5
SEED = 42

# One hand-written example claim per demo essay, chosen to exercise both a
# claim that should verify as supported and one that shouldn't -- not
# real production feedback (this study has no access to that).
DEMO_CLAIMS = [
    "sentence_variety_high", "sentence_variety_low", "transition_density_high",
    "lexical_repetition_high", "paragraph_consistency_high",
]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    human, _ = load_and_clean(PERSUADE_PATH)

    reference_values_by_feature = {
        name: human["full_text"].apply(fn).to_numpy() for name, fn in TIER1_FEATURES.items()
    }

    demo_essays = human.sample(n=N_DEMO_ESSAYS, random_state=SEED)
    rows = []
    for (_, essay), claim_type in zip(demo_essays.iterrows(), DEMO_CLAIMS):
        result = verify_claim(essay["full_text"], claim_type, reference_values_by_feature)
        rows.append({
            "essay_id": essay["essay_id_comp"], "claim_type": claim_type,
            "feature": result["feature"], "feature_value": result["value"],
            "percentile_in_corpus": result["percentile"], "claim_supported": result["supported"],
            "essay_snippet": essay["full_text"][:100],
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "experiment6_verification_demo.csv", index=False)
    print("Feedback verification harness demo:\n" + results_df.drop(columns=["essay_snippet"]).to_string(index=False))
    print(f"\nAll {len(CLAIM_FEATURE_MAP)} claim types map to a real Tier 1 feature "
          f"(see src/feedback_verification.py:CLAIM_FEATURE_MAP for the full list).")
    print(f"Results written to {RESULTS_DIR / 'experiment6_verification_demo.csv'}")


if __name__ == "__main__":
    main()
