"""Experiment 4's manual light-edit condition, run separately from
scripts/experiment4_raid_robustness.py.

`RESEARCH_PLAN.md`, "4. Does it survive paraphrasing?": "Run the RAID
adversarial subsets, plus a manual light-edit condition (vary sentence
lengths, add contractions) simulating a student who has been told what
detectors look for. If detection collapses under light editing, the panel
predominantly catches honest students who happen to write uniformly."

This arm was not run when Experiment 4 was first reported -- logged and
disclosed in `AMENDMENTS.md`, then run rather than left disclosed-but-
dropped. It reuses the exact same data already on disk
(`data/raid_abstracts_subset.csv`, produced by
`scripts/experiment4_raid_robustness.py`) and the same frozen composite and
threshold, so no new streaming or model calls are needed. The light-edit
transformation itself (`src/light_edit.py`) is deterministic pure Python:
merge every third sentence into the next with ", and " (mechanically
raising sentence-length variance) and apply a fixed contraction map --
a crude, mechanical edit, not a paraphrase-tool rewrite, matching what a
student following a surface-level tip sheet would actually do by hand.

Scope: same single-model limitation as the main RAID experiment --
llama-chat is the only AI model that surfaced within that experiment's
row cap for the abstracts/greedy/no-repetition-penalty filter.

Usage: uv run python scripts/experiment4b_light_edit.py
Requires: data/raid_abstracts_subset.csv, results/experiment3_frozen_composite.joblib
Output: results/experiment4b_light_edit.csv, results/experiment4b_light_edit_manifest.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src import features as feat
from src.light_edit import light_edit
from src.stats_utils import bootstrap_stat_ci, rate_at_threshold

RAID_PATH = Path("data/raid_abstracts_subset.csv")
FROZEN_MODEL_PATH = Path("results/experiment3_frozen_composite.joblib")
RESULTS_DIR = Path("results")
SEED = 42
N_BOOT = 1000
MODEL = "llama-chat"


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({name: texts.apply(fn) for name, fn in feat.TIER1_FEATURES.items()}, index=texts.index)


def score(texts: pd.Series, frozen: dict):
    X = compute_feature_matrix(texts).to_numpy()
    Xs = frozen["scaler"].transform(X)
    return frozen["model"].predict_proba(Xs)[:, 1]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RAID_PATH)
    frozen = joblib.load(FROZEN_MODEL_PATH)  # our own artifact from experiment3_separation.py, not external/untrusted
    threshold = frozen["threshold"]

    ai_baseline = df[(df["model"] == MODEL) & (df["attack"] == "none")].copy()
    human_baseline = df[(df["model"] == "human") & (df["attack"] == "none")].copy()
    print(f"AI baseline ({MODEL}, none): n={len(ai_baseline)}")
    print(f"Human baseline (none): n={len(human_baseline)}")

    ai_baseline["edited"] = ai_baseline["generation"].apply(light_edit)
    human_baseline["edited"] = human_baseline["generation"].apply(light_edit)

    ai_baseline["p_ai_score_before"] = score(ai_baseline["generation"], frozen)
    ai_baseline["p_ai_score_after"] = score(ai_baseline["edited"], frozen)
    human_baseline["p_ai_score_before"] = score(human_baseline["generation"], frozen)
    human_baseline["p_ai_score_after"] = score(human_baseline["edited"], frozen)

    rows = []
    for label, before_col, after_col, group in [
        ("AI (TPR)", "p_ai_score_before", "p_ai_score_after", ai_baseline),
        ("Human (FPR)", "p_ai_score_before", "p_ai_score_after", human_baseline),
    ]:
        rate_before, lo_before, hi_before = bootstrap_stat_ci(
            [group[before_col].to_numpy()], lambda s: rate_at_threshold(s, threshold), n_boot=N_BOOT, seed=SEED,
        )
        rate_after, lo_after, hi_after = bootstrap_stat_ci(
            [group[after_col].to_numpy()], lambda s: rate_at_threshold(s, threshold), n_boot=N_BOOT, seed=SEED,
        )
        delta, delta_lo, delta_hi = bootstrap_stat_ci(
            [group[before_col].to_numpy(), group[after_col].to_numpy()],
            lambda a, b: float(rate_at_threshold(b, threshold) - rate_at_threshold(a, threshold)),
            n_boot=N_BOOT, seed=SEED,
        )
        rows.append({
            "group": label, "n": len(group),
            "rate_before": rate_before, "rate_before_ci_low": lo_before, "rate_before_ci_high": hi_before,
            "rate_after": rate_after, "rate_after_ci_low": lo_after, "rate_after_ci_high": hi_after,
            "delta": delta, "delta_ci_low": delta_lo, "delta_ci_high": delta_hi,
        })
    result_df = pd.DataFrame(rows)
    print("\nBefore/after light edit:\n" + result_df.to_string(index=False))
    result_df.to_csv(RESULTS_DIR / "experiment4b_light_edit.csv", index=False)

    # Sanity check the transformation is doing what it claims on the actual
    # data used, not just on the unit-test toy examples.
    sld_before = ai_baseline["generation"].apply(feat.TIER1_FEATURES["sentence_length_std"])
    sld_after = ai_baseline["edited"].apply(feat.TIER1_FEATURES["sentence_length_std"])
    print(f"\nMean sentence_length_std, AI baseline: before={sld_before.mean():.2f}, after={sld_after.mean():.2f}")

    ai_row = result_df[result_df["group"] == "AI (TPR)"].iloc[0]
    manifest = {
        "seed": SEED, "n_bootstrap": N_BOOT, "model": MODEL,
        "pre_registration": (
            "RESEARCH_PLAN.md Experiment 4: run the RAID adversarial subsets "
            "plus a manual light-edit condition (vary sentence lengths, add "
            "contractions). Not run when Experiment 4 was first reported; "
            "disclosed as dropped in AMENDMENTS.md, then run rather than left "
            "unrun -- this file is that run."
        ),
        "transformation": "src/light_edit.py::light_edit -- deterministic, no model, no randomness",
        "tpr_before": ai_row["rate_before"], "tpr_after": ai_row["rate_after"],
        "tpr_delta_pts": float((ai_row["rate_after"] - ai_row["rate_before"]) * 100),
        "tpr_delta_ci_pts": [float(ai_row["delta_ci_low"] * 100), float(ai_row["delta_ci_high"] * 100)],
        "mean_sentence_length_std_before": float(sld_before.mean()),
        "mean_sentence_length_std_after": float(sld_after.mean()),
        "single_model_limitation": (
            "Same scope as scripts/experiment4_raid_robustness.py: llama-chat is "
            "the only AI model available in the persisted RAID subset for this "
            "domain/decoding/repetition-penalty filter."
        ),
        "comparison_to_paraphrase": (
            "For context against results/experiment4_tpr_by_attack.csv: none "
            "(baseline) TPR is 0.523, paraphrase TPR is 0.308 (-21.5pts) for "
            "the same model. This light-edit figure is directly comparable -- "
            "same baseline documents, same frozen threshold."
        ),
    }
    with open(RESULTS_DIR / "experiment4b_light_edit_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment4b_light_edit_manifest.json'}")


if __name__ == "__main__":
    main()
