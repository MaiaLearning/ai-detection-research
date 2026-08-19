"""Experiment 2 (GATE): Is the score anti-correlated with essay quality?

RESEARCH_PLAN.md, "2. Is the score anti-correlated with essay quality? (GATE)"

Same human-only PERSUADE 2.0 sample as Experiment 1's cleaning criteria —
here we use the *full* cleaned analytic sample (not the ELL-matching
subsample from Experiment 1), since the confound to control here is word
count, not language background, and using the full sample gives tighter
CIs. This interpretation is logged in the manifest for auditability.

For each feature we report:
  - raw Spearman correlation with holistic_essay_score
  - partial Spearman correlation controlling for word count (PERSUADE
    holistic scores correlate with essay length, and several Tier 1
    features are themselves length-confounded — the plan explicitly
    calls this out)
Both with bootstrap 95% CIs. A feature fails the gate if its partial
correlation is negative, "meaningful" by Cohen's small-effect convention
(<= -0.1), and the CI excludes zero.

Usage: uv run python scripts/experiment2_quality_gate.py
Output: results/experiment2_correlations.csv, results/experiment2_correlations.png,
        results/experiment2_manifest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat
from src.data import load_and_clean, sha256_of
from src.stats_utils import bootstrap_corr_ci, bootstrap_stat_ci, partial_spearman

SEED = 42
N_BOOT = 1000
CORPUS_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
RESULTS_DIR = Path("results")
QUALITY_FAIL_RHO = -0.1  # Cohen: |r| >= 0.1 is a "small" effect, the floor for "meaningful"

FEATURE_FUNCS = feat.TIER1_FEATURES

# Features that failed this gate outright (see per-feature results below);
# excluded from the combined-model check run after the per-feature pass, to
# see whether the *recommended* reduced feature set is clean as a group.
DROPPED_DUE_TO_QUALITY_ANTICORRELATION = ["sentence_length_std", "type_token_ratio"]


def spearman_stat(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def compute_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, fn in FEATURE_FUNCS.items():
        out[name] = df["full_text"].apply(fn)
    return pd.DataFrame(out, index=df.index)


def evaluate_against_quality(x, score, word_count) -> dict:
    raw_point, raw_lo, raw_hi = bootstrap_corr_ci(x, score, spearman_stat, n_boot=N_BOOT, seed=SEED)
    rho_feat_wc = spearman_stat(x, word_count)
    partial_point, partial_lo, partial_hi = bootstrap_stat_ci(
        [x, score, word_count], partial_spearman, n_boot=N_BOOT, seed=SEED,
    )
    is_meaningful_negative = partial_point <= QUALITY_FAIL_RHO and partial_hi < 0
    is_borderline = (not is_meaningful_negative) and partial_point <= QUALITY_FAIL_RHO
    verdict = "FAIL (anti-correlated with quality)" if is_meaningful_negative else (
        "borderline" if is_borderline else "pass"
    )
    return {
        "raw_rho": raw_point, "raw_ci_low": raw_lo, "raw_ci_high": raw_hi,
        "rho_vs_word_count": rho_feat_wc,
        "partial_rho": partial_point, "partial_ci_low": partial_lo, "partial_ci_high": partial_hi,
        "verdict": verdict,
    }


def combined_model_oof_prediction(features_df: pd.DataFrame, feature_cols: list, score: np.ndarray) -> np.ndarray:
    # holistic_essay_score has only 6 distinct values, so stratifying on it
    # directly (like Experiment 1's stratification on ELL status) keeps
    # quality bins balanced across folds.
    X = StandardScaler().fit_transform(features_df[feature_cols].to_numpy())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(Ridge(random_state=SEED), X, score, cv=cv)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    corpus_hash = sha256_of(CORPUS_PATH)

    df, manifest = load_and_clean(CORPUS_PATH)
    print(f"Analytic sample: {len(df)} essays (of {manifest['n_total_rows']} total)")

    features_df = compute_feature_matrix(df)
    score = df["holistic_essay_score"].to_numpy(dtype=float)
    word_count = df["word_count"].to_numpy(dtype=float)

    print(f"rho(word_count, holistic_score) = {spearman_stat(word_count, score):.3f}  "
          f"(context: how much of the raw correlation length alone could explain)")

    rows = []
    for col in features_df.columns:
        x = features_df[col].to_numpy(dtype=float)
        row = evaluate_against_quality(x, score, word_count)
        rows.append({"feature": col, "n": len(df), **row})

    results_df = pd.DataFrame(rows)

    reduced_features = [c for c in FEATURE_FUNCS if c not in DROPPED_DUE_TO_QUALITY_ANTICORRELATION]
    oof_pred = combined_model_oof_prediction(features_df, reduced_features, score)
    combined_row = evaluate_against_quality(oof_pred, score, word_count)
    combined_label = (
        f"combined_model (Ridge, 5-fold CV, drops {', '.join(DROPPED_DUE_TO_QUALITY_ANTICORRELATION)})"
    )
    results_df = pd.concat([results_df, pd.DataFrame([{"feature": combined_label, "n": len(df), **combined_row}])], ignore_index=True)

    results_df.to_csv(RESULTS_DIR / "experiment2_correlations.csv", index=False)
    print("\n" + results_df.to_string(index=False))

    plot_results(results_df)

    failing_features = [r["feature"] for r in rows if r["verdict"].startswith("FAIL")]
    manifest.update({
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_hash,
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "sample_definition": (
            "Full cleaned PERSUADE analytic sample (same filter as Experiment 1: "
            "valid ell_status/grade_level/holistic_essay_score/word_count/prompt_name), "
            "NOT restricted to Experiment 1's ELL-matching subsample. Word count is "
            "controlled via partial Spearman correlation instead, since it is the "
            "confound this gate specifically calls out."
        ),
        "quality_fail_rho_threshold": QUALITY_FAIL_RHO,
        "rho_word_count_vs_holistic_score": spearman_stat(word_count, score),
        "features": list(FEATURE_FUNCS.keys()),
        "full_feature_set_gate_verdict": "FAIL" if failing_features else "PASS",
        "features_failing_full_set_gate": failing_features,
        "reduced_feature_set": reduced_features,
        "reduced_feature_set_combined_verdict": (
            "PASS" if combined_row["verdict"] == "pass" else combined_row["verdict"].upper()
        ),
        "note": (
            "RESEARCH_PLAN.md's literal decision rule is 'Fail 1 or 2 -> no scoring "
            "panel. Report and stop.' sentence_length_std and type_token_ratio fail "
            "outright on the full 9-feature set. Per user direction, we additionally "
            "checked whether a combined model over the remaining 7 features is clean "
            "as a group before deciding whether to proceed to Experiments 3/4."
        ),
    })
    with open(RESULTS_DIR / "experiment2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment2_manifest.json'}")
    print(f"Full 9-feature gate verdict: {manifest['full_feature_set_gate_verdict']} "
          f"(failing: {failing_features})")
    print(f"Reduced 7-feature combined-model verdict: {manifest['reduced_feature_set_combined_verdict']}")


def plot_results(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(results_df) + 2))
    y_pos = np.arange(len(results_df))
    errors = np.vstack([
        results_df["partial_rho"] - results_df["partial_ci_low"],
        results_df["partial_ci_high"] - results_df["partial_rho"],
    ])
    ax.errorbar(results_df["partial_rho"], y_pos, xerr=errors, fmt="o", color="black", capsize=3)
    ax.set_yticks(y_pos)
    labels = [f[:28] + "…" if len(f) > 28 else f for f in results_df["feature"]]
    ax.set_yticklabels(labels)
    ax.axvline(0.0, linestyle="--", color="gray", label="no correlation (0.0)")
    ax.axvline(-0.1, linestyle="--", color="red", label="gate fail threshold (-0.1)")
    ax.set_xlabel("Partial Spearman rho vs holistic quality score (controlling for word count, 95% bootstrap CI)")
    ax.set_title("Experiment 2 (GATE): is each feature anti-correlated with essay quality?")
    lo_bound = min(-0.3, results_df["partial_ci_low"].min() - 0.02)
    hi_bound = max(0.3, results_df["partial_ci_high"].max() + 0.02)
    ax.set_xlim(lo_bound, hi_bound)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment2_correlations.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment2_correlations.png'}")


if __name__ == "__main__":
    main()
