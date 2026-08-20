"""Where in the quality range does the composite's positive quality
correlation live -- spread evenly, or concentrated at the top?

Follow-up to the Pearson-vs-Spearman conditioning check
(`scripts/analyze_discriminant_conditioning.py`, logged in AMENDMENTS.md
item 6): Gate 2's partial Spearman rho (+0.135, CI 0.123-0.148) and the
Pearson partial on the identical sample (-0.027, CI -0.046 to 0.002) are
both tight, don't overlap, and have opposite signs. That combination is
what a threshold/tail effect looks like -- a monotone relationship
concentrated in part of the distribution (most plausibly the top, where a
detectability penalty on the strongest essays would matter most for
admissions review) rather than a relationship spread evenly across it, so
a rank statistic (Spearman) picks it up and a linear estimator (Pearson,
which weights the whole range equally) does not. Not in the original plan
or EXPERIMENT_7.md; uses only data already on disk.

Method: bin PERSUADE's human essays by their integer holistic_essay_score
(1-6, the corpus's native scale -- no quantile binning needed, each level
already has a few hundred to several thousand essays). Within each bin,
report both the RAW mean composite P(AI) score and the mean AFTER removing
each essay's word-count-predicted component (a linear regression of
P(AI) on word count, fit on the full human sample -- the same conditioning
variable and the same linear-in-word-count assumption Gate 2's Pearson
partial correlation used, so this plot and that correlation are asking the
same underlying question, just with the distribution visible instead of
collapsed to one number). If the word-count-controlled curve is flat
across the lower/middle bins and rises at the top, that is a specific,
defensible mechanism claim -- the penalty falls on the strongest essays,
not proportionally across the range -- and a materially different, more
actionable finding than any single correlation coefficient.

Usage: uv run python scripts/analyze_quality_bin_profile.py
Requires: results/experiment3_human_scores.csv (from experiment3_separation.py)
Output: results/experiment7_quality_bin_profile.csv, results/experiment7_quality_bin_profile.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.stats_utils import bootstrap_stat_ci

HUMAN_SCORES_PATH = Path("results/experiment3_human_scores.csv")
RESULTS_DIR = Path("results")

SEED = 42
N_BOOT = 1000


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(HUMAN_SCORES_PATH)
    p_ai = df["p_ai_oof_score"].to_numpy(dtype=float)
    word_count = df["word_count"].to_numpy(dtype=float)
    score = df["holistic_essay_score"].to_numpy(dtype=int)

    # Word-count residualization: same conditioning variable (raw word
    # count, not log) and same linear assumption as the Pearson partial
    # correlation this plot follows up on, fit on the full human sample.
    reg = LinearRegression().fit(word_count.reshape(-1, 1), p_ai)
    residual_p_ai = p_ai - reg.predict(word_count.reshape(-1, 1))

    rows = []
    for level in sorted(np.unique(score)):
        mask = score == level
        n = int(mask.sum())
        raw_point, raw_lo, raw_hi = bootstrap_stat_ci(
            [p_ai[mask]], lambda a: float(np.mean(a)), n_boot=N_BOOT, seed=SEED,
        )
        resid_point, resid_lo, resid_hi = bootstrap_stat_ci(
            [residual_p_ai[mask]], lambda a: float(np.mean(a)), n_boot=N_BOOT, seed=SEED,
        )
        rows.append({
            "holistic_essay_score": int(level), "n": n,
            "mean_p_ai_raw": raw_point, "mean_p_ai_raw_ci_low": raw_lo, "mean_p_ai_raw_ci_high": raw_hi,
            "mean_p_ai_word_count_controlled": resid_point,
            "mean_p_ai_word_count_controlled_ci_low": resid_lo,
            "mean_p_ai_word_count_controlled_ci_high": resid_hi,
        })
    bin_df = pd.DataFrame(rows)
    bin_df.to_csv(RESULTS_DIR / "experiment7_quality_bin_profile.csv", index=False)
    print("Mean composite P(AI) by holistic quality score (human essays only):")
    print(bin_df.to_string(index=False))

    # Whether the rise is concentrated at the top: compare the controlled
    # mean's move from score 5->6 against its move across all lower steps.
    controlled = bin_df["mean_p_ai_word_count_controlled"].to_numpy()
    steps = np.diff(controlled)
    print(f"\nStep-to-step change in word-count-controlled mean P(AI), score 1->2 ... 5->6: "
          f"{np.round(steps, 4).tolist()}")
    if len(steps) >= 2:
        top_step = steps[-1]
        other_steps_mean_abs = float(np.mean(np.abs(steps[:-1])))
        print(f"Top step (score {int(bin_df['holistic_essay_score'].iloc[-2])}->"
              f"{int(bin_df['holistic_essay_score'].iloc[-1])}): {top_step:.4f}. "
              f"Mean |step| over the rest of the range: {other_steps_mean_abs:.4f}.")

    plot_bin_profile(bin_df)


def plot_bin_profile(bin_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    x = bin_df["holistic_essay_score"].to_numpy()

    raw = bin_df["mean_p_ai_raw"].to_numpy()
    raw_err = np.vstack([
        raw - bin_df["mean_p_ai_raw_ci_low"].to_numpy(),
        bin_df["mean_p_ai_raw_ci_high"].to_numpy() - raw,
    ])
    ax.errorbar(x, raw, yerr=raw_err, fmt="o--", color="gray", alpha=0.6, label="raw mean P(AI) (not controlled)")

    controlled = bin_df["mean_p_ai_word_count_controlled"].to_numpy()
    controlled_err = np.vstack([
        controlled - bin_df["mean_p_ai_word_count_controlled_ci_low"].to_numpy(),
        bin_df["mean_p_ai_word_count_controlled_ci_high"].to_numpy() - controlled,
    ])
    ax.errorbar(x, controlled, yerr=controlled_err, fmt="o-", color="black", linewidth=2,
                label="word-count-controlled mean P(AI)")

    ax.set_xlabel("PERSUADE holistic essay score (1-6)")
    ax.set_ylabel("Mean composite P(AI) score")
    ax.set_title("Composite P(AI) by essay quality, word-count controlled\n(bootstrap 95% CI per bin)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment7_quality_bin_profile.png", dpi=150)
    print(f"\nPlot written to {RESULTS_DIR / 'experiment7_quality_bin_profile.png'}")


if __name__ == "__main__":
    main()
