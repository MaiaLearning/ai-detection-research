"""Within-Claude TPR by task type (independent vs. text-dependent prompts) --
the confound check reported in TECHNICAL_REPORT.md Section 5.6.

Not previously persisted to results/: the figures that shipped in an earlier
draft (75.5% vs 70.1%) lived only in the docstring of the shelved
scripts/experiment5_prompt_effort.py, computed before the OpenAI generations
were added to the training corpus -- the same "stale pre-refresh number"
issue found and fixed for the DAIGT Claude vintages in Section 5.6's main
table. This script recomputes from the current, final experiment3_ai_scores.csv
and experiment3_separation.csv (the same OOF scores and threshold everything
else in Section 5.6 uses) and persists the result so it is reproducible like
every other figure in this study, and discloses the rows with no task label
rather than silently dropping them.

Usage: uv run python scripts/analyze_claude_task_split.py
Requires: results/experiment3_ai_scores.csv, results/experiment3_separation.csv
Output: results/experiment3_claude_task_split.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stats_utils import bootstrap_stat_ci, rate_at_threshold

AI_SCORES_PATH = Path("results/experiment3_ai_scores.csv")
SEPARATION_PATH = Path("results/experiment3_separation.csv")
RESULTS_DIR = Path("results")
SEED = 42
N_BOOT = 1000


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(AI_SCORES_PATH)
    threshold = pd.read_csv(SEPARATION_PATH).set_index("metric").loc["threshold_at_target_fpr", "value"]

    claude = df[df["source"] == "claude_sonnet_5_bedrock"]
    n_total = len(claude)
    n_nan_task = int(claude["task"].isna().sum())
    print(f"claude_sonnet_5_bedrock: n={n_total}, {n_nan_task} rows have no task label "
          f"({n_nan_task / n_total:.1%}) -- reported separately, not silently dropped")

    rows = []
    for task in ["Independent", "Text dependent"]:
        sub = claude[claude["task"] == task]["p_ai_oof_score"].to_numpy()
        point, lo, hi = bootstrap_stat_ci(
            [sub], lambda a: rate_at_threshold(a, threshold), n_boot=N_BOOT, seed=SEED,
        )
        rows.append({"task": task, "n": len(sub), "tpr": point, "tpr_ci_low": lo, "tpr_ci_high": hi})

    nan_sub = claude[claude["task"].isna()]["p_ai_oof_score"].to_numpy()
    nan_point, nan_lo, nan_hi = bootstrap_stat_ci(
        [nan_sub], lambda a: rate_at_threshold(a, threshold), n_boot=N_BOOT, seed=SEED,
    )
    rows.append({"task": "(no task label)", "n": len(nan_sub), "tpr": nan_point, "tpr_ci_low": nan_lo, "tpr_ci_high": nan_hi})

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    result_df.to_csv(RESULTS_DIR / "experiment3_claude_task_split.csv", index=False)

    indep_tpr = result_df.loc[result_df["task"] == "Independent", "tpr"].iloc[0]
    dep_tpr = result_df.loc[result_df["task"] == "Text dependent", "tpr"].iloc[0]
    indep_scores = claude[claude["task"] == "Independent"]["p_ai_oof_score"].to_numpy()
    dep_scores = claude[claude["task"] == "Text dependent"]["p_ai_oof_score"].to_numpy()
    gap_point, gap_lo, gap_hi = bootstrap_stat_ci(
        [indep_scores, dep_scores],
        lambda a, b: float(rate_at_threshold(a, threshold) - rate_at_threshold(b, threshold)),
        n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nGap (Independent - Text dependent): {gap_point:.4f} ({gap_lo:.4f}, {gap_hi:.4f}) "
          f"= {gap_point*100:.1f}pp, CI [{gap_lo*100:.1f}, {gap_hi*100:.1f}]pp")

    manifest = {
        "seed": SEED, "n_bootstrap": N_BOOT,
        "note": (
            "Recomputed from results/experiment3_ai_scores.csv and the final "
            "threshold in results/experiment3_separation.csv -- the same OOF "
            "scores Section 5.6's main TPR table uses. An earlier draft cited "
            "75.5%/70.1% from scripts/experiment5_prompt_effort.py's docstring, "
            "computed before the OpenAI generations were folded into the "
            "training corpus; those figures are stale for the same reason the "
            "DAIGT Claude vintages needed correcting."
        ),
        "n_total_claude": n_total, "n_nan_task": n_nan_task,
        "independent_tpr": {"point": float(indep_tpr), "ci_low": float(result_df.loc[0, "tpr_ci_low"]), "ci_high": float(result_df.loc[0, "tpr_ci_high"])},
        "text_dependent_tpr": {"point": float(dep_tpr), "ci_low": float(result_df.loc[1, "tpr_ci_low"]), "ci_high": float(result_df.loc[1, "tpr_ci_high"])},
        "gap_pp": {"point": float(gap_point * 100), "ci_low": float(gap_lo * 100), "ci_high": float(gap_hi * 100)},
    }
    with open(RESULTS_DIR / "experiment3_claude_task_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment3_claude_task_split_manifest.json'}")


if __name__ == "__main__":
    main()
