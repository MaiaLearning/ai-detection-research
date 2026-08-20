"""Resolves an apparent contradiction between Gate 2 (partial Spearman rho =
+0.135: better essays score more AI-like) and the quality-bin-profile plot
(`scripts/analyze_quality_bin_profile.py`), whose ARITHMETIC MEANS per
quality bin appeared to run the opposite direction (score 1 elevated at
+0.063, score 6 the lowest at -0.012).

RESOLUTION, stated before any interpretation below: **the bin-profile
plot's use of arithmetic means was misleading. Gate 2's sign is correct
and reproduces exactly on the identical residualized values the bin plot
used.** The bug was in how that plot was read, not in Gate 2.

Four checks, run in the order raised:

1. Sign convention. Scored 10 known-human (PERSUADE) and 10 known-AI
   (DAIGT, label==1, excluding train_essays) essays through the frozen
   composite directly. Mean P(AI): human 0.132, AI 0.845 -- correct
   direction, consistent with (and a small-n version of) Experiment 3's
   AUC = 0.945 computed over the full corpus. Not the source of the
   discrepancy.

2. Same sample. Gate 2 and the bin plot both use
   `results/experiment3_human_scores.csv`, written once by
   experiment3_separation.py from the same `human` dataframe and the same
   `human_scores` array used for the Gate 2 computation in that script --
   not merely the same row count by coincidence, but the same rows by
   construction. Confirmed further below by recomputing Gate 2's exact
   statistic from that file.

3. Recompute Spearman on the exact residualized values in the bin plot.
   `partial_spearman(p_ai, quality, word_count)` recomputed from
   experiment3_human_scores.csv reproduces +0.1350 exactly, confirming (2).
   Separately: residualizing P(AI) on RAW word count via linear regression
   (exactly what the bin plot did) and computing a plain Spearman
   correlation between that residual and quality gives **+0.136** --
   matching Gate 2, not contradicting it. The rank relationship in the
   bin plot's own residualized values is positive. The arithmetic MEAN
   per bin is not.

4. Look at the scatter, not the means. The residualized P(AI) values are
   heavily right-skewed (skew 1.2 to 2.5 across bins, driven by a
   long tail of essays the composite scores as much more AI-like than
   their word count predicts) with unbalanced bin sizes (n=842 to
   n=7,965). A skewed variable's arithmetic mean is not a stable summary
   and is disproportionately moved by a handful of extreme values,
   especially in the smaller bins. The mean RANK PERCENTILE per bin (the
   quantity Spearman actually reflects) tells a materially different and
   much more monotonic story:

     score  mean rank percentile of residualized P(AI)
       1    0.521
       2    0.440   <- lowest
       3    0.481
       4    0.536
       5    0.557
       6    0.587   <- highest

   Scores 2 through 6 increase cleanly and monotonically. Score 1 sits
   slightly out of order (between scores 3 and 4, not at either extreme)
   rather than at the bottom the mean plot suggested, and not at the top
   either. This is the actual shape of Gate 2's finding: essays are more
   likely to read as more AI-like as quality rises across most of the
   range, with score 1 a mild, partial exception rather than a
   contradiction.

On the score-1 "structural degeneracy" hypothesis: word count for score-1
essays is indeed much shorter (median 249 words vs 824 for score 6), so a
partial version of that hypothesis has support. But note: no "5.7" or
comparable section discussing degenerate feature values under broken
tokenization was found anywhere in this repository (checked README.md,
AMENDMENTS.md, PRACTITIONER_BRIEF.md, all EXPERIMENT_*.md, and the
scripts) -- the closest existing finding in this repo is the RAID
homoglyph/zero-width-space ADVERSARIAL attacks breaking tokenization
(experiment 4, a different mechanism: deliberately injected unicode into
AI-generated text, not naturally short/weak human writing). If that
citation refers to a document outside this repository, it cannot be
verified from what's here and is not repeated as if it were.
Score-1's skew coefficient (1.20) is in fact the LOWEST of the six bins
(score 6's is highest, 2.46) -- its mean is elevated mainly because a
smaller n (1,024) gives a few large positive residuals more leverage over
the average, not because its distribution is unusually extreme relative
to its own spread.

Usage: uv run python scripts/analyze_gate2_sign_resolution.py
Requires: results/experiment3_frozen_composite.joblib, results/experiment3_human_scores.csv,
    data/persuade_2.0_human_scores_demo_id_github.csv, data/train_v2_drcat_02.csv
Output: results/experiment7_gate2_sign_resolution.json, results/experiment7_quality_bin_rank_profile.png
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from sklearn.linear_model import LinearRegression

from src import features as feat
from src.data import load_and_clean
from src.stats_utils import partial_spearman

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
DAIGT_PATH = Path("data/train_v2_drcat_02.csv")
FROZEN_COMPOSITE_PATH = Path("results/experiment3_frozen_composite.joblib")
HUMAN_SCORES_PATH = Path("results/experiment3_human_scores.csv")
RESULTS_DIR = Path("results")
SEED = 0  # for the sign-convention sampling check only (Check 1)
N_BOOT = 1000


def score_text(text, composite):
    vec = np.array([[feat.TIER1_FEATURES[name](text) for name in composite["feature_names"]]])
    return float(composite["model"].predict_proba(composite["scaler"].transform(vec))[0, 1])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    composite = joblib.load(FROZEN_COMPOSITE_PATH)  # our own artifact from experiment3_separation.py, not external/untrusted

    print("=== Check 1: sign convention on known examples ===")
    human, _ = load_and_clean(PERSUADE_PATH)
    daigt = pd.read_csv(DAIGT_PATH)
    daigt_ai = daigt[(daigt["label"] == 1) & (daigt["source"] != "train_essays")]
    human_sample = human.sample(10, random_state=SEED)["full_text"]
    ai_sample = daigt_ai.sample(10, random_state=SEED)["text"]
    human_scores = [score_text(t, composite) for t in human_sample]
    ai_scores = [score_text(t, composite) for t in ai_sample]
    print(f"Mean P(AI): human={np.mean(human_scores):.3f}, AI={np.mean(ai_scores):.3f} "
          f"(should be human << AI)")
    sign_ok = bool(np.mean(ai_scores) > np.mean(human_scores))
    print(f"Sign convention OK: {sign_ok} (cross-checked against Experiment 3's full-corpus AUC=0.945)")

    print("\n=== Checks 2-3: same sample, recompute Spearman on the bin plot's own residuals ===")
    df = pd.read_csv(HUMAN_SCORES_PATH)
    p_ai = df["p_ai_oof_score"].to_numpy(dtype=float)
    quality = df["holistic_essay_score"].to_numpy(dtype=int)
    wc = df["word_count"].to_numpy(dtype=float)

    gate2_recomputed = partial_spearman(p_ai, quality.astype(float), wc)
    print(f"Gate 2's partial Spearman, recomputed from experiment3_human_scores.csv: {gate2_recomputed:.4f} "
          f"(originally reported: +0.135)")

    reg = LinearRegression().fit(wc.reshape(-1, 1), p_ai)
    resid = p_ai - reg.predict(wc.reshape(-1, 1))
    spearman_of_bin_plot_residuals = float(spearmanr(resid, quality).statistic)
    print(f"Spearman(bin plot's residualized P(AI), quality) = {spearman_of_bin_plot_residuals:.4f} "
          f"-- matches Gate 2's sign and magnitude; the bin plot's underlying rank relationship is positive")

    print("\n=== Check 4: rank percentile per bin (what Spearman reflects) vs the arithmetic mean ===")
    ranks_pct = (pd.Series(resid).rank().to_numpy() - 1) / (len(resid) - 1)
    levels = sorted(np.unique(quality))

    print("Bootstrapping CIs on mean rank percentile per bin (resampling the full "
          f"n={len(resid)} sample jointly, so bin membership and rank both vary together)...")
    rng = np.random.default_rng(42)  # project-wide bootstrap seed convention; distinct from Check 1's SEED=0
    n = len(resid)
    boot_by_level = {lvl: [] for lvl in levels}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ranks_boot = (pd.Series(resid[idx]).rank().to_numpy() - 1) / (n - 1)
        q_boot = quality[idx]
        for lvl in levels:
            mask = q_boot == lvl
            if mask.any():
                boot_by_level[lvl].append(float(ranks_boot[mask].mean()))
    rank_pct_ci = {
        lvl: (float(np.percentile(boot_by_level[lvl], 2.5)), float(np.percentile(boot_by_level[lvl], 97.5)))
        for lvl in levels
    }

    rows = []
    for level in levels:
        mask = quality == level
        r = resid[mask]
        ci_lo, ci_hi = rank_pct_ci[level]
        rows.append({
            "holistic_essay_score": int(level), "n": int(mask.sum()),
            "mean_residual_p_ai": float(r.mean()), "median_residual_p_ai": float(np.median(r)),
            "mean_rank_percentile": float(ranks_pct[mask].mean()),
            "mean_rank_percentile_ci_low": ci_lo, "mean_rank_percentile_ci_high": ci_hi,
            "skew_residual_p_ai": float(skew(r)),
            "word_count_median": float(np.median(wc[mask])),
        })
    bin_df = pd.DataFrame(rows)
    print(bin_df.to_string(index=False))
    bin_df.to_csv(RESULTS_DIR / "experiment7_quality_bin_rank_profile.csv", index=False)

    plot_rank_profile(bin_df)

    manifest = {
        "seed": SEED,
        "check1_sign_convention": {
            "mean_p_ai_human_sample": float(np.mean(human_scores)),
            "mean_p_ai_ai_sample": float(np.mean(ai_scores)),
            "sign_ok": sign_ok,
            "note": "Cross-checked against experiment 3's full-corpus AUC=0.945 (human=0, AI=1, P(AI)=predict_proba[:,1]).",
        },
        "check2_same_sample": {
            "note": (
                "results/experiment3_human_scores.csv is written from the SAME "
                "`human` dataframe and `human_scores` array Gate 2 is computed "
                "from, within experiment3_separation.py -- same rows by "
                "construction, confirmed below by recomputing Gate 2's exact "
                "statistic from that file."
            ),
        },
        "check3_recompute": {
            "gate2_partial_spearman_recomputed": gate2_recomputed,
            "gate2_partial_spearman_original": 0.135,
            "spearman_of_bin_plot_raw_linear_residuals_vs_quality": spearman_of_bin_plot_residuals,
            "conclusion": (
                "The bin plot's own residualized values, correlated by rank "
                "against quality, reproduce Gate 2's positive sign and "
                "magnitude. There is no contradiction in the underlying "
                "statistic -- the arithmetic-mean bin visualization in "
                "analyze_quality_bin_profile.py was misleading, not Gate 2."
            ),
        },
        "check4_why_means_misled": {
            "per_bin": rows,
            "note": (
                "Residualized P(AI) is heavily right-skewed (skew 1.2-2.5) "
                "with unbalanced bin sizes (n=842-7965); arithmetic means are "
                "not robust to this and are dominated by a handful of large "
                "positive outliers, especially in smaller bins. Mean rank "
                "percentile per bin -- what Spearman reflects -- increases "
                "close to monotonically from score 2 through score 6; score 1 "
                "is a mild, partial exception (rank percentile between scores "
                "3 and 4), not the bottom-of-range effect the mean plot "
                "appeared to show."
            ),
        },
        "score1_degeneracy_note": (
            "Score-1 essays are shorter (median word count 249 vs 824 for "
            "score 6), a partial match for a length/degeneracy hypothesis. "
            "No '5.7' or similar section on degenerate feature values under "
            "broken tokenization was found anywhere in this repository "
            "(README.md, AMENDMENTS.md, PRACTITIONER_BRIEF.md, "
            "EXPERIMENT_*.md, scripts/ were checked) -- the closest existing "
            "finding here is RAID's homoglyph/zero-width-space adversarial "
            "attacks breaking tokenization (experiment 4), a different "
            "mechanism (deliberately injected unicode in AI-generated text, "
            "not naturally short human writing). If that citation is from a "
            "document outside this repository, it is not verifiable from "
            "what's here and is not repeated as fact. Score 1's own residual "
            "distribution has the LOWEST skew of the six bins (1.20 vs 2.46 "
            "for score 6), which cuts against 'score 1 is uniquely extreme' "
            "-- its elevated mean looks more like ordinary outlier leverage "
            "in a smaller bin than a distinct degeneracy regime."
        ),
    }
    with open(RESULTS_DIR / "experiment7_gate2_sign_resolution.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment7_gate2_sign_resolution.json'}")


def plot_rank_profile(bin_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    x = bin_df["holistic_essay_score"].to_numpy()

    ax1.plot(x, bin_df["mean_rank_percentile"], "o-", color="black")
    ax1.set_xlabel("PERSUADE holistic essay score (1-6)")
    ax1.set_ylabel("Mean rank percentile of residualized P(AI)")
    ax1.set_title("What Spearman reflects: mean rank percentile\n(reproduces Gate 2's positive sign)")
    ax1.axhline(0.5, linestyle=":", color="gray")

    ax2.plot(x, bin_df["mean_residual_p_ai"], "o--", color="firebrick", label="mean (misleading: skew-sensitive)")
    ax2.plot(x, bin_df["median_residual_p_ai"], "o-", color="black", label="median (robust to skew)")
    ax2.set_xlabel("PERSUADE holistic essay score (1-6)")
    ax2.set_ylabel("Residualized P(AI) score")
    ax2.set_title("Why the earlier mean-only plot misled:\nmean vs median per bin")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment7_quality_bin_rank_profile.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment7_quality_bin_rank_profile.png'}")


if __name__ == "__main__":
    main()
