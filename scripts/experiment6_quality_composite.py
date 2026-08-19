"""Experiment 6: Do the Tier-1 features carry essay quality above word count?

EXPERIMENT_6.md. The last experiment in the study (Experiment 5 shelved --
see that file's closing note). PERSUADE human essays only, no AI text.

The deliverable is INCREMENTAL predictive power over word count alone, not
raw R²: raw rho(word_count, holistic_score) = 0.76, so any quality model
built on these features will look impressive and be a length proxy unless
the comparison is nested against a word-count-only baseline.

Three nested models, same folds, same seed, fit via RidgeCV (interpretable
coefficients matter more here than squeezing out performance, since the
coefficient signs are what would condition a feedback prompt):
  M0 -- word_count only
  M1 -- word_count + 8 features (drops raw type_token_ratio; Experiment 2
        showed it carries a residual length confound partial correlation
        doesn't fully scrub -- MTLD is the length-normalized replacement)
  M2 -- the 8 features only, no word_count (diagnostic: how much of the
        features' apparent signal is just proxying length)

Validation: split into DEV (80%) and a genuinely untouched HELD-OUT (20%),
stratified by ELL status so both are representative. Primary results are
5-fold CV out-of-fold predictions within DEV (RidgeCV re-tunes alpha inside
each fold). The HELD-OUT set is scored once, at the end, with models fit on
all of DEV, as a final unseen-data confirmation -- not used for any
tuning or selection decision.

Pre-registered expectation (recorded before running): incremental rho of
~0.2-0.3 over length. If the observed delta comes back far above that,
suspect leakage before celebrating (per the document's own instruction).

Usage: uv run python scripts/experiment6_quality_composite.py
Output: results/experiment6_nested.csv, results/experiment6_subgroup_deltas.csv,
        results/experiment6_feature_directions.csv, results/experiment6_manifest.json,
        results/experiment6_deltas.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from src import features as feat
from src.data import load_and_clean
from src.stats_utils import bootstrap_paired_delta_ci, bootstrap_stat_ci

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
RESULTS_DIR = Path("results")
SEED = 42
N_BOOT = 2000
N_FOLDS = 5
HELD_OUT_FRAC = 0.20
PRE_REGISTERED_EXPECTATION = "incremental rho of ~0.2-0.3 over word-count alone"

# Drops raw type_token_ratio (residual length confound per Experiment 2);
# MTLD is the length-normalized replacement, already one of the Tier 1 nine.
QUALITY_FEATURES = [
    "sentence_length_std", "mean_sentence_length", "mtld", "transition_phrase_rate",
    "paragraph_length_variance", "punctuation_variety", "contraction_rate", "function_word_entropy",
]
FEATURE_DIRECTION_HINTS = {
    "sentence_length_std": "higher = more erratic sentence lengths ('burstiness')",
    "mean_sentence_length": "higher = longer average sentences",
    "mtld": "higher = more lexically diverse (length-normalized)",
    "transition_phrase_rate": "higher = more transition/discourse markers per 100 words",
    "paragraph_length_variance": "higher = less consistent paragraph lengths",
    "punctuation_variety": "higher = more distinct punctuation marks used",
    "contraction_rate": "higher = more contractions/colloquialisms per 100 words",
    "function_word_entropy": "higher = more even distribution across function words",
}


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    fns = {name: feat.TIER1_FEATURES[name] for name in QUALITY_FEATURES}
    return pd.DataFrame({name: texts.apply(fn) for name, fn in fns.items()}, index=texts.index)


def fit_oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    return cross_val_predict(model, Xs, y, cv=cv)


def r2_and_rho(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"r2": r2_score(y_true, y_pred), "rho": float(spearmanr(y_true, y_pred).statistic)}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    human, manifest = load_and_clean(PERSUADE_PATH)
    print(f"Human essays (PERSUADE, cleaned): {len(human)}")

    dev, held_out = train_test_split(
        human, test_size=HELD_OUT_FRAC, random_state=SEED, stratify=human["ell_clean"]
    )
    print(f"Dev: {len(dev)}, held-out: {len(held_out)} (stratified by ELL status)")

    dev_feats = compute_feature_matrix(dev["full_text"])
    dev_wc = dev["word_count"].to_numpy(dtype=float).reshape(-1, 1)
    dev_y = dev["holistic_essay_score"].to_numpy(dtype=float)
    X0 = dev_wc
    X1 = np.hstack([dev_wc, dev_feats.to_numpy()])
    X2 = dev_feats.to_numpy()

    print("Fitting M0/M1/M2 via 5-fold CV (RidgeCV re-tuned per fold)...")
    pred0 = fit_oof(X0, dev_y, SEED)
    pred1 = fit_oof(X1, dev_y, SEED)
    pred2 = fit_oof(X2, dev_y, SEED)

    rows = []
    for name, pred in [("M0_word_count_only", pred0), ("M1_word_count_plus_features", pred1),
                        ("M2_features_only", pred2)]:
        metrics = r2_and_rho(dev_y, pred)
        r2_pt, r2_lo, r2_hi = bootstrap_stat_ci([dev_y, pred], lambda yt, yp: r2_score(yt, yp), n_boot=N_BOOT, seed=SEED)
        rho_pt, rho_lo, rho_hi = bootstrap_stat_ci(
            [dev_y, pred], lambda yt, yp: float(spearmanr(yt, yp).statistic), n_boot=N_BOOT, seed=SEED,
        )
        rows.append({
            "model": name, "n": len(dev_y),
            "r2": r2_pt, "r2_ci_low": r2_lo, "r2_ci_high": r2_hi,
            "rho": rho_pt, "rho_ci_low": rho_lo, "rho_ci_high": rho_hi,
        })
    results_df = pd.DataFrame(rows)

    delta_rho, delta_rho_lo, delta_rho_hi = bootstrap_paired_delta_ci(
        dev_y, pred0, pred1, metric_fn=lambda yt, yp: float(spearmanr(yt, yp).statistic),
        n_boot=N_BOOT, seed=SEED,
    )
    delta_r2, delta_r2_lo, delta_r2_hi = bootstrap_paired_delta_ci(
        dev_y, pred0, pred1, metric_fn=r2_score, n_boot=N_BOOT, seed=SEED,
    )
    print("\nNested model comparison (dev, 5-fold OOF):\n" + results_df.to_string(index=False))
    print(f"\nM1 - M0 delta rho: {delta_rho:.4f} ({delta_rho_lo:.4f}, {delta_rho_hi:.4f})")
    print(f"M1 - M0 delta R²:  {delta_r2:.4f} ({delta_r2_lo:.4f}, {delta_r2_hi:.4f})")
    print(f"Pre-registered expectation: {PRE_REGISTERED_EXPECTATION}")

    # Secondary check: does gradient boosting leave meaningful non-linearity on the table?
    Xs1 = StandardScaler().fit_transform(X1)
    gbm_pred1 = cross_val_predict(
        HistGradientBoostingRegressor(random_state=SEED), Xs1, dev_y,
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
    )
    gbm_metrics = r2_and_rho(dev_y, gbm_pred1)
    print(f"\nGBM (M1 features, secondary check): R²={gbm_metrics['r2']:.4f}, rho={gbm_metrics['rho']:.4f} "
          f"(ridge M1: R²={r2_and_rho(dev_y, pred1)['r2']:.4f}, rho={r2_and_rho(dev_y, pred1)['rho']:.4f})")

    # Subgroup deltas (ELL/non-ELL, grade band) -- reuse the same OOF predictions
    subgroup_rows = []
    for label, mask in [
        ("ELL", (dev["ell_clean"] == "Yes").to_numpy()),
        ("non_ELL", (dev["ell_clean"] == "No").to_numpy()),
    ]:
        d_rho, d_lo, d_hi = bootstrap_paired_delta_ci(
            dev_y[mask], pred0[mask], pred1[mask],
            metric_fn=lambda yt, yp: float(spearmanr(yt, yp).statistic), n_boot=N_BOOT, seed=SEED,
        )
        subgroup_rows.append({"subgroup": label, "n": int(mask.sum()), "delta_rho": d_rho,
                               "delta_rho_ci_low": d_lo, "delta_rho_ci_high": d_hi})
    for grade, group in dev.groupby("grade_level"):
        mask = (dev["grade_level"] == grade).to_numpy()
        if mask.sum() < 100:
            continue
        d_rho, d_lo, d_hi = bootstrap_paired_delta_ci(
            dev_y[mask], pred0[mask], pred1[mask],
            metric_fn=lambda yt, yp: float(spearmanr(yt, yp).statistic), n_boot=N_BOOT, seed=SEED,
        )
        subgroup_rows.append({"subgroup": f"grade_{int(grade)}", "n": int(mask.sum()), "delta_rho": d_rho,
                               "delta_rho_ci_low": d_lo, "delta_rho_ci_high": d_hi})
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(RESULTS_DIR / "experiment6_subgroup_deltas.csv", index=False)
    print("\nM1-M0 delta rho by subgroup:\n" + subgroup_df.to_string(index=False))

    # Final held-out confirmation: fit on ALL of dev, score once on held-out
    held_out_feats = compute_feature_matrix(held_out["full_text"])
    held_out_wc = held_out["word_count"].to_numpy(dtype=float).reshape(-1, 1)
    held_out_y = held_out["holistic_essay_score"].to_numpy(dtype=float)

    scaler0, scaler1 = StandardScaler().fit(X0), StandardScaler().fit(X1)
    model0 = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler0.transform(X0), dev_y)
    model1 = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler1.transform(X1), dev_y)
    held_out_pred0 = model0.predict(scaler0.transform(held_out_wc))
    held_out_pred1 = model1.predict(scaler1.transform(np.hstack([held_out_wc, held_out_feats.to_numpy()])))
    held_out_delta_rho, ho_lo, ho_hi = bootstrap_paired_delta_ci(
        held_out_y, held_out_pred0, held_out_pred1,
        metric_fn=lambda yt, yp: float(spearmanr(yt, yp).statistic), n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nHeld-out confirmation (never used for fitting/selection), n={len(held_out)}:")
    print(f"  M0 rho={r2_and_rho(held_out_y, held_out_pred0)['rho']:.4f}, "
          f"M1 rho={r2_and_rho(held_out_y, held_out_pred1)['rho']:.4f}, "
          f"delta={held_out_delta_rho:.4f} ({ho_lo:.4f}, {ho_hi:.4f})")

    # Feature direction table: refit ridge on ALL dev data (M1) for interpretable coefficients
    final_scaler = StandardScaler().fit(X1)
    final_model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(final_scaler.transform(X1), dev_y)
    coefs = final_model.coef_[1:]  # drop word_count's own coefficient (index 0)
    direction_rows = []
    for name, coef in zip(QUALITY_FEATURES, coefs):
        direction_rows.append({
            "feature": name, "standardized_coef": float(coef),
            "direction_vs_quality": "higher feature value -> HIGHER quality" if coef > 0 else "higher feature value -> LOWER quality",
            "feature_meaning": FEATURE_DIRECTION_HINTS[name],
        })
    direction_df = pd.DataFrame(direction_rows).sort_values("standardized_coef", key=abs, ascending=False)
    direction_df.to_csv(RESULTS_DIR / "experiment6_feature_directions.csv", index=False)
    print("\nFeature direction vs quality (ridge M1, standardized coefficients, controlling word count):\n" +
          direction_df.to_string(index=False))

    results_df.to_csv(RESULTS_DIR / "experiment6_nested.csv", index=False)
    plot_results(results_df, delta_rho, delta_rho_lo, delta_rho_hi)

    verdict = "YES" if delta_rho_lo > 0.2 else ("NO" if delta_rho_hi < 0.2 else "AMBIGUOUS")
    manifest_out = {
        **manifest,
        "seed": SEED, "n_bootstrap": N_BOOT, "n_folds": N_FOLDS, "held_out_frac": HELD_OUT_FRAC,
        "features_used": QUALITY_FEATURES,
        "dropped_raw_ttr_reason": "residual length confound per Experiment 2's partial-correlation gate",
        "pre_registered_expectation": PRE_REGISTERED_EXPECTATION,
        "dev_n": len(dev), "held_out_n": len(held_out),
        "delta_rho_dev_oof": {"point": delta_rho, "ci_low": delta_rho_lo, "ci_high": delta_rho_hi},
        "delta_r2_dev_oof": {"point": delta_r2, "ci_low": delta_r2_lo, "ci_high": delta_r2_hi},
        "delta_rho_held_out": {"point": held_out_delta_rho, "ci_low": ho_lo, "ci_high": ho_hi},
        "gbm_secondary_check": gbm_metrics,
        "verdict": verdict,
        "verdict_rule": "YES if delta_rho CI entirely above 0.2 (the document's pre-registered floor); NO if CI entirely below; else AMBIGUOUS",
        "no_percentile_shown_to_students": (
            "Per EXPERIMENT_6.md: PERSUADE is grades 6-12 argumentative essays averaging "
            "~418 words; production input is 300-650 word college personal statements. No "
            "percentile, band, or score derived from this corpus goes in front of a student. "
            "Any output here is for internal conditioning only."
        ),
    }
    with open(RESULTS_DIR / "experiment6_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"\nVerdict: {verdict} (delta_rho={delta_rho:.4f}, CI {delta_rho_lo:.4f}-{delta_rho_hi:.4f})")
    print(f"Manifest written to {RESULTS_DIR / 'experiment6_manifest.json'}")


def plot_results(results_df: pd.DataFrame, delta, delta_lo, delta_hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    y_pos = np.arange(len(results_df))
    errors = np.vstack([results_df["rho"] - results_df["rho_ci_low"], results_df["rho_ci_high"] - results_df["rho"]])
    ax1.barh(y_pos, results_df["rho"], xerr=errors, color="black", capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(results_df["model"])
    ax1.set_xlabel("Spearman rho vs holistic quality (dev, 5-fold OOF)")
    ax1.set_title("Nested models")

    ax2.errorbar([delta], [0], xerr=[[delta - delta_lo], [delta_hi - delta]], fmt="o", color="black", capsize=4, markersize=8)
    ax2.axvline(0.0, linestyle="--", color="gray", label="no incremental value")
    ax2.axvline(0.2, linestyle="--", color="blue", label="pre-registered floor (~0.2)")
    ax2.set_yticks([])
    ax2.set_xlabel("M1 - M0 delta rho")
    ax2.set_title("Incremental value of features over word count")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment6_deltas.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment6_deltas.png'}")


if __name__ == "__main__":
    main()
