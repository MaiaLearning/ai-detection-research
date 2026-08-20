"""Two conditioning checks on Experiment 7's discriminant-vs-quality cosine
similarity (cosine=0.067, angle=86.2 degrees, logged in
results/experiment7_manifest.json and AMENDMENTS.md item 6).

Not in the original plan, EXPERIMENT_7.md, or the initial post-hoc angle
measurement -- added after that near-orthogonal result raised a conditioning
question: the two vectors compared were NOT conditioned on word count the
same way. Uses only data already on disk; no generation, no API spend.

WHY THIS EXISTS

The composite's discriminant vector was fit on raw standardized features
(word-count information included). The human-mean-to-top-quality-quartile
vector was computed the same way -- also including word-count information.
But Experiment 2's headline number (partial rho = +0.135) explicitly
removes word count before correlating P(AI) score with quality. Word count
itself carries rho=0.76 with quality in this corpus, so the original cosine
may have been measuring the angle between a length-inclusive direction and
a length-inclusive direction that happens to look different from the
length-REMOVED relationship Gate 2 actually reports.

Separately: +0.135 is a Spearman (rank) statistic; cosine similarity is a
linear-geometry statistic. A gap between the two could reflect a genuine
monotone-but-nonlinear relationship that cosine similarity, which only sees
linear structure, cannot detect.

CHECK 1 -- same conditioning on both vectors.
Residualize all nine features on log word count within the human corpus
(regression fit on human only, applied uniformly to AI), in the SAME
standardized space used for the original cosine test (the frozen
composite's own StandardScaler). Refit a fresh discriminant (a new
LogisticRegression on the residualized human+AI matrix -- the original
composite's coefficients are not meaningful once the feature space itself
has changed) and recompute the quality vector in that same residualized
space. Report cosine and angle, against the same ~1/sqrt(9)=0.333 noise
band used for the original result (expected cosine magnitude between two
random 9-dimensional vectors).

CHECK 2 -- rank vs linear.
Pearson partial correlation between the composite's out-of-fold P(AI) score
(results/experiment3_human_scores.csv, the exact scores Gate 2 was computed
from -- not recomputed here, to avoid a fold-assignment or row-order
mismatch) and holistic quality, controlling word count, on the same human
sample Gate 2 used. Compared directly against Gate 2's Spearman partial
(+0.135).

HOW TO READ THE FOUR OUTCOMES (state which one obtained BEFORE
interpreting -- see the printed summary at the end of this script):

  angle after residualizing ~ still ~86 deg, Pearson partial ~ 0.135:
    genuine puzzle -- orthogonal directions, real scalar tilt.
  angle drops well below 86 deg, Pearson partial ~ 0.135:
    original cosine was a conditioning artifact; retract orthogonality claim.
  angle ~ still ~86 deg, Pearson partial much below 0.135:
    nonlinearity explains the original tension.
  angle drops, Pearson partial much below 0.135:
    both artifacts.

Logged alongside (not replacing) the original cosine in AMENDMENTS.md item
6 -- the discrepancy between the two conditionings is itself part of the
finding, whichever way it comes out.

Usage: uv run python scripts/analyze_discriminant_conditioning.py
Requires: results/experiment3_frozen_composite.joblib (from
    experiment3_separation.py) and results/experiment3_human_scores.csv
    (persisted by the same script -- rerun it first if missing).
Output: results/experiment7_conditioning_checks.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src import features as feat
from src.data import load_and_clean
from src.dispersion import centroid, cosine_similarity, residualize_on_length
from src.stats_utils import bootstrap_stat_ci, partial_pearson

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
DAIGT_PATH = Path("data/train_v2_drcat_02.csv")
BEDROCK_PATH = Path("data/bedrock_claude_essays.csv")
OPENAI_PATH = Path("data/openai_gpt56terra_essays.csv")
FROZEN_COMPOSITE_PATH = Path("results/experiment3_frozen_composite.joblib")
HUMAN_SCORES_PATH = Path("results/experiment3_human_scores.csv")
EXPERIMENT7_MANIFEST_PATH = Path("results/experiment7_manifest.json")
RESULTS_DIR = Path("results")

SEED = 42
N_BOOT = 1000
MIN_AI_WORD_COUNT = 20
NOISE_BAND_1SD = 1 / np.sqrt(9)  # expected |cosine| between two random 9-D vectors


def load_ai_sources() -> pd.DataFrame:
    """Same loader as scripts/experiment7_dispersion.py -- duplicated per
    this repo's convention of keeping each script self-contained."""
    daigt = pd.read_csv(DAIGT_PATH)
    daigt = daigt[(daigt["label"] == 1) & (daigt["source"] != "train_essays")].copy()
    daigt["word_count"] = daigt["text"].str.split().str.len()
    daigt = daigt[daigt["word_count"] >= MIN_AI_WORD_COUNT]
    daigt = daigt.rename(columns={"text": "full_text"})[["full_text", "word_count", "source", "prompt_name"]]

    bedrock = pd.read_csv(BEDROCK_PATH)[["full_text", "word_count", "prompt_name"]]
    bedrock["source"] = "claude_sonnet_5_bedrock"
    openai_df = pd.read_csv(OPENAI_PATH)[["full_text", "word_count", "prompt_name"]]
    openai_df["source"] = "gpt_5.6_terra_openai"
    return pd.concat([daigt, bedrock, openai_df], ignore_index=True)


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({name: texts.apply(fn) for name, fn in feat.TIER1_FEATURES.items()}, index=texts.index)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    composite = joblib.load(FROZEN_COMPOSITE_PATH)  # our own artifact from experiment3_separation.py, not external/untrusted
    feature_names = composite["feature_names"]

    human, _ = load_and_clean(PERSUADE_PATH)
    human = human.reset_index(drop=True)
    ai = load_ai_sources().reset_index(drop=True)

    print("=== Check 1: same conditioning on both vectors (residualize on word count) ===")
    human_feats_raw = compute_feature_matrix(human["full_text"]).to_numpy()
    ai_feats_raw = compute_feature_matrix(ai["full_text"]).to_numpy()
    assert list(compute_feature_matrix(human["full_text"]).columns) == feature_names

    # One standardized space -- the composite's own StandardScaler, the same
    # space the original (non-residualized) cosine test used.
    human_std = composite["scaler"].transform(human_feats_raw)
    ai_std = composite["scaler"].transform(ai_feats_raw)

    human_log_wc = np.log1p(human["word_count"].to_numpy(dtype=float))
    ai_log_wc = np.log1p(ai["word_count"].to_numpy(dtype=float))
    reg_models = [
        LinearRegression().fit(human_log_wc.reshape(-1, 1), human_std[:, j])
        for j in range(human_std.shape[1])
    ]
    human_resid = residualize_on_length(human_std, human_log_wc, reg_models)
    ai_resid = residualize_on_length(ai_std, ai_log_wc, reg_models)

    # Refit the discriminant IN the residualized space -- the original
    # composite's coefficients are not meaningful here, since the feature
    # space itself has changed.
    X_resid = np.vstack([human_resid, ai_resid])
    y = np.concatenate([np.zeros(len(human)), np.ones(len(ai))])
    resid_model = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_resid, y)
    discriminant_vec_resid = resid_model.coef_[0]

    quality = human["holistic_essay_score"].to_numpy(dtype=float)
    q75 = np.quantile(quality, 0.75)
    high_quality_mask = quality >= q75
    quality_vec_resid = centroid(human_resid[high_quality_mask]) - centroid(human_resid)

    cosine_resid = cosine_similarity(discriminant_vec_resid, quality_vec_resid)
    angle_resid_deg = float(np.degrees(np.arccos(np.clip(cosine_resid, -1.0, 1.0))))

    with open(EXPERIMENT7_MANIFEST_PATH) as f:
        exp7_manifest = json.load(f)
    original = exp7_manifest["discriminant_quality_angle"]
    print(f"Original (not conditioned on length) : cosine={original['cosine_similarity']:.3f}, "
          f"angle={original['angle_degrees']:.1f} deg")
    print(f"Residualized (both vectors, on word count): cosine={cosine_resid:.3f}, "
          f"angle={angle_resid_deg:.1f} deg")
    print(f"Noise band (|cosine| expected from 2 random 9-D vectors, 1 SD): {NOISE_BAND_1SD:.3f}")

    print("\n=== Check 2: rank (Spearman) vs linear (Pearson) ===")
    human_scores_df = pd.read_csv(HUMAN_SCORES_PATH)
    p_ai = human_scores_df["p_ai_oof_score"].to_numpy(dtype=float)
    hq = human_scores_df["holistic_essay_score"].to_numpy(dtype=float)
    wc = human_scores_df["word_count"].to_numpy(dtype=float)

    raw_pearson = float(np.corrcoef(p_ai, hq)[0, 1])
    partial_pearson_point, partial_pearson_lo, partial_pearson_hi = bootstrap_stat_ci(
        [p_ai, hq, wc], partial_pearson, n_boot=N_BOOT, seed=SEED,
    )
    # gate2_composite_partial_rho lives in experiment3_separation.csv, not
    # experiment3_manifest.json.
    sep_csv = pd.read_csv(RESULTS_DIR / "experiment3_separation.csv").set_index("metric")
    gate2_point = float(sep_csv.loc["gate2_composite_partial_rho", "value"])
    gate2_lo = float(sep_csv.loc["gate2_composite_partial_rho", "ci_low"])
    gate2_hi = float(sep_csv.loc["gate2_composite_partial_rho", "ci_high"])

    print(f"Raw Pearson (P(AI), quality)        : {raw_pearson:.3f}")
    print(f"Partial Pearson (controlling word count): {partial_pearson_point:.3f} "
          f"({partial_pearson_lo:.3f}, {partial_pearson_hi:.3f})")
    print(f"Gate 2's partial Spearman (for comparison): {gate2_point:.3f} ({gate2_lo:.3f}, {gate2_hi:.3f})")

    print("\n=== State the outcome before interpreting ===")
    print(f"Angle after residualizing: {angle_resid_deg:.1f} deg "
          f"(original was {original['angle_degrees']:.1f} deg; noise band +/-{NOISE_BAND_1SD:.3f} on cosine)")
    print(f"Partial Pearson: {partial_pearson_point:.3f} vs partial Spearman {gate2_point:.3f}")

    manifest = {
        "seed": SEED, "n_bootstrap": N_BOOT,
        "noise_band_1sd_cosine": float(NOISE_BAND_1SD),
        "check1_residualized_direction_angle": {
            "original_cosine": original["cosine_similarity"],
            "original_angle_degrees": original["angle_degrees"],
            "residualized_cosine": cosine_resid,
            "residualized_angle_degrees": angle_resid_deg,
            "standardization": (
                "Both vectors expressed in the frozen composite's own "
                "StandardScaler space (same space the original cosine used), "
                "then residualized on log1p(word_count) via regressions fit "
                "on human data only and applied uniformly to AI. The "
                "discriminant vector is a FRESH LogisticRegression fit on "
                "the residualized human+AI matrix -- the original "
                "composite's coefficients are not meaningful in this "
                "changed feature space."
            ),
        },
        "check2_pearson_vs_spearman": {
            "raw_pearson": raw_pearson,
            "partial_pearson": {"point": partial_pearson_point, "ci_low": partial_pearson_lo, "ci_high": partial_pearson_hi},
            "gate2_partial_spearman_for_comparison": {"point": gate2_point, "ci_low": gate2_lo, "ci_high": gate2_hi},
            "note": "Both computed on results/experiment3_human_scores.csv -- the exact OOF P(AI) scores Gate 2 used, not recomputed.",
        },
    }
    with open(RESULTS_DIR / "experiment7_conditioning_checks.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment7_conditioning_checks.json'}")


if __name__ == "__main__":
    main()
