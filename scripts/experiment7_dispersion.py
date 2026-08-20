"""Experiment 7: within-document polish (H1) vs across-document
homogenization (H2) as drivers of detectability.

EXPERIMENT_7.md. Uses only data already on disk -- no generation, no API
spend, no new corpora. Logged in AMENDMENTS.md as added post hoc, after
Experiment 3 produced an unexpected capability gradient (frontier models
most detectable), to test a mechanism hypothesis raised during analysis.
Exploratory: n=17 sources is small, and this cannot establish *why* models
behave this way (training recipes aren't published) -- only whether the
observed dispersion structure is more consistent with H1 or H2.

PRE-REGISTERED PREDICTIONS (recorded before computing anything):
  - Dispersion ordering, tightest to widest: frontier proprietary <
    older proprietary < open-weight < human.
  - Correlation between per-source dispersion and per-source TPR@1%FPR:
    negative, |rho| > 0.5.

DEVIATION FROM THE DESIGN DOC, logged here rather than silently applied:
"hold the prompt set fixed" as a single set common to all 18 groups (human
+ 17 sources) turns out to leave only 2 of PERSUADE's 15 prompts ("Car-free
cities", "Does the electoral college work?") -- several high-volume
sources have as few as 12-18 essays on just those two. Using that strict
cut for every source would make most per-source dispersion estimates
unusably noisy. The PRIMARY analysis instead uses each source's own full
available prompt coverage (matched against human on that same prompt set,
pairwise, per source) -- still controlling the core confound (comparing a
source's dispersion to human dispersion computed on the SAME topics it
wrote about), just not with one single set common to all 18 groups at
once. The strict 2-prompt cut is also computed and reported separately as
a small-n robustness check.

Also computes a direct, single-number follow-up to the source-level
direction check: cosine similarity between the frozen composite's own
P(AI) discriminant coefficient vector and the human-mean-to-top-quality-
quartile vector, both expressed in the composite's own StandardScaler
space (requires results/experiment3_frozen_composite.joblib to exist).

Usage: uv run python scripts/experiment7_dispersion.py
Output: results/experiment7_dispersion.csv, results/experiment7_per_feature_sd.csv,
        results/experiment7_manifest.json (includes the discriminant/quality
        cosine similarity), results/experiment7_dispersion_vs_tpr.png,
        results/experiment7_feature_sd_heatmap.png
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src import features as feat
from src.data import load_and_clean
from src.dispersion import (
    centroid,
    cosine_similarity,
    covariance_determinant,
    covariance_trace,
    mean_distance_to_centroid,
    project_onto_vector,
    standardize_by_reference,
)
from src.stats_utils import bootstrap_stat_ci

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
DAIGT_PATH = Path("data/train_v2_drcat_02.csv")
BEDROCK_PATH = Path("data/bedrock_claude_essays.csv")
OPENAI_PATH = Path("data/openai_gpt56terra_essays.csv")
TPR_PATH = Path("results/experiment3_tpr_by_model.csv")
FROZEN_COMPOSITE_PATH = Path("results/experiment3_frozen_composite.joblib")
RESULTS_DIR = Path("results")

SEED = 42
N_BOOT = 1000
MIN_AI_WORD_COUNT = 20
STRICT_COMMON_PROMPTS = ["Car-free cities", "Does the electoral college work?"]

# Proprietary/frontier vs older-proprietary vs open-weight classification,
# for the pre-registered ordering prediction. Judgment calls, stated plainly:
# "older proprietary" = pre-2024-frontier-era proprietary vendor models;
# "open-weight" = publicly released weights (Llama, Mistral, Falcon).
FRONTIER_PROPRIETARY = ["claude_sonnet_5_bedrock", "gpt_5.6_terra_openai"]
OLDER_PROPRIETARY = ["darragh_claude_v6", "darragh_claude_v7", "chat_gpt_moth", "radekgpt4",
                     "cohere-command", "palm-text-bison1", "kingki19_palm", "radek_500"]
OPEN_WEIGHT = ["llama2_chat", "llama_70b_v1", "NousResearch/Llama-2-7b-chat-hf",
               "mistral7binstruct_v1", "mistral7binstruct_v2", "mistralai/Mistral-7B-Instruct-v0.1",
               "falcon_180b_v1"]

PRE_REGISTERED_ORDERING = "frontier proprietary < older proprietary < open-weight < human (tightest to widest)"
PRE_REGISTERED_CORRELATION = "dispersion vs TPR: negative, |rho| > 0.5"


def load_ai_sources() -> pd.DataFrame:
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


def residualize_on_length(X: np.ndarray, log_wc: np.ndarray, reg_models: list) -> np.ndarray:
    """Subtract each feature's length-predicted component (regression fit
    on human data only, applied uniformly) -- removes residual length
    confound from the standardized features before computing dispersion."""
    residuals = np.zeros_like(X)
    for j, model in enumerate(reg_models):
        predicted = model.predict(log_wc.reshape(-1, 1))
        residuals[:, j] = X[:, j] - predicted
    return residuals


def dispersion_with_ci(X: np.ndarray, common_n: int, seed: int) -> dict:
    """Bootstrap dispersion statistics at a common subsample size (so
    sources with more documents don't appear more/less dispersed for
    purely statistical reasons). Point estimate is the mean of the
    bootstrap distribution, not a single arbitrary draw."""
    n = len(X)
    sample_n = min(common_n, n)
    rng = np.random.default_rng(seed)

    boot = {"dist": [], "trace": [], "det": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, sample_n)
        sample = X[idx]
        boot["dist"].append(mean_distance_to_centroid(sample))
        boot["trace"].append(covariance_trace(sample))
        boot["det"].append(covariance_determinant(sample))

    out = {"n_full": n, "n_subsampled": sample_n}
    for key, arr in boot.items():
        arr = np.asarray(arr)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[f"{key}_ci_low"], out[f"{key}_ci_high"] = float(lo), float(hi)
        out[f"{key}_mean"] = float(np.mean(arr))
    out["mean_dist_to_centroid"] = out["dist_mean"]
    out["cov_trace"] = out["trace_mean"]
    out["cov_det"] = out["det_mean"]
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("PRE-REGISTERED PREDICTIONS:")
    print(f"  Ordering: {PRE_REGISTERED_ORDERING}")
    print(f"  Correlation: {PRE_REGISTERED_CORRELATION}")

    human, _ = load_and_clean(PERSUADE_PATH)
    human = human.reset_index(drop=True)
    ai = load_ai_sources().reset_index(drop=True)
    tpr_df = pd.read_csv(TPR_PATH).set_index("source")["tpr"]
    print(f"\nHuman: {len(human)}. AI: {len(ai)} across {ai['source'].nunique()} sources.")

    print("Computing Tier 1 features...")
    human_feats_raw = compute_feature_matrix(human["full_text"]).to_numpy()
    ai_feats_raw = compute_feature_matrix(ai["full_text"]).to_numpy()

    # Standardize by HUMAN reference, then residualize on log word count
    # (regression fit on human data only, applied to all groups uniformly).
    human_z = standardize_by_reference(human_feats_raw, human_feats_raw)
    ai_z = standardize_by_reference(ai_feats_raw, human_feats_raw)

    human_log_wc = np.log1p(human["word_count"].to_numpy(dtype=float))
    ai_log_wc = np.log1p(ai["word_count"].to_numpy(dtype=float))
    reg_models = []
    for j in range(human_z.shape[1]):
        m = LinearRegression().fit(human_log_wc.reshape(-1, 1), human_z[:, j])
        reg_models.append(m)
    human_resid = residualize_on_length(human_z, human_log_wc, reg_models)
    ai_resid = residualize_on_length(ai_z, ai_log_wc, reg_models)

    feature_names = list(feat.TIER1_FEATURES.keys())
    common_n = int(min(ai.groupby("source").size().min(), 500))
    print(f"Common subsample n for dispersion bootstrap: {common_n}")

    # --- PRIMARY: each source's own full prompt coverage, matched pairwise vs human ---
    rows, sd_rows = [], []
    for source in tpr_df.index:
        mask = (ai["source"] == source).to_numpy()
        source_prompts = set(ai.loc[mask, "prompt_name"].unique())
        X_source = ai_resid[mask]
        human_mask_matched = human["prompt_name"].isin(source_prompts).to_numpy()

        disp = dispersion_with_ci(X_source, common_n, SEED)
        disp["source"] = source
        disp["tpr"] = float(tpr_df[source])
        disp["n_prompts_matched"] = len(source_prompts)
        disp["centroid_dist_from_human"] = float(
            np.linalg.norm(centroid(X_source) - centroid(human_resid[human_mask_matched]))
        )
        rows.append(disp)

        sd_row = {"source": source}
        for j, fname in enumerate(feature_names):
            sd_row[fname] = float(np.std(X_source[:, j], ddof=1))
        sd_rows.append(sd_row)

    human_disp = dispersion_with_ci(human_resid, common_n, SEED)
    human_disp["source"] = "human (full corpus)"
    human_disp["tpr"] = np.nan
    human_disp["n_prompts_matched"] = human["prompt_name"].nunique()
    human_disp["centroid_dist_from_human"] = 0.0
    rows.append(human_disp)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "experiment7_dispersion.csv", index=False)
    print("\nPer-source dispersion (primary cut, own prompt coverage):\n" +
          results_df[["source", "n_full", "n_subsampled", "mean_dist_to_centroid",
                       "cov_trace", "cov_det", "centroid_dist_from_human", "tpr"]].to_string(index=False))

    sd_df = pd.DataFrame(sd_rows)
    sd_df.to_csv(RESULTS_DIR / "experiment7_per_feature_sd.csv", index=False)

    # --- Correlation test: dispersion vs TPR (17 AI sources only) ---
    ai_rows = results_df[results_df["source"] != "human (full corpus)"]
    disp_vals = ai_rows["mean_dist_to_centroid"].to_numpy()
    tpr_vals = ai_rows["tpr"].to_numpy()
    rho_point, rho_lo, rho_hi = bootstrap_stat_ci(
        [disp_vals, tpr_vals], lambda a, b: float(spearmanr(a, b).statistic), n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nSpearman rho(dispersion, TPR) across {len(ai_rows)} sources: "
          f"{rho_point:.3f} ({rho_lo:.3f}, {rho_hi:.3f})")
    print(f"Pre-registered: {PRE_REGISTERED_CORRELATION}")
    correlation_held = rho_hi < -0.5 or (rho_point <= -0.5 and rho_lo < -0.5)

    # --- Centroid distance vs TPR, and joint regression ---
    cdist_vals = ai_rows["centroid_dist_from_human"].to_numpy()
    rho_cdist, cdist_lo, cdist_hi = bootstrap_stat_ci(
        [cdist_vals, tpr_vals], lambda a, b: float(spearmanr(a, b).statistic), n_boot=N_BOOT, seed=SEED,
    )
    print(f"Spearman rho(centroid_distance, TPR): {rho_cdist:.3f} ({cdist_lo:.3f}, {cdist_hi:.3f})")

    joint_X = np.column_stack([disp_vals, cdist_vals])
    joint_model = LinearRegression().fit(joint_X, tpr_vals)
    joint_r2 = joint_model.score(joint_X, tpr_vals)
    print(f"TPR ~ dispersion + centroid_distance: coefs={joint_model.coef_}, "
          f"intercept={joint_model.intercept_:.3f}, R2={joint_r2:.3f}")

    # --- Direction check: does frontier centroid project toward high-quality human region? ---
    quality = human["holistic_essay_score"].to_numpy(dtype=float)
    q75 = np.quantile(quality, 0.75)
    high_quality_mask = quality >= q75
    human_centroid_all = centroid(human_resid)
    human_centroid_highq = centroid(human_resid[high_quality_mask])
    direction = human_centroid_highq - human_centroid_all

    projections = {}
    for source in tpr_df.index:
        mask = (ai["source"] == source).to_numpy()
        source_centroid = centroid(ai_resid[mask])
        projections[source] = project_onto_vector(source_centroid, human_centroid_all, direction)
    proj_df = pd.DataFrame({"source": list(projections.keys()), "projection_toward_high_quality": list(projections.values())})
    proj_df = proj_df.merge(tpr_df.rename("tpr"), left_on="source", right_index=True)
    proj_df.to_csv(RESULTS_DIR / "experiment7_direction_check.csv", index=False)
    print("\nDirection check (projection onto human-mean -> high-quality-quartile vector):\n" +
          proj_df.sort_values("projection_toward_high_quality", ascending=False).to_string(index=False))
    rho_proj, proj_lo, proj_hi = bootstrap_stat_ci(
        [proj_df["projection_toward_high_quality"].to_numpy(), proj_df["tpr"].to_numpy()],
        lambda a, b: float(spearmanr(a, b).statistic), n_boot=N_BOOT, seed=SEED,
    )
    print(f"Spearman rho(projection_toward_high_quality, TPR): {rho_proj:.3f} ({proj_lo:.3f}, {proj_hi:.3f})")

    # --- Discriminant-direction vs quality-direction angle ---
    # A direct, single-number test of the same question the direction check
    # above asks indirectly: does the composite's own P(AI) discriminant
    # axis point toward higher human-rated quality? Both vectors must live
    # in the SAME standardized space to make cosine similarity meaningful --
    # here that's the frozen composite's own StandardScaler space (fit on
    # its pooled human+AI training data in experiment 3), NOT the
    # human-referenced/length-residualized space used elsewhere in this
    # script. The discriminant vector only has a well-defined direction in
    # the space it was fit in, so the quality vector is transformed into
    # that same space rather than the other way around.
    composite = joblib.load(FROZEN_COMPOSITE_PATH)  # our own artifact from experiment3_separation.py, not external/untrusted
    assert composite["feature_names"] == feature_names, (
        "Frozen composite's feature order doesn't match this script's -- "
        "cosine similarity would silently compare mismatched axes."
    )
    discriminant_vec = composite["model"].coef_[0]
    human_centroid_all_raw = human_feats_raw.mean(axis=0)
    human_centroid_highq_raw = human_feats_raw[high_quality_mask].mean(axis=0)
    quality_vec_composite_space = (human_centroid_highq_raw - human_centroid_all_raw) / composite["scaler"].scale_
    discriminant_quality_cosine = cosine_similarity(discriminant_vec, quality_vec_composite_space)
    discriminant_quality_angle_deg = float(np.degrees(np.arccos(np.clip(discriminant_quality_cosine, -1.0, 1.0))))
    print(f"\nCosine similarity, discriminant direction (P(AI) coefficients) vs "
          f"human-mean-to-top-quality-quartile vector (both in the frozen composite's "
          f"own standardized space): {discriminant_quality_cosine:.3f} "
          f"(angle = {discriminant_quality_angle_deg:.1f} degrees)")

    # --- Human subgroup dispersion (quality quartiles, ELL) ---
    q25 = np.quantile(quality, 0.25)
    subgroup_masks = {
        "human_full": np.ones(len(human), dtype=bool),
        "human_top_quality_quartile": quality >= q75,
        "human_bottom_quality_quartile": quality <= q25,
        "human_ELL": (human["ell_clean"] == "Yes").to_numpy(),
        "human_non_ELL": (human["ell_clean"] == "No").to_numpy(),
    }
    subgroup_rows = []
    for label, mask in subgroup_masks.items():
        d = dispersion_with_ci(human_resid[mask], common_n, SEED)
        d["subgroup"] = label
        subgroup_rows.append(d)
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(RESULTS_DIR / "experiment7_human_subgroup_dispersion.csv", index=False)
    print("\nHuman subgroup dispersion:\n" +
          subgroup_df[["subgroup", "n_full", "mean_dist_to_centroid", "cov_trace"]].to_string(index=False))
    top_q_disp = subgroup_df.loc[subgroup_df["subgroup"] == "human_top_quality_quartile", "mean_dist_to_centroid"].iloc[0]
    bottom_q_disp = subgroup_df.loc[subgroup_df["subgroup"] == "human_bottom_quality_quartile", "mean_dist_to_centroid"].iloc[0]
    quality_dispersion_prediction_held = bool(top_q_disp < bottom_q_disp)
    print(f"\nPrediction 'top quality quartile has LOWER dispersion than bottom': "
          f"{quality_dispersion_prediction_held} (top={top_q_disp:.3f}, bottom={bottom_q_disp:.3f})")

    # --- Ordering prediction check ---
    def group_mean_disp(sources):
        return results_df[results_df["source"].isin(sources)]["mean_dist_to_centroid"].mean()
    ordering_values = {
        "frontier_proprietary": float(group_mean_disp(FRONTIER_PROPRIETARY)),
        "older_proprietary": float(group_mean_disp(OLDER_PROPRIETARY)),
        "open_weight": float(group_mean_disp(OPEN_WEIGHT)),
        "human": float(human_disp["mean_dist_to_centroid"]),
    }
    print(f"\nGroup mean dispersion: {ordering_values}")
    ordering_held = bool(ordering_values["frontier_proprietary"] < ordering_values["older_proprietary"]
                          < ordering_values["open_weight"] < ordering_values["human"])
    print(f"Pre-registered ordering held exactly: {ordering_held}")

    # --- Strict common-prompt (2 prompts) robustness check ---
    print(f"\nStrict common-prompt robustness check ({STRICT_COMMON_PROMPTS}):")
    strict_rows = []
    human_strict_mask = human["prompt_name"].isin(STRICT_COMMON_PROMPTS).to_numpy()
    for source in tpr_df.index:
        mask = (ai["source"] == source).to_numpy() & ai["prompt_name"].isin(STRICT_COMMON_PROMPTS).to_numpy()
        n = int(mask.sum())
        if n < 5:
            strict_rows.append({"source": source, "n": n, "mean_dist_to_centroid": np.nan})
            continue
        strict_rows.append({"source": source, "n": n,
                             "mean_dist_to_centroid": mean_distance_to_centroid(ai_resid[mask])})
    strict_df = pd.DataFrame(strict_rows)
    strict_df.to_csv(RESULTS_DIR / "experiment7_strict_common_prompt_check.csv", index=False)
    print(strict_df.to_string(index=False))
    strict_merged = strict_df.merge(tpr_df.rename("tpr"), left_on="source", right_index=True).dropna()
    if len(strict_merged) >= 4:
        rho_strict, _, _ = bootstrap_stat_ci(
            [strict_merged["mean_dist_to_centroid"].to_numpy(), strict_merged["tpr"].to_numpy()],
            lambda a, b: float(spearmanr(a, b).statistic), n_boot=N_BOOT, seed=SEED,
        )
        print(f"Strict-cut rho(dispersion, TPR), n={len(strict_merged)} sources: {rho_strict:.3f} (small-n, exploratory)")

    plot_dispersion_vs_tpr(results_df)
    plot_feature_sd_heatmap(sd_df, tpr_df)

    manifest = {
        "seed": SEED, "n_bootstrap": N_BOOT, "common_subsample_n": common_n,
        "pre_registered_ordering": PRE_REGISTERED_ORDERING,
        "pre_registered_correlation": PRE_REGISTERED_CORRELATION,
        "ordering_group_means": ordering_values,
        "ordering_held_exactly": ordering_held,
        "correlation_dispersion_tpr": {"rho": rho_point, "ci_low": rho_lo, "ci_high": rho_hi},
        "correlation_prediction_held": bool(correlation_held),
        "correlation_centroid_distance_tpr": {"rho": rho_cdist, "ci_low": cdist_lo, "ci_high": cdist_hi},
        "correlation_projection_toward_quality_tpr": {"rho": rho_proj, "ci_low": proj_lo, "ci_high": proj_hi},
        "discriminant_quality_angle": {
            "cosine_similarity": discriminant_quality_cosine,
            "angle_degrees": discriminant_quality_angle_deg,
            "note": (
                "Cosine similarity between the frozen composite's P(AI) "
                "discriminant coefficient vector and the human-mean-to-"
                "top-quality-quartile vector, both expressed in the "
                "composite's own StandardScaler space (fit on its pooled "
                "human+AI training data in experiment 3) -- not the "
                "human-referenced/length-residualized space used for "
                "dispersion elsewhere in this script, since the "
                "discriminant vector is only meaningful in the space it "
                "was fit in. Added post hoc to quantify, rather than "
                "infer from a source-level correlation, whether the "
                "composite's own discriminant axis points toward higher "
                "human-rated quality -- this is what Experiment 2's "
                "positive partial correlation (+0.135) implies "
                "geometrically."
            ),
        },
        "joint_regression_tpr_on_dispersion_and_centroid_distance": {
            "coef_dispersion": float(joint_model.coef_[0]), "coef_centroid_distance": float(joint_model.coef_[1]),
            "intercept": float(joint_model.intercept_), "r2": float(joint_r2),
        },
        "quality_dispersion_prediction_held": quality_dispersion_prediction_held,
        "standardization": "z-scored by human corpus mean/SD, then residualized on log1p(word_count) via a regression fit on human data only and applied uniformly to all groups",
        "prompt_matching_deviation": (
            "Strict common-prompt-set-across-all-18-groups leaves only 2 of 15 "
            "PERSUADE prompts, with several sources down to n=12-18 there. "
            "PRIMARY analysis uses each source's own full prompt coverage, "
            "matched pairwise against human on that same prompt set. The "
            "strict 2-prompt cut is reported separately as a small-n check "
            "(results/experiment7_strict_common_prompt_check.csv)."
        ),
        "group_classification_caveat": (
            "FRONTIER_PROPRIETARY / OLDER_PROPRIETARY / OPEN_WEIGHT groupings "
            "are judgment calls made for the pre-registered ordering test, "
            "stated explicitly in this script rather than left implicit."
        ),
        "exploratory_caveat": (
            "n=17 sources; this is suggestive evidence about a mechanism "
            "hypothesis, not a decisive test, and cannot establish WHY models "
            "behave this way (training recipes are not published)."
        ),
    }
    with open(RESULTS_DIR / "experiment7_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment7_manifest.json'}")


def plot_dispersion_vs_tpr(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ai_rows = results_df[results_df["source"] != "human (full corpus)"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ai_rows["mean_dist_to_centroid"], ai_rows["tpr"], color="black")
    for _, row in ai_rows.iterrows():
        ax.annotate(row["source"], (row["mean_dist_to_centroid"], row["tpr"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    human_row = results_df[results_df["source"] == "human (full corpus)"].iloc[0]
    ax.axvline(human_row["mean_dist_to_centroid"], linestyle="--", color="blue", label="human dispersion (reference)")
    ax.set_xlabel("Mean distance to own centroid (standardized, length-residualized)")
    ax.set_ylabel("TPR @ 1% FPR (Experiment 3)")
    ax.set_title("Experiment 7: dispersion vs detectability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment7_dispersion_vs_tpr.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment7_dispersion_vs_tpr.png'}")


def plot_feature_sd_heatmap(sd_df: pd.DataFrame, tpr_df: pd.Series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sd_df = sd_df.set_index("source").loc[tpr_df.sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sd_df.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sd_df.columns)))
    ax.set_xticklabels(sd_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(sd_df.index)))
    ax.set_yticklabels(sd_df.index)
    ax.set_title("Per-feature SD by source (standardized, length-residualized), ordered by TPR (low->high)")
    fig.colorbar(im, label="SD (human-referenced z-score units)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment7_feature_sd_heatmap.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment7_feature_sd_heatmap.png'}")


if __name__ == "__main__":
    main()
