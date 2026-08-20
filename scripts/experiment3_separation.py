"""Experiment 3: Does the composite separate human from AI at a usable FPR?

RESEARCH_PLAN.md, "3. Does it separate human from AI at a usable false-
positive rate?" — reordered per user direction (see conversation record):
gate 2 (quality anti-correlation) is reapplied HERE, to the fitted
composite's out-of-fold score, rather than to individual Tier 1 features.
Individual-feature gate 2 has two defects that make it the wrong test:

1. It tests the RAW sign of corr(feature, quality), not the sign relative
   to the feature's direction of "AI-likeness." transition_phrase_rate is
   a good example: raw correlation with quality is positive (+0.15,
   "passes"), but high transition-phrase rate is itself an AI-like signal
   (template connectors were a production indicator) — so on its actual
   AI-suspicion axis, better essays look MORE AI-like, which is the exact
   same failure mode sentence_length_std shows with the opposite raw sign.
2. The panel emits one composite conclusion, not nine independent ones.
   Features can lean in different directions and cancel in the composite,
   or the composite can fail even when every individual feature passes.

Fitting the composite (a classifier separating human PERSUADE essays from
AI-generated text, using ALL 9 Tier 1 features — no feature is pre-dropped
here) sidesteps problem 1 entirely: logistic regression learns each
feature's sign empirically from the human/AI labels, rather than us
guessing "which direction is AI-like" per feature by folk theory and
risking exactly the kind of sign error found in the first pass.

Composite scores are computed strictly out-of-fold (5-fold stratified CV)
for every essay, so no gate-2 or threshold decision below is made using a
prediction the model was trained on.

AI corpus: DAIGT-v2 (`thedrcat/daigt-v2-train-dataset`), label==1 rows,
excluding the small `train_essays` source (a handful of mislabeled rows).
This is 15 generating models (source labels), including ~2,000 already-existing Claude
generations (darragh_claude_v6/v7) — but NOT yet the plan's separately-
called-for fresh Bedrock-generated set matched to production's current
Claude model and prompts ("no other corpus covers our own generator").
That requires AWS/Bedrock access and real spend and is deliberately left
as a follow-up decision rather than assumed here.

Usage: uv run python scripts/experiment3_separation.py
Output: results/experiment3_separation.csv, results/experiment3_tpr_by_model.png,
        results/experiment3_gate2_composite.png, results/experiment3_manifest.json,
        results/experiment3_ai_scores.csv, results/experiment3_human_scores.csv
        (the last two persist per-essay OOF P(AI) scores for reuse by later
        analyses, e.g. the Experiment 7 conditioning checks)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat
from src.data import load_and_clean, sha256_of
from src.stats_utils import (
    bootstrap_auc_ci,
    bootstrap_stat_ci,
    partial_spearman,
    rate_at_threshold,
    threshold_at_fpr,
)

SEED = 42
N_BOOT = 1000
TARGET_FPR = 0.01
PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
DAIGT_PATH = Path("data/train_v2_drcat_02.csv")
BEDROCK_CLAUDE_PATH = Path("data/bedrock_claude_essays.csv")
OPENAI_PATH = Path("data/openai_gpt56terra_essays.csv")
RESULTS_DIR = Path("results")
MIN_AI_WORD_COUNT = 20  # drop degenerate near-empty generations

FEATURE_FUNCS = feat.TIER1_FEATURES  # full 9-feature set, nothing pre-dropped


def load_daigt_ai_essays(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ai = df[(df["label"] == 1) & (df["source"] != "train_essays")].copy()
    ai["word_count"] = ai["text"].str.split().str.len()
    ai = ai[ai["word_count"] >= MIN_AI_WORD_COUNT].copy()
    ai = ai.rename(columns={"text": "full_text"})
    return ai[["full_text", "word_count", "source", "prompt_name"]]


def load_generated_essays(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = source_label
    return df[["full_text", "word_count", "source", "prompt_name"]]


def load_ai_essays() -> pd.DataFrame:
    daigt = load_daigt_ai_essays(DAIGT_PATH)
    bedrock = load_generated_essays(BEDROCK_CLAUDE_PATH, "claude_sonnet_5_bedrock")
    openai_essays = load_generated_essays(OPENAI_PATH, "gpt_5.6_terra_openai")
    return pd.concat([daigt, bedrock, openai_essays], ignore_index=True)


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    out = {name: texts.apply(fn) for name, fn in FEATURE_FUNCS.items()}
    return pd.DataFrame(out, index=texts.index)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    persuade_hash = sha256_of(PERSUADE_PATH)
    daigt_hash = sha256_of(DAIGT_PATH)
    bedrock_hash = sha256_of(BEDROCK_CLAUDE_PATH)
    openai_hash = sha256_of(OPENAI_PATH)

    human, human_manifest = load_and_clean(PERSUADE_PATH)
    human = human.reset_index(drop=True)
    ai = load_ai_essays().reset_index(drop=True)
    prompt_task_map = human[["prompt_name", "task"]].drop_duplicates().set_index("prompt_name")["task"]
    ai["task"] = ai["prompt_name"].map(prompt_task_map)
    print(f"Human (PERSUADE): {len(human)} essays. AI: {len(ai)} essays across "
          f"{ai['source'].nunique()} sources (DAIGT-v2 + "
          f"{(ai['source'] == 'claude_sonnet_5_bedrock').sum()} Claude Sonnet 5/Bedrock + "
          f"{(ai['source'] == 'gpt_5.6_terra_openai').sum()} GPT-5.6 Terra/OpenAI generations).")

    human_feats = compute_feature_matrix(human["full_text"])
    ai_feats = compute_feature_matrix(ai["full_text"])
    X = pd.concat([human_feats, ai_feats], ignore_index=True).to_numpy()
    y = np.concatenate([np.zeros(len(human)), np.ones(len(ai))])

    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_p_ai = cross_val_predict(
        LogisticRegression(max_iter=1000, random_state=SEED), Xs, y, cv=cv, method="predict_proba",
    )[:, 1]

    n_human = len(human)
    human_scores = oof_p_ai[:n_human]
    ai_scores = oof_p_ai[n_human:]

    auc_point, auc_lo, auc_hi = bootstrap_auc_ci(y, oof_p_ai, n_boot=N_BOOT, seed=SEED)
    print(f"\nOverall composite AUC (human vs AI): {auc_point:.3f} ({auc_lo:.3f}, {auc_hi:.3f})")

    threshold = threshold_at_fpr(human_scores, TARGET_FPR)
    overall_fpr = rate_at_threshold(human_scores, threshold)
    overall_tpr = rate_at_threshold(ai_scores, threshold)
    print(f"Threshold for {TARGET_FPR:.0%} target FPR (on human OOF scores): {threshold:.4f}")
    print(f"Realized overall FPR: {overall_fpr:.4f}, overall TPR: {overall_tpr:.4f}")

    ell_mask = (human["ell_clean"] == "Yes").to_numpy()
    fpr_ell = rate_at_threshold(human_scores[ell_mask], threshold)
    fpr_non_ell = rate_at_threshold(human_scores[~ell_mask], threshold)
    print(f"FPR by subgroup: ELL={fpr_ell:.4f} (n={ell_mask.sum()}), "
          f"non-ELL={fpr_non_ell:.4f} (n={(~ell_mask).sum()})")

    ai_scores_df = pd.DataFrame({
        "source": ai["source"], "prompt_name": ai["prompt_name"], "task": ai["task"],
        "word_count": ai["word_count"], "p_ai_oof_score": ai_scores,
    })
    ai_scores_df.to_csv(RESULTS_DIR / "experiment3_ai_scores.csv", index=False)

    # Persisted so downstream analyses can reuse the EXACT same out-of-fold
    # P(AI) scores Gate 2 above is computed from, instead of recomputing a
    # fresh CV fit and risking a subtle mismatch (fold assignment, row order).
    human_scores_df = pd.DataFrame({
        "prompt_name": human["prompt_name"], "task": human["task"],
        "word_count": human["word_count"], "ell_clean": human["ell_clean"],
        "holistic_essay_score": human["holistic_essay_score"],
        "p_ai_oof_score": human_scores,
    })
    human_scores_df.to_csv(RESULTS_DIR / "experiment3_human_scores.csv", index=False)

    # Frozen composite for reuse by later experiments: fit ONCE on all of
    # experiment 3's human+AI data (not cross-validated), so a downstream
    # experiment can score genuinely new essays without ever refitting.
    # New essays are held out from this fit, so this isn't leakage for them
    # — it would only be leakage to reuse it on essays already in X/y here.
    final_scaler = StandardScaler().fit(X)
    final_model = LogisticRegression(max_iter=1000, random_state=SEED).fit(final_scaler.transform(X), y)
    final_human_scores = final_model.predict_proba(final_scaler.transform(human_feats.to_numpy()))[:, 1]
    final_threshold = threshold_at_fpr(final_human_scores, TARGET_FPR)
    joblib.dump({
        "scaler": final_scaler, "model": final_model,
        "feature_names": list(FEATURE_FUNCS.keys()), "threshold": final_threshold,
        "target_fpr": TARGET_FPR, "seed": SEED,
        "trained_on_n_human": len(human), "trained_on_n_ai": len(ai),
    }, RESULTS_DIR / "experiment3_frozen_composite.joblib")
    print(f"\nFrozen composite saved: results/experiment3_frozen_composite.joblib "
          f"(threshold={final_threshold:.4f}, in-sample FPR check={rate_at_threshold(final_human_scores, final_threshold):.4f})")

    tpr_by_model = []
    for source, group in pd.DataFrame({"source": ai["source"], "score": ai_scores}).groupby("source"):
        tpr_by_model.append({
            "source": source, "n": len(group),
            "tpr": rate_at_threshold(group["score"], threshold),
        })
    tpr_by_model_df = pd.DataFrame(tpr_by_model).sort_values("tpr")
    print("\nTPR by generating model:\n" + tpr_by_model_df.to_string(index=False))

    # --- Gate 2, reapplied to the fitted composite (not individual features) ---
    quality = human["holistic_essay_score"].to_numpy(dtype=float)
    word_count = human["word_count"].to_numpy(dtype=float)

    raw_point, raw_lo, raw_hi = bootstrap_stat_ci(
        [human_scores, quality], lambda a, b: float(spearmanr(a, b).statistic),
        n_boot=N_BOOT, seed=SEED,
    )
    partial_point, partial_lo, partial_hi = bootstrap_stat_ci(
        [human_scores, quality, word_count], partial_spearman, n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nGate 2 on composite P(AI) score vs holistic quality (human essays only):")
    print(f"  raw rho     = {raw_point:.3f} ({raw_lo:.3f}, {raw_hi:.3f})")
    print(f"  partial rho = {partial_point:.3f} ({partial_lo:.3f}, {partial_hi:.3f})  [controlling word count]")
    print("  Sign convention: POSITIVE correlation here is the failure mode —")
    print("  it means better essays are scored as MORE AI-like.")

    separation_effect = 2 * auc_point - 1  # AUC as a rank-correlation-equivalent effect size
    quality_effect = abs(partial_point)
    ratio = quality_effect / separation_effect if separation_effect > 0 else float("inf")
    print(f"\nEffect-size comparison: |quality partial rho| = {quality_effect:.3f}, "
          f"separation effect (2*AUC-1) = {separation_effect:.3f}, ratio = {ratio:.3f}")
    print("  No threshold specified for this ratio — reporting magnitude only, per user direction.")

    plot_tpr_by_model(tpr_by_model_df, overall_tpr, fpr_ell, fpr_non_ell, overall_fpr)
    plot_gate2_composite(partial_point, partial_lo, partial_hi)

    summary_rows = [
        {"metric": "overall_auc", "value": auc_point, "ci_low": auc_lo, "ci_high": auc_hi},
        {"metric": "threshold_at_target_fpr", "value": threshold, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "overall_fpr", "value": overall_fpr, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "overall_tpr", "value": overall_tpr, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "fpr_ell", "value": fpr_ell, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "fpr_non_ell", "value": fpr_non_ell, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "gate2_composite_raw_rho", "value": raw_point, "ci_low": raw_lo, "ci_high": raw_hi},
        {"metric": "gate2_composite_partial_rho", "value": partial_point, "ci_low": partial_lo, "ci_high": partial_hi},
        {"metric": "separation_effect_2auc_minus_1", "value": separation_effect, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "quality_to_separation_ratio", "value": ratio, "ci_low": np.nan, "ci_high": np.nan},
    ]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "experiment3_separation.csv", index=False)
    tpr_by_model_df.to_csv(RESULTS_DIR / "experiment3_tpr_by_model.csv", index=False)

    manifest = {
        **human_manifest,
        "persuade_path": str(PERSUADE_PATH), "persuade_sha256": persuade_hash,
        "daigt_path": str(DAIGT_PATH), "daigt_sha256": daigt_hash,
        "bedrock_claude_path": str(BEDROCK_CLAUDE_PATH), "bedrock_claude_sha256": bedrock_hash,
        "openai_path": str(OPENAI_PATH), "openai_sha256": openai_hash,
        "n_human": n_human, "n_ai": len(ai), "ai_sources": sorted(ai["source"].unique().tolist()),
        "min_ai_word_count_filter": MIN_AI_WORD_COUNT,
        "seed": SEED, "n_bootstrap": N_BOOT, "target_fpr": TARGET_FPR,
        "features_used": list(FEATURE_FUNCS.keys()),
        "note_ai_corpus_scope": (
            "DAIGT-v2 (15 models) plus two fresh 1000-essay matched sets generated "
            "on the identical PERSUADE-prompt instructions (src/essay_prompts.py): "
            "us.anthropic.claude-sonnet-5 via Bedrock, and gpt-5.6-terra via the "
            "OpenAI API (user-directed addition, chosen as a proxy for free-tier "
            "ChatGPT output — not confirmed against OpenAI docs that Terra is what "
            "the free ChatGPT web app actually defaults to). Both use a generic "
            "instruction, not production's actual system prompt (unavailable to "
            "this project), and for the 7 text-dependent PERSUADE prompts both "
            "wrote from general knowledge rather than citing a source article, "
            "since the source article text itself isn't in the PERSUADE corpus "
            "(only a citation/title is)."
        ),
        "note_gate2_sign_convention": (
            "composite score = P(AI) from the fitted classifier. A POSITIVE "
            "correlation with holistic_essay_score is the failure mode (better "
            "essays score as more AI-like) — equivalent to the plan's framing "
            "of a human-likeness score anti-correlating with quality, sign-flipped."
        ),
        "note_ratio_threshold": (
            "No pass/fail threshold set for quality_to_separation_ratio; "
            "reported as a magnitude for judgment, not auto-decided."
        ),
        "note_held_out": (
            "All composite scores (human and AI) are out-of-fold predictions "
            "from 5-fold stratified CV; no threshold, subgroup rate, or "
            "correlation above uses a prediction the model was trained on."
        ),
        "frozen_composite_path": "results/experiment3_frozen_composite.joblib",
        "note_frozen_composite": (
            "Separate from the OOF-CV numbers above: a StandardScaler + "
            "LogisticRegression fit ONCE on all of this experiment's human+AI "
            "data (not cross-validated), for later experiments to score brand-new "
            "essays without refitting. Reusing it on essays already in this run's "
            "training data would leak; scoring genuinely new essays with it does not."
        ),
        "ai_scores_path": "results/experiment3_ai_scores.csv",
        "human_scores_path": "results/experiment3_human_scores.csv",
    }
    with open(RESULTS_DIR / "experiment3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment3_manifest.json'}")


def plot_tpr_by_model(tpr_by_model_df, overall_tpr, fpr_ell, fpr_non_ell, overall_fpr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(tpr_by_model_df) + 2.5))
    y_pos = np.arange(len(tpr_by_model_df))
    ax.barh(y_pos, tpr_by_model_df["tpr"], color="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tpr_by_model_df["source"])
    ax.axvline(overall_tpr, linestyle="--", color="blue", label=f"overall TPR ({overall_tpr:.2f})")
    ax.set_xlabel(f"TPR at threshold fit for {TARGET_FPR:.0%} target FPR")
    ax.set_title("Experiment 3: TPR by generating model", pad=12)
    fig.text(0.5, 0.955, f"FPR: overall={overall_fpr:.3f}, ELL={fpr_ell:.3f}, non-ELL={fpr_non_ell:.3f}",
              ha="center", fontsize=9, color="dimgray")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(RESULTS_DIR / "experiment3_tpr_by_model.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment3_tpr_by_model.png'}")


def plot_gate2_composite(partial_point, partial_lo, partial_hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.errorbar([partial_point], [0], xerr=[[partial_point - partial_lo], [partial_hi - partial_point]],
                fmt="o", color="black", capsize=4, markersize=8)
    ax.axvline(0.0, linestyle="--", color="gray", label="no correlation (0.0)")
    ax.set_yticks([])
    ax.set_xlabel("Partial Spearman rho: composite P(AI) vs holistic quality (controlling word count)")
    ax.set_title("Experiment 3 (composite gate 2): positive = better essays score more AI-like")
    ax.set_xlim(-0.3, 0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment3_gate2_composite.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment3_gate2_composite.png'}")


if __name__ == "__main__":
    main()
