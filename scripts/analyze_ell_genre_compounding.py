"""Do ELL status and genre/topic shift compound, or are they independent?

Not in the original plan or either EXPERIMENT_*.md design doc -- see
AMENDMENTS.md, item 3, for when and why this was added mid-study.

User-directed follow-up. ELLIPSE's "different task" FPR (3.73%) mixes two
things: ELL population AND unseen topics, both relative to PERSUADE. To
separate them, this fits a SEPARATE diagnostic composite (not the main
frozen one used everywhere else) with 4 PERSUADE prompts held out
entirely (both human and AI training rows for those prompts excluded),
chosen for strong ELL representation so the held-out cells have usable N:
Distance learning, Exploring Venus, Facial action coding system,
Mandatory extracurricular activities.

This gives a clean within-PERSUADE 2x2 (ELL x genre-shift), holding corpus
identity constant — isolating the genre-shift effect from the "different
corpus entirely" effect that ELLIPSE's numbers necessarily include.
ELLIPSE's numbers are kept as an external corroborating reference, not the
primary compounding test.

Compounding test: convert each cell's FPR to log-odds. If effects are
independent (additive in log-odds / multiplicative in odds), the held-out
ELL cell's logit should equal baseline + ELL_effect + genre_effect. The gap
between observed and this additive prediction (with a bootstrap CI on that
gap) is the compounding/interaction signal.

Usage: uv run python scripts/analyze_ell_genre_compounding.py
Output: results/ell_genre_compounding.csv, results/ell_genre_compounding.json,
        results/ell_genre_compounding.png
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import features as feat
from src.data import load_and_clean
from src.stats_utils import bootstrap_stat_ci, rate_at_threshold

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
DAIGT_PATH = Path("data/train_v2_drcat_02.csv")
BEDROCK_PATH = Path("data/bedrock_claude_essays.csv")
OPENAI_PATH = Path("data/openai_gpt56terra_essays.csv")
RESULTS_DIR = Path("results")

HELD_OUT_PROMPTS = [
    "Distance learning", "Exploring Venus", "Facial action coding system",
    "Mandatory extracurricular activities",
]
SEED = 42
N_BOOT = 2000
TARGET_FPR = 0.01
MIN_AI_WORD_COUNT = 20

# External reference (different corpus entirely, kept separate from the primary test)
ELLIPSE_SAME_TASK_FPR = 0.024123
ELLIPSE_DIFFERENT_TASK_FPR = 0.037343
RAID_ABSTRACTS_FPR = (0.3692, 0.3286, 0.4118)  # point, lo, hi -- from prior run


def load_ai_with_prompt() -> pd.DataFrame:
    daigt = pd.read_csv(DAIGT_PATH)
    daigt = daigt[(daigt["label"] == 1) & (daigt["source"] != "train_essays")].copy()
    daigt["word_count"] = daigt["text"].str.split().str.len()
    daigt = daigt[daigt["word_count"] >= MIN_AI_WORD_COUNT]
    daigt = daigt.rename(columns={"text": "full_text"})[["full_text", "prompt_name"]]

    bedrock = pd.read_csv(BEDROCK_PATH)[["full_text", "prompt_name"]]
    openai_df = pd.read_csv(OPENAI_PATH)[["full_text", "prompt_name"]]
    return pd.concat([daigt, bedrock, openai_df], ignore_index=True)


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    out = {name: texts.apply(fn) for name, fn in feat.TIER1_FEATURES.items()}
    return pd.DataFrame(out, index=texts.index)


def fpr_ci(scores: np.ndarray, threshold: float) -> tuple:
    return bootstrap_stat_ci([scores], lambda s: rate_at_threshold(s, threshold), n_boot=N_BOOT, seed=SEED)


def logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def bootstrap_logit_gap(scores_a: np.ndarray, threshold_a: float,
                         scores_b: np.ndarray, threshold_b: float,
                         scores_c: np.ndarray, threshold_c: float,
                         scores_d: np.ndarray, threshold_d: float,
                         n_boot: int, seed: int) -> tuple:
    """Bootstrap CI for (observed logit(D) - predicted logit(D)) where the
    additive-in-log-odds prediction is logit(A) + [logit(B)-logit(A)] +
    [logit(C)-logit(A)]. A=baseline, B=ELL-only, C=genre-only, D=both."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        a = scores_a[rng.integers(0, len(scores_a), len(scores_a))]
        b = scores_b[rng.integers(0, len(scores_b), len(scores_b))]
        c = scores_c[rng.integers(0, len(scores_c), len(scores_c))]
        d = scores_d[rng.integers(0, len(scores_d), len(scores_d))]
        fa, fb, fc, fd = (rate_at_threshold(a, threshold_a), rate_at_threshold(b, threshold_b),
                          rate_at_threshold(c, threshold_c), rate_at_threshold(d, threshold_d))
        if 0 in (fa, fb, fc, fd) or 1 in (fa, fb, fc, fd):
            continue
        predicted = logit(fa) + (logit(fb) - logit(fa)) + (logit(fc) - logit(fa))
        gaps.append(logit(fd) - predicted)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    point_fa = rate_at_threshold(scores_a, threshold_a)
    point_fb = rate_at_threshold(scores_b, threshold_b)
    point_fc = rate_at_threshold(scores_c, threshold_c)
    point_fd = rate_at_threshold(scores_d, threshold_d)
    predicted_point = logit(point_fa) + (logit(point_fb) - logit(point_fa)) + (logit(point_fc) - logit(point_fa))
    return logit(point_fd) - predicted_point, float(lo), float(hi)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    human, _ = load_and_clean(PERSUADE_PATH)
    ai = load_ai_with_prompt()

    held_out_mask = human["prompt_name"].isin(HELD_OUT_PROMPTS)
    train_human = human[~held_out_mask]
    heldout_human = human[held_out_mask]
    train_ai = ai[~ai["prompt_name"].isin(HELD_OUT_PROMPTS)]

    print(f"Held-out prompts: {HELD_OUT_PROMPTS}")
    print(f"Train human: {len(train_human)}, held-out human: {len(heldout_human)}, train AI: {len(train_ai)}")
    print(f"Held-out human by ELL status:\n{heldout_human['ell_clean'].value_counts()}")

    train_human_feats = compute_feature_matrix(train_human["full_text"])
    train_ai_feats = compute_feature_matrix(train_ai["full_text"])
    X = pd.concat([train_human_feats, train_ai_feats], ignore_index=True).to_numpy()
    y = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_ai))])

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler.transform(X), y)
    train_human_scores = model.predict_proba(scaler.transform(train_human_feats.to_numpy()))[:, 1]
    threshold = float(np.quantile(train_human_scores, 1 - TARGET_FPR))
    print(f"\nDiagnostic model threshold (1% target FPR on train-prompt human essays): {threshold:.4f}")

    joblib.dump({
        "scaler": scaler, "model": model, "threshold": threshold,
        "held_out_prompts": HELD_OUT_PROMPTS,
    }, RESULTS_DIR / "ell_genre_diagnostic_model.joblib")

    heldout_feats = compute_feature_matrix(heldout_human["full_text"])
    heldout_scores = model.predict_proba(scaler.transform(heldout_feats.to_numpy()))[:, 1]
    heldout_human = heldout_human.assign(p_ai_score=heldout_scores)
    train_human = train_human.assign(p_ai_score=train_human_scores)

    cells = {
        "A_baseline_nonELL_trainprompts": train_human[train_human["ell_clean"] == "No"]["p_ai_score"].to_numpy(),
        "B_ELL_trainprompts": train_human[train_human["ell_clean"] == "Yes"]["p_ai_score"].to_numpy(),
        "C_nonELL_heldoutprompts": heldout_human[heldout_human["ell_clean"] == "No"]["p_ai_score"].to_numpy(),
        "D_ELL_heldoutprompts": heldout_human[heldout_human["ell_clean"] == "Yes"]["p_ai_score"].to_numpy(),
    }

    rows = []
    for name, scores in cells.items():
        fpr, lo, hi = fpr_ci(scores, threshold)
        rows.append({"cell": name, "n": len(scores), "fpr": fpr, "fpr_ci_low": lo, "fpr_ci_high": hi})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "ell_genre_compounding.csv", index=False)
    print("\n2x2 (within PERSUADE, diagnostic model, prompts held out entirely):\n" +
          results_df.to_string(index=False))

    gap, gap_lo, gap_hi = bootstrap_logit_gap(
        cells["A_baseline_nonELL_trainprompts"], threshold,
        cells["B_ELL_trainprompts"], threshold,
        cells["C_nonELL_heldoutprompts"], threshold,
        cells["D_ELL_heldoutprompts"], threshold,
        n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nCompounding test: observed logit(D) minus additive-prediction logit(D) = "
          f"{gap:.3f} ({gap_lo:.3f}, {gap_hi:.3f})")
    print("  gap > 0 and CI excludes 0  -> effects COMPOUND (super-additive, worse than either alone predicts)")
    print("  gap ~ 0                    -> effects are roughly INDEPENDENT/additive in log-odds")
    print("  gap < 0 and CI excludes 0  -> effects PARTIALLY CANCEL (sub-additive)")

    print(f"\nExternal reference (different corpus, ELLIPSE, NOT part of the primary test):")
    print(f"  ELLIPSE same-task (ELL, PERSUADE-identical prompts):    {ELLIPSE_SAME_TASK_FPR:.4f}")
    print(f"  ELLIPSE different-task (ELL, unseen prompts):           {ELLIPSE_DIFFERENT_TASK_FPR:.4f}")
    print(f"  RAID abstracts (far genre, non-ELL by construction):    {RAID_ABSTRACTS_FPR[0]:.4f} "
          f"({RAID_ABSTRACTS_FPR[1]:.4f}, {RAID_ABSTRACTS_FPR[2]:.4f})")

    plot_results(results_df, gap, gap_lo, gap_hi)

    manifest = {
        "held_out_prompts": HELD_OUT_PROMPTS,
        "n_train_human": len(train_human), "n_heldout_human": len(heldout_human), "n_train_ai": len(train_ai),
        "diagnostic_threshold": threshold, "target_fpr": TARGET_FPR, "seed": SEED, "n_bootstrap": N_BOOT,
        "cells": results_df.to_dict("records"),
        "compounding_gap_logodds": {"point": gap, "ci_low": gap_lo, "ci_high": gap_hi},
        "external_reference_ellipse_same_task_fpr": ELLIPSE_SAME_TASK_FPR,
        "external_reference_ellipse_different_task_fpr": ELLIPSE_DIFFERENT_TASK_FPR,
        "external_reference_raid_abstracts_fpr": {
            "point": RAID_ABSTRACTS_FPR[0], "ci_low": RAID_ABSTRACTS_FPR[1], "ci_high": RAID_ABSTRACTS_FPR[2],
        },
        "note": (
            "This diagnostic model is separate from the main Experiment 3 frozen composite "
            "(results/experiment3_frozen_composite.joblib) used everywhere else in this project "
            "-- it exists solely to test ELL x genre-shift compounding via a clean within-PERSUADE "
            "held-out-prompt design, and should not be reused for other reported figures."
        ),
    }
    with open(RESULTS_DIR / "ell_genre_compounding.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'ell_genre_compounding.json'}")


def plot_results(results_df: pd.DataFrame, gap: float, gap_lo: float, gap_hi: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    y_pos = np.arange(len(results_df))
    errors = np.vstack([results_df["fpr"] - results_df["fpr_ci_low"], results_df["fpr_ci_high"] - results_df["fpr"]])
    ax1.barh(y_pos, results_df["fpr"], xerr=errors, color="black", capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(results_df["cell"])
    ax1.set_xlabel("FPR")
    ax1.set_title("Within-PERSUADE 2x2: ELL x held-out prompts")

    ax2.errorbar([gap], [0], xerr=[[gap - gap_lo], [gap_hi - gap]], fmt="o", color="black", capsize=4, markersize=8)
    ax2.axvline(0.0, linestyle="--", color="gray", label="independent (additive in log-odds)")
    ax2.set_yticks([])
    ax2.set_xlabel("Observed - predicted logit(FPR), cell D")
    ax2.set_title("Compounding test")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ell_genre_compounding.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'ell_genre_compounding.png'}")


if __name__ == "__main__":
    main()
