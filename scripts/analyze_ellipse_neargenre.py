"""Near-genre FPR check on ELLIPSE, using Experiment 3's frozen composite.

User-directed follow-up before the writeup: RAID/abstracts showed severe
domain-shift degradation (1% -> 36.9% FPR). ELLIPSE is a far smaller shift
(still student essays responding to short-answer prompts) and does double
duty: it's entirely ELL writers, so its FPR is also a second, independent
read on the ELL fairness question from Experiment 1/3.

ELLIPSE's 44 prompts include 7 that are IDENTICAL to PERSUADE's own prompt
names (all 7 of the independent-task prompts used elsewhere in this
project) plus 37 prompts never seen in PERSUADE. This lets the shift be
decomposed into two cuts:
  - "same task" subset (912 essays): different population, same prompts
    as training -- isolates population/writer shift.
  - "different task" subset (5,570 essays): different population AND
    unseen prompts -- population shift + topic shift together.
No refitting: uses the frozen scaler+model+threshold from Experiment 3.

Usage: uv run python scripts/analyze_ellipse_neargenre.py
Output: results/ellipse_neargenre_fpr.csv, results/ellipse_manifest.json,
        results/ellipse_neargenre_fpr.png
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as feat
from src.data import sha256_of
from src.stats_utils import bootstrap_stat_ci, rate_at_threshold

ELLIPSE_PATH = Path("data/ellipse_corpus.csv")
FROZEN_MODEL_PATH = Path("results/experiment3_frozen_composite.joblib")
RESULTS_DIR = Path("results")
SEED = 42
N_BOOT = 1000

# Reference numbers from Experiment 3 (same frozen model, PERSUADE-domain FPR)
PERSUADE_OVERALL_FPR = 0.0100
PERSUADE_ELL_FPR = 0.0156
PERSUADE_NON_ELL_FPR = 0.0094
RAID_ABSTRACTS_FPR = 0.3692


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    out = {name: texts.apply(fn) for name, fn in feat.TIER1_FEATURES.items()}
    return pd.DataFrame(out, index=texts.index)


def fpr_with_ci(scores: np.ndarray, threshold: float) -> tuple:
    return bootstrap_stat_ci([scores], lambda s: rate_at_threshold(s, threshold), n_boot=N_BOOT, seed=SEED)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ellipse_hash = sha256_of(ELLIPSE_PATH)

    df = pd.read_csv(ELLIPSE_PATH)
    df["word_count"] = df["full_text"].str.split().str.len()
    print(f"ELLIPSE: {len(df)} essays, {df['prompt'].nunique()} unique prompts")

    persuade_prompts = pd.read_csv(
        "data/persuade_2.0_human_scores_demo_id_github.csv", usecols=["prompt_name"]
    )["prompt_name"].unique()
    same_task_mask = df["prompt"].isin(persuade_prompts)
    print(f"Same-task (prompt overlaps PERSUADE): {same_task_mask.sum()} essays across "
          f"{df.loc[same_task_mask, 'prompt'].nunique()} prompts")
    print(f"Different-task (unseen prompts): {(~same_task_mask).sum()} essays across "
          f"{df.loc[~same_task_mask, 'prompt'].nunique()} prompts")

    frozen = joblib.load(FROZEN_MODEL_PATH)  # our own artifact from experiment3_separation.py, not external/untrusted
    X = compute_feature_matrix(df["full_text"]).to_numpy()
    Xs = frozen["scaler"].transform(X)
    df["p_ai_score"] = frozen["model"].predict_proba(Xs)[:, 1]
    threshold = frozen["threshold"]

    rows = []
    for label, mask in [
        ("ellipse_overall", pd.Series(True, index=df.index)),
        ("ellipse_same_task_as_persuade", same_task_mask),
        ("ellipse_different_task", ~same_task_mask),
    ]:
        scores = df.loc[mask, "p_ai_score"].to_numpy()
        fpr, lo, hi = fpr_with_ci(scores, threshold)
        rows.append({"subset": label, "n": int(mask.sum()), "fpr": fpr, "fpr_ci_low": lo, "fpr_ci_high": hi})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "ellipse_neargenre_fpr.csv", index=False)
    print("\nELLIPSE FPR at Experiment 3's frozen threshold:\n" + results_df.to_string(index=False))
    print(f"\nReference points (same frozen threshold, from prior experiments):")
    print(f"  PERSUADE overall FPR:  {PERSUADE_OVERALL_FPR:.4f}")
    print(f"  PERSUADE ELL FPR:      {PERSUADE_ELL_FPR:.4f}")
    print(f"  PERSUADE non-ELL FPR:  {PERSUADE_NON_ELL_FPR:.4f}")
    print(f"  RAID abstracts FPR:    {RAID_ABSTRACTS_FPR:.4f}  (far genre)")

    plot_results(results_df)

    manifest = {
        "ellipse_path": str(ELLIPSE_PATH), "ellipse_sha256": ellipse_hash,
        "n_essays": len(df), "n_prompts": int(df["prompt"].nunique()),
        "n_same_task": int(same_task_mask.sum()), "n_different_task": int((~same_task_mask).sum()),
        "same_task_prompts": sorted(df.loc[same_task_mask, "prompt"].unique().tolist()),
        "frozen_threshold": threshold, "seed": SEED, "n_bootstrap": N_BOOT,
        "reference_points": {
            "persuade_overall_fpr": PERSUADE_OVERALL_FPR,
            "persuade_ell_fpr": PERSUADE_ELL_FPR,
            "persuade_non_ell_fpr": PERSUADE_NON_ELL_FPR,
            "raid_abstracts_fpr": RAID_ABSTRACTS_FPR,
        },
        "note": (
            "ELLIPSE is entirely ELL writers, so its FPR is a second, independent read "
            "on the ELL fairness question alongside Experiment 3's PERSUADE ELL/non-ELL "
            "split, not just a genre-shift check."
        ),
    }
    with open(RESULTS_DIR / "ellipse_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'ellipse_manifest.json'}")


def plot_results(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results_df["subset"]) + ["persuade_overall", "persuade_ell", "persuade_non_ell", "raid_abstracts_FAR_GENRE"]
    values = list(results_df["fpr"]) + [PERSUADE_OVERALL_FPR, PERSUADE_ELL_FPR, PERSUADE_NON_ELL_FPR, RAID_ABSTRACTS_FPR]
    errors_lo = list(results_df["fpr"] - results_df["fpr_ci_low"]) + [0, 0, 0, 0]
    errors_hi = list(results_df["fpr_ci_high"] - results_df["fpr"]) + [0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 2))
    y_pos = np.arange(len(labels))
    colors = ["black"] * len(results_df) + ["gray", "gray", "gray", "red"]
    ax.barh(y_pos, values, xerr=[errors_lo, errors_hi], color=colors, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.axvline(0.01, linestyle="--", color="blue", label="1% target FPR")
    ax.set_xlabel("FPR at Experiment 3's frozen threshold")
    ax.set_title("Near-genre (ELLIPSE) vs far-genre (RAID abstracts) FPR degradation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ellipse_neargenre_fpr.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'ellipse_neargenre_fpr.png'}")


if __name__ == "__main__":
    main()
