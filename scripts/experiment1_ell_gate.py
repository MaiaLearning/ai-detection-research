"""Experiment 1 (GATE): Does the texture score predict ELL status?

RESEARCH_PLAN.md, "1. Does the score predict ELL status? (GATE)"

Human PERSUADE 2.0 essays only. ELL and non-ELL essays are matched on grade
level, prompt, word count, and holistic score so that any AUC we measure
reflects the feature itself, not a length/quality/grade confound. Per
feature, and for a combined model, we report AUC of predicting ELL status
with a bootstrap confidence interval.

Usage: uv run python scripts/experiment1_ell_gate.py
Output: results/experiment1_auc.csv, results/experiment1_auc.png,
        results/experiment1_match_diagnostics.csv,
        results/experiment1_manifest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat
from src.data import add_standardized_covariates, load_and_clean, sha256_of
from src.matching import nearest_neighbor_match
from src.stats_utils import bootstrap_auc_ci, compute_auc

SEED = 42
N_BOOT = 2000
CORPUS_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
RESULTS_DIR = Path("results")
CALIPER_CANDIDATES = [0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]
SMD_BALANCE_THRESHOLD = 0.1  # Austin (2011): |SMD| < 0.1 is "well balanced"
GATE_FAIL_AUC = 0.65

FEATURE_FUNCS = feat.TIER1_FEATURES


def smd(a: pd.Series, b: pd.Series, pooled_sd: float) -> float:
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def sweep_calipers(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Try each candidate caliper, report retention + balance, pick the
    smallest caliper that achieves |SMD| < SMD_BALANCE_THRESHOLD on both
    covariates; falls back to the best-balance caliper if none qualify."""
    ell = df[df["ell_clean"] == "Yes"]
    pooled_sd_wc = np.sqrt((df.groupby("ell_clean")["word_count"].var(ddof=1)).mean())
    pooled_sd_score = np.sqrt((df.groupby("ell_clean")["holistic_essay_score"].var(ddof=1)).mean())

    rows = []
    for caliper in CALIPER_CANDIDATES:
        pairs = nearest_neighbor_match(
            df, id_col="essay_id_comp", treatment_col="ell_clean",
            treatment_value="Yes", control_value="No",
            exact_cols=["prompt_name", "grade_level"],
            distance_cols=["z_logwc", "z_holistic"], caliper=caliper,
        )
        n_matched = len(pairs)
        if n_matched == 0:
            rows.append({"caliper": caliper, "n_matched": 0, "pct_ell_retained": 0.0,
                         "smd_word_count": np.nan, "smd_holistic_score": np.nan})
            continue
        matched_ell = df.set_index("essay_id_comp").loc[pairs["essay_id_comp_treated"]]
        matched_ctrl = df.set_index("essay_id_comp").loc[pairs["essay_id_comp_control"]]
        rows.append({
            "caliper": caliper,
            "n_matched": n_matched,
            "pct_ell_retained": n_matched / len(ell) * 100,
            "smd_word_count": smd(matched_ell["word_count"], matched_ctrl["word_count"], pooled_sd_wc),
            "smd_holistic_score": smd(matched_ell["holistic_essay_score"], matched_ctrl["holistic_essay_score"], pooled_sd_score),
        })

    sweep_df = pd.DataFrame(rows)
    balanced = sweep_df[
        (sweep_df["smd_word_count"].abs() < SMD_BALANCE_THRESHOLD)
        & (sweep_df["smd_holistic_score"].abs() < SMD_BALANCE_THRESHOLD)
        & (sweep_df["n_matched"] > 0)
    ]
    if not balanced.empty:
        chosen = balanced.iloc[0]["caliper"]
    else:
        sweep_df["max_abs_smd"] = sweep_df[["smd_word_count", "smd_holistic_score"]].abs().max(axis=1)
        chosen = sweep_df.loc[sweep_df["max_abs_smd"].idxmin(), "caliper"]
    return sweep_df, float(chosen)


def build_matched_sample(df: pd.DataFrame, caliper: float) -> pd.DataFrame:
    pairs = nearest_neighbor_match(
        df, id_col="essay_id_comp", treatment_col="ell_clean",
        treatment_value="Yes", control_value="No",
        exact_cols=["prompt_name", "grade_level"],
        distance_cols=["z_logwc", "z_holistic"], caliper=caliper,
    )
    indexed = df.set_index("essay_id_comp")
    treated = indexed.loc[pairs["essay_id_comp_treated"]].copy()
    treated["pair_id"] = range(len(treated))
    control = indexed.loc[pairs["essay_id_comp_control"]].copy()
    control["pair_id"] = range(len(control))
    matched = pd.concat([treated, control], ignore_index=False)
    matched["is_ell"] = (matched["ell_clean"] == "Yes").astype(int)
    return matched


def compute_feature_matrix(matched: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, fn in FEATURE_FUNCS.items():
        out[name] = matched["full_text"].apply(fn)
    return pd.DataFrame(out, index=matched.index)


def gate_orient(raw_auc, lo, hi):
    """Reorient AUC/CI so the reported number reflects discrimination
    strength regardless of direction (plan's ">0.65" gate is direction-
    agnostic — a feature that anti-predicts ELL is just as disqualifying)."""
    if raw_auc >= 0.5:
        return raw_auc, lo, hi
    return 1 - raw_auc, 1 - hi, 1 - lo


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    corpus_hash = sha256_of(CORPUS_PATH)

    df, manifest = load_and_clean(CORPUS_PATH)
    df = add_standardized_covariates(df)

    sweep_df, chosen_caliper = sweep_calipers(df)
    sweep_df.to_csv(RESULTS_DIR / "experiment1_match_diagnostics.csv", index=False)
    print(f"Caliper sweep written. Chosen caliper: {chosen_caliper}")
    print(sweep_df.to_string(index=False))

    matched = build_matched_sample(df, chosen_caliper)
    n_pairs = matched["pair_id"].nunique()
    print(f"\nMatched sample: {n_pairs} pairs ({len(matched)} essays)")

    features_df = compute_feature_matrix(matched)
    y = matched["is_ell"].to_numpy()

    rows = []
    for col in features_df.columns:
        raw_auc, lo, hi = bootstrap_auc_ci(y, features_df[col].to_numpy(), n_boot=N_BOOT, seed=SEED)
        gate_auc, gate_lo, gate_hi = gate_orient(raw_auc, lo, hi)
        verdict = "FAIL (tracks ELL status)" if gate_auc > GATE_FAIL_AUC else "pass"
        rows.append({
            "feature": col, "n_pairs": n_pairs,
            "raw_auc": raw_auc, "raw_ci_low": lo, "raw_ci_high": hi,
            "gate_auc": gate_auc, "gate_ci_low": gate_lo, "gate_ci_high": gate_hi,
            "verdict": verdict,
        })

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.to_numpy())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(
        LogisticRegression(max_iter=1000, random_state=SEED), X, y, cv=cv, method="predict_proba",
    )[:, 1]
    raw_auc, lo, hi = bootstrap_auc_ci(y, oof_proba, n_boot=N_BOOT, seed=SEED)
    gate_auc, gate_lo, gate_hi = gate_orient(raw_auc, lo, hi)
    rows.append({
        "feature": "combined_model (logistic regression, 5-fold CV)", "n_pairs": n_pairs,
        "raw_auc": raw_auc, "raw_ci_low": lo, "raw_ci_high": hi,
        "gate_auc": gate_auc, "gate_ci_low": gate_lo, "gate_ci_high": gate_hi,
        "verdict": "FAIL (tracks ELL status)" if gate_auc > GATE_FAIL_AUC else "pass",
    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "experiment1_auc.csv", index=False)
    print("\n" + results_df.to_string(index=False))

    plot_results(results_df)

    manifest.update({
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_hash,
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "chosen_caliper": chosen_caliper,
        "smd_balance_threshold": SMD_BALANCE_THRESHOLD,
        "gate_fail_auc_threshold": GATE_FAIL_AUC,
        "n_matched_pairs": n_pairs,
        "exact_match_columns": ["prompt_name", "grade_level"],
        "caliper_distance_columns": ["z_logwc (standardized log1p word_count)", "z_holistic (standardized holistic_essay_score)"],
        "features": list(FEATURE_FUNCS.keys()),
        "overall_gate_verdict": "FAIL" if (results_df["verdict"] != "pass").any() else "PASS",
    })
    with open(RESULTS_DIR / "experiment1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment1_manifest.json'}")
    print(f"Overall gate verdict: {manifest['overall_gate_verdict']}")


def plot_results(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(results_df) + 2))
    y_pos = np.arange(len(results_df))
    errors = np.vstack([
        results_df["gate_auc"] - results_df["gate_ci_low"],
        results_df["gate_ci_high"] - results_df["gate_auc"],
    ])
    ax.errorbar(results_df["gate_auc"], y_pos, xerr=errors, fmt="o", color="black", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df["feature"])
    ax.axvline(0.5, linestyle="--", color="gray", label="no signal (0.5)")
    ax.axvline(0.65, linestyle="--", color="red", label="gate fail threshold (0.65)")
    ax.set_xlabel("AUC vs ELL status (direction-agnostic, with 95% bootstrap CI)")
    ax.set_title("Experiment 1 (GATE): does each feature predict ELL status?")
    ax.set_xlim(0.4, 1.0)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment1_auc.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment1_auc.png'}")


if __name__ == "__main__":
    main()
