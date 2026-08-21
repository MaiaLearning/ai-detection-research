"""Experiment 1, Tier-2 arm (GATE): does a model-based zero-shot detector
(Binoculars) predict ELL status on the same matched sample Tier-1 was
tested on?

RESEARCH_PLAN.md, "1. Does the score predict ELL status? (GATE)" pre-
registers Binoculars/Fast-DetectGPT as a Tier-2 arm alongside the Tier-1
deterministic features. This arm was not run when Experiment 1 was first
reported -- disclosed as dropped in AMENDMENTS.md, then run on local GPU
rather than left disclosed-but-dropped.

Reuses the exact same matched sample as scripts/experiment1_ell_gate.py
(caliper=0.15, exact match on prompt_name/grade_level, distance match on
z_logwc/z_holistic -- see results/experiment1_manifest.json), so any AUC
difference between the two arms reflects the detector, not a different
comparison population.

Model substitution: tiiuae/falcon-7b (observer) / tiiuae/falcon-7b-instruct
(performer) from the original paper do not fit this machine's 6GB VRAM
budget. Substitutes Qwen/Qwen2.5-0.5B (observer, base) /
Qwen/Qwen2.5-0.5B-Instruct (performer, instruction-tuned) -- a same-
tokenizer-family base/instruct pair, matching the original paper's own
Falcon base/instruct pairing convention, at a fraction of the parameter
count. RESEARCH_PLAN.md explicitly permits this ("Substitute smaller
models if VRAM is tight and record which"). Binoculars' own published
thresholds (BINOCULARS_ACCURACY_THRESHOLD / BINOCULARS_FPR_THRESHOLD) are
fit specifically to the Falcon pair and are not reused here -- this script
only needs the score's ranking ability (AUC), not a calibrated accept/
reject threshold.

Usage: uv run python scripts/experiment1b_tier2_ell_gate.py
Requires: data/persuade_2.0_human_scores_demo_id_github.csv, a CUDA GPU
(falls back to CPU if unavailable, but very slow for ~3600 essays)
Output: results/experiment1b_tier2_auc.csv, results/experiment1b_tier2_scores.csv,
        results/experiment1b_tier2_manifest.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import binoculars_score as bs
from src.data import add_standardized_covariates, load_and_clean, sha256_of
from src.matching import nearest_neighbor_match
from src.stats_utils import bootstrap_auc_ci

SEED = 42
N_BOOT = 2000
CORPUS_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
RESULTS_DIR = Path("results")
CALIPER = 0.15  # chosen by scripts/experiment1_ell_gate.py's sweep; see results/experiment1_manifest.json
GATE_FAIL_AUC = 0.65
OBSERVER_MODEL = "Qwen/Qwen2.5-0.5B"
PERFORMER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512
TIER1_COMBINED_AUC = 0.601  # results/experiment1_auc.csv, combined_model row -- for direct comparison
TIER1_COMBINED_CI = [0.583, 0.619]


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


def gate_orient(raw_auc, lo, hi):
    """Reorient AUC/CI so the reported number reflects discrimination
    strength regardless of direction, matching experiment1_ell_gate.py's
    convention -- the plan's ">0.65" gate is direction-agnostic."""
    if raw_auc >= 0.5:
        return raw_auc, lo, hi
    return 1 - raw_auc, 1 - hi, 1 - lo


def score_one(text, tokenizer, observer, performer, device, max_tokens):
    encoding = tokenizer(
        [text], return_tensors="pt", truncation=True,
        max_length=max_tokens, return_token_type_ids=False,
    ).to(device)
    with torch.no_grad():
        observer_logits = observer(**encoding, use_cache=False).logits
        performer_logits = performer(**encoding, use_cache=False).logits
        score = bs.binoculars_score(
            observer_logits, performer_logits,
            encoding["input_ids"], encoding["attention_mask"],
        )
    result = score.item()
    del encoding, observer_logits, performer_logits, score
    return result


def score_texts(texts, tokenizer, observer, performer, device) -> np.ndarray:
    # One essay at a time: a smoke test confirmed a full 512-token essay
    # peaks at ~2.8GB combined for this model pair, well within the 6GB
    # card's budget, whereas batching (even at 8) exhausted it because the
    # ~152k-token Qwen vocabulary makes each position's logits/softmax
    # buffers large relative to model size.
    scores = []
    for i, text in enumerate(texts):
        try:
            result = score_one(text, tokenizer, observer, performer, device, MAX_TOKENS)
        except torch.cuda.OutOfMemoryError:
            # A plausible real constraint on this card for an unusually
            # long essay -- retry once at half the token budget rather
            # than failing the whole run.
            torch.cuda.empty_cache()
            print(f"\n  OOM on essay {i} at {MAX_TOKENS} tokens, retrying at {MAX_TOKENS // 2}...")
            result = score_one(text, tokenizer, observer, performer, device, MAX_TOKENS // 2)
        scores.append(result)
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0 or i + 1 == len(texts):
            print(f"  scored {i + 1}/{len(texts)}", end="\r")
    print()
    return np.array(scores)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    corpus_hash = sha256_of(CORPUS_PATH)

    df, _ = load_and_clean(CORPUS_PATH)
    df = add_standardized_covariates(df)
    matched = build_matched_sample(df, CALIPER)
    n_pairs = matched["pair_id"].nunique()
    print(f"Matched sample: {n_pairs} pairs ({len(matched)} essays) -- caliper={CALIPER}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(OBSERVER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    observer = AutoModelForCausalLM.from_pretrained(OBSERVER_MODEL, torch_dtype=dtype).to(device).eval()
    performer = AutoModelForCausalLM.from_pretrained(PERFORMER_MODEL, torch_dtype=dtype).to(device).eval()

    texts = matched["full_text"].tolist()
    print(f"Scoring {len(texts)} essays with Binoculars ({OBSERVER_MODEL} / {PERFORMER_MODEL})...")
    binoculars_scores = score_texts(texts, tokenizer, observer, performer, device)
    matched = matched.copy()
    matched["binoculars_score"] = binoculars_scores

    y = matched["is_ell"].to_numpy()
    # Lower Binoculars score conventionally means "more likely AI-generated";
    # here it's being tested for whether it happens to also track ELL
    # status, so the sign of that relationship isn't meaningful on its own.
    # gate_orient makes the reported AUC direction-agnostic, matching Tier-1's
    # per-feature rows in experiment1_ell_gate.py.
    raw_auc, lo, hi = bootstrap_auc_ci(y, binoculars_scores, n_boot=N_BOOT, seed=SEED)
    gate_auc, gate_lo, gate_hi = gate_orient(raw_auc, lo, hi)
    verdict = "FAIL (tracks ELL status)" if gate_auc > GATE_FAIL_AUC else "pass"

    result_row = {
        "feature": f"binoculars_score ({OBSERVER_MODEL} / {PERFORMER_MODEL})",
        "n_pairs": n_pairs,
        "raw_auc": raw_auc, "raw_ci_low": lo, "raw_ci_high": hi,
        "gate_auc": gate_auc, "gate_ci_low": gate_lo, "gate_ci_high": gate_hi,
        "verdict": verdict,
    }
    result_df = pd.DataFrame([result_row])
    print("\n" + result_df.to_string(index=False))
    result_df.to_csv(RESULTS_DIR / "experiment1b_tier2_auc.csv", index=False)

    scores_out = matched.reset_index()[["essay_id_comp", "pair_id", "is_ell", "binoculars_score"]]
    scores_out.to_csv(RESULTS_DIR / "experiment1b_tier2_scores.csv", index=False)

    manifest_out = {
        "seed": SEED, "n_bootstrap": N_BOOT,
        "corpus_path": str(CORPUS_PATH), "corpus_sha256": corpus_hash,
        "caliper": CALIPER, "n_matched_pairs": n_pairs,
        "exact_match_columns": ["prompt_name", "grade_level"],
        "caliper_distance_columns": ["z_logwc", "z_holistic"],
        "pre_registration": (
            "RESEARCH_PLAN.md Experiment 1: Tier-2 model-based detector "
            "(Binoculars/Fast-DetectGPT) arm. Not run when Experiment 1 was "
            "first reported; disclosed as dropped in AMENDMENTS.md, then run "
            "on local GPU rather than left unrun -- this file is that run."
        ),
        "detector": "Binoculars (Hans et al., 2024, ICML)",
        "model_substitution": {
            "original_paper_pair": "tiiuae/falcon-7b (observer) / tiiuae/falcon-7b-instruct (performer)",
            "substituted_pair": f"{OBSERVER_MODEL} (observer) / {PERFORMER_MODEL} (performer)",
            "reason": (
                "6GB VRAM budget on the available machine; RESEARCH_PLAN.md "
                "explicitly permits substituting smaller models and recording "
                "which. Falcon-7B-fit thresholds (BINOCULARS_ACCURACY_THRESHOLD / "
                "BINOCULARS_FPR_THRESHOLD) from the original paper are not reused "
                "here -- only the score's ranking ability (AUC) is needed for "
                "this gate, not a calibrated accept/reject threshold."
            ),
        },
        "max_tokens_per_essay": MAX_TOKENS,
        "device": device,
        "gate_fail_auc_threshold": GATE_FAIL_AUC,
        "result": result_row,
        "tier1_combined_model_comparison": {
            "tier1_gate_auc": TIER1_COMBINED_AUC, "tier1_gate_ci": TIER1_COMBINED_CI,
            "source": "results/experiment1_auc.csv, combined_model row",
        },
        "overall_gate_verdict": "FAIL" if verdict != "pass" else "PASS",
    }
    with open(RESULTS_DIR / "experiment1b_tier2_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment1b_tier2_manifest.json'}")
    print(f"Overall gate verdict: {manifest_out['overall_gate_verdict']}")
    print(f"Tier-1 combined-model AUC for comparison: {TIER1_COMBINED_AUC} CI {TIER1_COMBINED_CI}")


if __name__ == "__main__":
    main()
