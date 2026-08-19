"""Experiment 4: Does the composite survive adversarial editing?

RESEARCH_PLAN.md, "4. Does it survive paraphrasing?" — uses RAID
(`liamdugan/raid`, ACL 2024), streamed and filtered rather than bulk
downloaded (RAID-train is 11.8GB with adversarial attacks included; per
CLAUDE.md, "ask before pulling multi-gigabyte datasets... do not download
the whole thing").

Scope, and why: RAID has no "essay" domain (its 8 domains are Abstracts,
Books, News, Poetry, Recipes, Reddit, Reviews, Wikipedia). "Abstracts"
(academic writing) is the closest register to formal persuasive essays, so
this experiment uses that domain only — a real domain-shift limitation
from PERSUADE, logged rather than glossed over. Restricted to `decoding
='greedy', repetition_penalty='no'` (RAID's simplest setting) to keep the
pulled subset small and comparisons apples-to-apples.

Of RAID's 11 attacks, this experiment scores the "meaningful style-level"
ones (paraphrase, synonym substitution, misspelling, alternative spelling,
article deletion) plus the unattacked baseline. RAID's remaining attacks
(zero-width space, homoglyph, whitespace insertion, upper/lower swap,
digit shuffling) are unicode/formatting obfuscations that would corrupt
this project's regex-based tokenization (src/features.py) in a way
unrelated to "does the essay read as more human" — scored separately and
reported as a distinct category, not conflated with the style attacks.

No individual document pairing is used (RAID's source_id/adv_source_id
linkage is per-attack-variant, not a simple 1:1 join across all
conditions); instead this compares AGGREGATE TPR per (model, attack)
group, since every attack condition is applied to the same underlying
pool of source documents. The frozen composite from Experiment 3 (scaler
+ model + threshold, no refitting) is used throughout.

Usage: uv run python scripts/experiment4_raid_robustness.py
Output: data/raid_abstracts_subset.csv, results/experiment4_tpr_by_attack.csv,
        results/experiment4_manifest.json, results/experiment4_tpr_by_attack.png
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat
from src.stats_utils import bootstrap_stat_ci, rate_at_threshold

DOMAIN = "abstracts"
DECODING = "greedy"
REPETITION_PENALTY = "no"
STYLE_ATTACKS = ["none", "paraphrase", "synonym", "perplexity_misspelling", "alternative_spelling",
                 "article_deletion", "insert_paragraphs"]
OBFUSCATION_ATTACKS = ["zero_width_space", "homoglyph", "whitespace", "upper_lower", "number"]
MAX_ROWS_PER_ATTACK = 600  # collected across whatever models (+ human) appear for that attack

FROZEN_MODEL_PATH = Path("results/experiment3_frozen_composite.joblib")
OUTPUT_PATH = Path("data/raid_abstracts_subset.csv")
RESULTS_DIR = Path("results")
SEED = 42
N_BOOT = 1000


def keep_row(row: dict, attack: str) -> bool:
    if row["domain"] != DOMAIN or row["attack"] != attack:
        return False
    if row["model"] == "human":
        return True  # keep human rows for every attack too (FPR-under-attack check)
    return row["decoding"] == DECODING and row["repetition_penalty"] == REPETITION_PENALTY


def collect_one_attack(attack: str) -> list:
    from datasets import load_dataset

    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    ds = ds.filter(lambda r: keep_row(r, attack))
    rows = []
    t0 = time.monotonic()
    for row in ds:
        rows.append(row)
        if len(rows) >= MAX_ROWS_PER_ATTACK:
            break
    print(f"  {attack}: collected {len(rows)} rows in {time.monotonic() - t0:.0f}s")
    return rows


def collect_subset(attacks_wanted: list) -> pd.DataFrame:
    all_rows = []
    for attack in attacks_wanted:
        all_rows.extend(collect_one_attack(attack))
    return pd.DataFrame(all_rows)


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    out = {name: texts.apply(fn) for name, fn in feat.TIER1_FEATURES.items()}
    return pd.DataFrame(out, index=texts.index)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Streaming RAID (domain={DOMAIN}, decoding={DECODING}, "
          f"repetition_penalty={REPETITION_PENALTY}, attacks={STYLE_ATTACKS + OBFUSCATION_ATTACKS})...")
    df = collect_subset(STYLE_ATTACKS + OBFUSCATION_ATTACKS)
    df = df[df["generation"].notna() & (df["generation"].str.len() > 0)].reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.groupby(["model", "attack"]).size().to_string())

    frozen = joblib.load(FROZEN_MODEL_PATH)  # our own artifact from experiment3_separation.py, not external/untrusted
    X = compute_feature_matrix(df["generation"]).to_numpy()
    Xs = frozen["scaler"].transform(X)
    df["p_ai_score"] = frozen["model"].predict_proba(Xs)[:, 1]
    threshold = frozen["threshold"]

    human_none = df[(df["model"] == "human") & (df["attack"] == "none")]
    fpr_baseline, fpr_lo, fpr_hi = bootstrap_stat_ci(
        [human_none["p_ai_score"].to_numpy()], lambda s: rate_at_threshold(s, threshold),
        n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nHuman baseline FPR at frozen threshold: {fpr_baseline:.4f} ({fpr_lo:.4f}, {fpr_hi:.4f}), n={len(human_none)}")

    fpr_rows = []
    for attack, group in df[df["model"] == "human"].groupby("attack"):
        fpr, lo, hi = bootstrap_stat_ci(
            [group["p_ai_score"].to_numpy()], lambda s: rate_at_threshold(s, threshold),
            n_boot=N_BOOT, seed=SEED,
        )
        fpr_rows.append({"attack": attack, "n": len(group), "fpr": fpr, "fpr_ci_low": lo, "fpr_ci_high": hi})
    fpr_df = pd.DataFrame(fpr_rows).sort_values("attack")
    fpr_df.to_csv(RESULTS_DIR / "experiment4_fpr_by_attack.csv", index=False)
    print("\nFPR by attack (human essays, frozen threshold):\n" + fpr_df.to_string(index=False))

    rows = []
    for (model, attack), group in df[df["model"] != "human"].groupby(["model", "attack"]):
        tpr, lo, hi = bootstrap_stat_ci(
            [group["p_ai_score"].to_numpy()], lambda s: rate_at_threshold(s, threshold),
            n_boot=N_BOOT, seed=SEED,
        )
        rows.append({
            "model": model, "attack": attack, "n": len(group),
            "category": "style" if attack in STYLE_ATTACKS else "obfuscation",
            "tpr": tpr, "tpr_ci_low": lo, "tpr_ci_high": hi,
        })
    results_df = pd.DataFrame(rows).sort_values(["category", "attack", "model"])
    results_df.to_csv(RESULTS_DIR / "experiment4_tpr_by_attack.csv", index=False)
    print("\nTPR by model x attack:\n" + results_df.to_string(index=False))

    pooled_style = results_df[results_df["attack"].isin(STYLE_ATTACKS)].groupby("attack")["tpr"].mean()
    pooled_obfuscation = results_df[results_df["attack"].isin(OBFUSCATION_ATTACKS)].groupby("attack")["tpr"].mean()
    print(f"\nPooled TPR by style attack (mean across models):\n{pooled_style.to_string()}")
    print(f"\nPooled TPR by obfuscation attack (mean across models):\n{pooled_obfuscation.to_string()}")

    plot_results(results_df)

    manifest = {
        "domain": DOMAIN, "decoding": DECODING, "repetition_penalty": REPETITION_PENALTY,
        "style_attacks": STYLE_ATTACKS, "obfuscation_attacks": OBFUSCATION_ATTACKS,
        "max_rows_per_attack": MAX_ROWS_PER_ATTACK, "seed": SEED, "n_bootstrap": N_BOOT,
        "n_rows_collected": len(df), "frozen_threshold": threshold,
        "human_baseline_fpr": {"value": fpr_baseline, "ci_low": fpr_lo, "ci_high": fpr_hi, "n": len(human_none)},
        "domain_shift_limitation": (
            "RAID has no essay domain; 'abstracts' (academic writing) is the closest "
            "register to PERSUADE's persuasive essays but is still a real domain shift "
            "from the corpus the frozen composite was calibrated on."
        ),
        "obfuscation_attack_caveat": (
            "zero_width_space/homoglyph/whitespace/upper_lower/number attacks corrupt "
            "this project's regex-based word/sentence tokenization directly; a TPR drop "
            "there reflects tokenizer breakage, not 'text reads as more human', and is "
            "reported as a separate category rather than conflated with style attacks."
        ),
        "no_document_level_pairing": (
            "Compares aggregate TPR per (model, attack) group rather than paired "
            "before/after scores per document; RAID's id linkage is per-attack-variant, "
            "not a simple join across all conditions."
        ),
        "single_model_limitation": (
            "Within MAX_ROWS_PER_ATTACK=600 and the domain=abstracts/decoding=greedy/"
            "repetition_penalty=no filter, only 'llama-chat' (107 rows/attack) appeared "
            "among AI models before the cap was hit for every attack; other RAID models "
            "did not surface in this bounded stream. Findings below are single-model, "
            "not the full 11-model RAID coverage — would need a much higher row cap "
            "(more streaming time, still no monetary cost) to broaden model coverage."
        ),
    }
    with open(RESULTS_DIR / "experiment4_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment4_manifest.json'}")


def plot_results(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = results_df.groupby(["attack", "category"])["tpr"].mean().reset_index()
    pooled = pooled.sort_values("tpr")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(pooled) + 2))
    y_pos = np.arange(len(pooled))
    colors = ["black" if c == "style" else "gray" for c in pooled["category"]]
    ax.barh(y_pos, pooled["tpr"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pooled["attack"])
    ax.set_xlabel("TPR at Experiment 3's frozen threshold (mean across models)")
    ax.set_title("Experiment 4: TPR by attack type (RAID, abstracts domain)\nblack=style attack, gray=tokenizer-breaking obfuscation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment4_tpr_by_attack.png", dpi=150)
    print(f"Plot written to {RESULTS_DIR / 'experiment4_tpr_by_attack.png'}")


if __name__ == "__main__":
    main()
