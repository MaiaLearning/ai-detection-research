"""Experiment 5: Does detectability measure prompting effort or model identity?

EXPERIMENT_5.md, adapted per conversation record (see manifest for full
detail):

- L2 DROPPED. Traced DAIGT-v2's description to its listed source datasets
  (darraghdog/Claude via a rendered discussion thread + the raw CSV,
  alejopaullier/chat_gpt_moth's "instructions" column, radek1, nbroad's
  Llama/Falcon description, kingki19's linked Colab notebook) — none
  document a template with persona + word-count target + "write as a
  student" instruction. Every one is ~L1-equivalent (the raw or lightly
  wrapped assignment text). Per the document's own rule ("if a level
  cannot be sourced, log that and stop"), and per user direction, L2 is
  dropped rather than invented.
- L3 sourced from Lu et al. 2024 (TMLR), "Large Language Models can be
  Guided to Evade AI-Generated Text Detection", Table 11: a paraphrase
  prompt the paper itself traces to a real 2023 YouTube video ("ChatGPT -
  pass detection 100% human written with this prompt"), not authored by
  this project. Implemented as a two-stage generation: L1 essay, then
  paraphrased with that exact prompt.
- Single vendor this round: Claude (us.anthropic.claude-sonnet-5 via
  Bedrock, same model as Experiment 3), per user direction. OpenAI arm
  not run.
- 7 of PERSUADE's 8 independent-task prompts are usable. "Phones and
  driving" has no ell_status/grade_level anywhere in the corpus — all
  1,168 of its rows are dropped by this project's existing cleaning
  criteria (src/data.py), used consistently across every experiment.

FREE CHECK (run first, before any generation): claude_sonnet_5_bedrock's
existing Experiment 3 TPR split by task type is Independent 75.5% vs
Text-dependent 70.1% (gap 5.4pp, 95% CI -0.1pp to 11.2pp) — small and in
the OPPOSITE direction from the "text-dependent essays are more generic,
therefore more detectable" confound hypothesis. Restricting to independent
prompts does not lower the baseline; this experiment's scope does not
narrow on that basis.

PRE-REGISTERED PREDICTION (recorded before generation or analysis): TPR
declines monotonically from L0 to L1 to L3. Given the free check and the
L2 finding above (DAIGT's real prompts are already ~L1-level, showing no
sign of deliberate "student-voice" engineering), L0 and L1 are expected to
be close to each other, with the real drop concentrated at L3
(paraphrase-for-evasion). If L0/L1 differ sharply, or L3 does not drop,
that contradicts this framing and is reported plainly either way.

Usage: uv run python scripts/experiment5_prompt_effort.py
Output: results/experiment5_essays.csv, results/experiment5_summary.csv,
        results/experiment5_manifest.json, results/experiment5_tpr_by_level.png
"""
import functools
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat
from src.data import load_and_clean
from src.rejection_sampling import RejectionSampler, bin_targets, compute_quantile_bins
from src.stats_utils import bootstrap_stat_ci, partial_spearman, rate_at_threshold

MODEL_ID = "us.anthropic.claude-sonnet-5"
REGION = "us-west-2"
AWS_PROFILE = "ml"
TEMPERATURE = 1.0
MAX_TOKENS = 1600
MAX_WORKERS = 10
SEED = 42
N_BOOT = 1000

PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
FROZEN_MODEL_PATH = Path("results/experiment3_frozen_composite.joblib")
RESULTS_DIR = Path("results")

N_PER_PROMPT_TARGET = 21      # ~150/7 (doc's target, distributed over the 7 usable prompts)
N_QUANTILE_BINS = 4
MAX_OVERSAMPLE_FACTOR = 7     # tuned from pilot acceptance rates to land ~$35-40 total;
# give up on a bin after this many raw generations per prompt/level. Per user direction,
# shortfalls (esp. for L0/L1, whose natural output length doesn't span PERSUADE's full
# range) are accepted and reported honestly rather than forcing full N at higher cost.

PRE_REGISTERED_PREDICTION = (
    "TPR declines monotonically from L0 to L1 to L3. L0 and L1 are expected to be "
    "close to each other (both low-effort prompting, consistent with the free-check "
    "and L2-sourcing finding that DAIGT's real generation prompts show no evidence of "
    "deliberate student-voice engineering beyond L1-level). The real drop is expected "
    "at L3 (paraphrase-for-evasion, sourced from Lu et al. 2024 Table 11)."
)

TABLE11_PARAPHRASE_PROMPT = (
    'When it comes to writing content, two factors are crucial, "perplexity" and '
    '"burstiness". Perplexity measures the complexity of text. Separately, burstiness '
    "compares the variations of sentences. Humans tend to write with greater "
    "burstiness, for example, with some longer or complex sentences alongside shorter "
    "ones. AI sentences tend to be more uniform.\n"
    "Paraphrase the following AI sentence to be human-like, with a good amount of "
    "perplexity and burstiness:\n"
    "Orig: {text}\n"
    "New:"
)


def l0_prompt(prompt_name: str) -> str:
    return f"write me an essay about {prompt_name.lower()}"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def call_claude(client, text: str) -> dict:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": text}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
    )
    blocks = [b["text"] for b in response["output"]["message"]["content"] if "text" in b]
    if not blocks:
        raise ValueError("Empty response content — no text block returned.")
    return {"text": "".join(blocks), "usage": response["usage"]}


def generate_essay(client, level: str, prompt_name: str, assignment: str) -> dict:
    timestamp = time.time()
    if level == "L0":
        prompt_text = l0_prompt(prompt_name)
        result = call_claude(client, prompt_text)
        return {
            "level": level, "prompt_name": prompt_name, "prompt_text_hash": prompt_hash(prompt_text),
            "full_text": result["text"], "temperature": TEMPERATURE, "timestamp": timestamp,
            "input_tokens": result["usage"]["inputTokens"], "output_tokens": result["usage"]["outputTokens"],
        }
    elif level == "L1":
        prompt_text = assignment
        result = call_claude(client, prompt_text)
        return {
            "level": level, "prompt_name": prompt_name, "prompt_text_hash": prompt_hash(prompt_text),
            "full_text": result["text"], "temperature": TEMPERATURE, "timestamp": timestamp,
            "input_tokens": result["usage"]["inputTokens"], "output_tokens": result["usage"]["outputTokens"],
        }
    elif level == "L3":
        l1_result = call_claude(client, assignment)
        paraphrase_prompt = TABLE11_PARAPHRASE_PROMPT.format(text=l1_result["text"])
        l3_result = call_claude(client, paraphrase_prompt)
        return {
            "level": level, "prompt_name": prompt_name, "prompt_text_hash": prompt_hash(paraphrase_prompt),
            "full_text": l3_result["text"], "temperature": TEMPERATURE, "timestamp": timestamp,
            "input_tokens": l1_result["usage"]["inputTokens"] + l3_result["usage"]["inputTokens"],
            "output_tokens": l1_result["usage"]["outputTokens"] + l3_result["usage"]["outputTokens"],
            "l1_intermediate_text": l1_result["text"],
        }
    raise ValueError(level)


def generate_cell(client, level: str, prompt_name: str, assignment: str, edges: np.ndarray, targets: list) -> tuple:
    """Generates raw essays concurrently in batches, offering each to the
    rejection sampler as it completes. Concurrency is essential here: each
    Bedrock call takes ~15-25s, and rejection sampling can need several
    attempts per bin — strictly serial generation makes even a small pilot
    impractically slow (measured: ~50 minutes for a 3-per-prompt pilot)."""
    sampler = RejectionSampler(edges, targets)
    accepted_rows = []
    max_offers = MAX_OVERSAMPLE_FACTOR * sum(targets)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        while not sampler.is_full() and sampler.n_offered < max_offers:
            batch_size = min(MAX_WORKERS, max_offers - sampler.n_offered)
            futures = [pool.submit(generate_essay, client, level, prompt_name, assignment) for _ in range(batch_size)]
            for future in as_completed(futures):
                row = future.result()
                row["word_count"] = len(row["full_text"].split())
                accepted, bin_idx = sampler.offer(row["word_count"])
                row["bin_idx"] = bin_idx
                row["rejection_sampled_accept"] = accepted
                if accepted:
                    accepted_rows.append(row)
    return accepted_rows, sampler


def compute_feature_matrix(texts: pd.Series) -> pd.DataFrame:
    out = {name: texts.apply(fn) for name, fn in feat.TIER1_FEATURES.items()}
    return pd.DataFrame(out, index=texts.index)


def ngram_overlap(text: str, reference_texts: list, n: int = 5) -> float:
    def ngrams(s):
        words = s.lower().split()
        return set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    text_ngrams = ngrams(text)
    if not text_ngrams:
        return 0.0
    ref_ngrams = set()
    for r in reference_texts:
        ref_ngrams |= ngrams(r)
    return len(text_ngrams & ref_ngrams) / len(text_ngrams)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("PRE-REGISTERED PREDICTION:", PRE_REGISTERED_PREDICTION)

    human, human_manifest = load_and_clean(PERSUADE_PATH)
    indep = human[human["task"] == "Independent"]
    prompt_counts = indep["prompt_name"].value_counts()
    print(f"\nUsable independent-task prompts: {len(prompt_counts)} (of 8; "
          f"'Phones and driving' excluded, see manifest)")

    # Self-generated artifact from this repo's own experiment3_separation.py run, not an
    # external/untrusted source.
    frozen = joblib.load(FROZEN_MODEL_PATH)
    print(f"Loaded frozen composite (threshold={frozen['threshold']:.4f}, "
          f"trained on {frozen['trained_on_n_human']} human + {frozen['trained_on_n_ai']} AI essays)")

    prompts_df = indep[["prompt_name", "assignment"]].drop_duplicates().reset_index(drop=True)
    targets_per_bin = bin_targets(N_PER_PROMPT_TARGET, N_QUANTILE_BINS)

    session = boto3.Session(profile_name=AWS_PROFILE)
    client = session.client(
        "bedrock-runtime", region_name=REGION,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    all_rows = []
    sampler_diagnostics = []
    for level in ["L0", "L1", "L3"]:
        for _, prow in prompts_df.iterrows():
            prompt_name, assignment = prow["prompt_name"], prow["assignment"]
            ref_wc = indep.loc[indep["prompt_name"] == prompt_name, "word_count"]
            edges = compute_quantile_bins(ref_wc, N_QUANTILE_BINS)
            rows, sampler = generate_cell(client, level, prompt_name, assignment, edges, targets_per_bin)
            for r in rows:
                r["essay_id"] = f"exp5_{level}_{prompt_name.replace(' ', '_')[:20]}_{len(all_rows)}"
                all_rows.append(r)
            sampler_diagnostics.append({
                "level": level, "prompt_name": prompt_name,
                "n_offered": sampler.n_offered, "n_accepted": sampler.n_accepted,
                "discard_rate": sampler.discard_rate, "hit_target": sampler.is_full(),
            })
            print(f"  {level} / {prompt_name}: accepted {sampler.n_accepted}/{sum(targets_per_bin)}, "
                  f"offered {sampler.n_offered}, discard_rate={sampler.discard_rate:.2f}")

    essays_df = pd.DataFrame(all_rows)
    diagnostics_df = pd.DataFrame(sampler_diagnostics)
    diagnostics_df.to_csv(RESULTS_DIR / "experiment5_sampling_diagnostics.csv", index=False)
    print("\nSampling diagnostics:\n" + diagnostics_df.to_string(index=False))

    # Contamination check: n-gram overlap against the prompt's own human PERSUADE essays
    contamination_rows = []
    for prompt_name, group in essays_df.groupby("prompt_name"):
        human_texts = indep.loc[indep["prompt_name"] == prompt_name, "full_text"].tolist()
        for _, row in group.iterrows():
            overlap = ngram_overlap(row["full_text"], human_texts)
            contamination_rows.append({"essay_id": row["essay_id"], "ngram5_overlap_vs_human": overlap})
    contamination_df = pd.DataFrame(contamination_rows)
    essays_df = essays_df.merge(contamination_df, on="essay_id")
    max_overlap = essays_df["ngram5_overlap_vs_human"].max()
    print(f"\nContamination check: max 5-gram overlap vs same-prompt human essays = {max_overlap:.4f}")

    # Score with the FROZEN composite (no refitting)
    X = compute_feature_matrix(essays_df["full_text"]).to_numpy()
    Xs = frozen["scaler"].transform(X)
    essays_df["p_ai_score"] = frozen["model"].predict_proba(Xs)[:, 1]
    essays_df.drop(columns=["l1_intermediate_text"], errors="ignore").to_csv(
        RESULTS_DIR / "experiment5_essays.csv", index=False
    )

    threshold = frozen["threshold"]
    summary_rows = []
    for level, group in essays_df.groupby("level"):
        tpr = rate_at_threshold(group["p_ai_score"].to_numpy(), threshold)
        summary_rows.append({"level": level, "n": len(group), "tpr_at_frozen_threshold": tpr})
    summary_df = pd.DataFrame(summary_rows).set_index("level").loc[["L0", "L1", "L3"]].reset_index()
    print("\nTPR by level (frozen threshold, no refitting):\n" + summary_df.to_string(index=False))

    # Gate 2 reference number (human-side, invariant across cells; see docstring)
    indep_scores_df = pd.DataFrame({
        "full_text": indep["full_text"], "quality": indep["holistic_essay_score"], "word_count": indep["word_count"],
    })
    indep_feats = compute_feature_matrix(indep_scores_df["full_text"])
    indep_p_ai = frozen["model"].predict_proba(frozen["scaler"].transform(indep_feats.to_numpy()))[:, 1]
    gate2_point, gate2_lo, gate2_hi = bootstrap_stat_ci(
        [indep_p_ai, indep_scores_df["quality"].to_numpy(dtype=float), indep_scores_df["word_count"].to_numpy(dtype=float)],
        partial_spearman, n_boot=N_BOOT, seed=SEED,
    )
    print(f"\nGate 2 reference (human, 7-prompt independent subset): partial rho = "
          f"{gate2_point:.3f} ({gate2_lo:.3f}, {gate2_hi:.3f})")

    monotone_held = (
        summary_df.set_index("level").loc["L0", "tpr_at_frozen_threshold"]
        >= summary_df.set_index("level").loc["L1", "tpr_at_frozen_threshold"]
        >= summary_df.set_index("level").loc["L3", "tpr_at_frozen_threshold"]
    )
    print(f"\nPre-registered monotone-decline prediction held: {monotone_held}")

    manifest = {
        **human_manifest,
        "model_id": MODEL_ID, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": SEED,
        "n_per_prompt_target": N_PER_PROMPT_TARGET, "n_quantile_bins": N_QUANTILE_BINS,
        "max_oversample_factor": MAX_OVERSAMPLE_FACTOR,
        "pre_registered_prediction": PRE_REGISTERED_PREDICTION,
        "pre_registered_prediction_held": bool(monotone_held),
        "usable_independent_prompts": sorted(prompts_df["prompt_name"].tolist()),
        "excluded_prompt": "Phones and driving (no ell_status/grade_level in corpus at all)",
        "l2_dropped_reason": (
            "No documented DAIGT source dataset (checked: darraghdog/Claude, "
            "alejopaullier/chat_gpt_moth, radek1, nbroad/Llama-Falcon, kingki19/PaLM) "
            "shows a template beyond L1-level (raw/lightly-wrapped assignment text). "
            "No persona, word-count target, or student-voice instruction is documented "
            "anywhere in the actual generation methodology. Dropped per user direction "
            "rather than invented."
        ),
        "l3_source": (
            "Lu, Liu, He, Ong, Wang, Tang (2024), 'Large Language Models can be Guided "
            "to Evade AI-Generated Text Detection', TMLR, Table 11 ('Human-designed "
            "prompt to evade AI-generated text detection'). That paper traces the prompt "
            "to: Youtube Uploader (2023), 'ChatGPT - pass detection 100% human written "
            "with this prompt', https://www.youtube.com/watch?v=Xgc-d7SO4OQ, and to "
            "GPTZero creator Edward Tian's stated perplexity/burstiness detection "
            "philosophy as reported in The Conversation."
        ),
        "l3_implementation": "Two-stage: generate via L1, then paraphrase that output with the Table 11 prompt.",
        "frozen_composite_path": str(FROZEN_MODEL_PATH),
        "frozen_threshold": threshold,
        "max_ngram5_contamination_overlap": float(max_overlap),
        "gate2_reference_human_independent_subset": {
            "partial_rho": gate2_point, "ci_low": gate2_lo, "ci_high": gate2_hi,
            "note": "Computed once (human 7-prompt independent subset vs frozen composite score, "
                    "controlling word count); invariant across cells, not literally per-cell.",
        },
        "note_full_scale_run": (
            f"N_PER_PROMPT_TARGET={N_PER_PROMPT_TARGET} (~150/cell target across 7 usable "
            f"prompts), MAX_OVERSAMPLE_FACTOR={MAX_OVERSAMPLE_FACTOR}, tuned from a prior "
            "pilot's measured per-offer acceptance rates to land ~$35-40 total. Per user "
            "direction, shortfalls below the 150/cell target (expected for L0/L1, whose "
            "natural output length doesn't span PERSUADE's full per-prompt range) are "
            "accepted and reported honestly via the discard-rate/hit_target diagnostics, "
            "rather than spending further to force the full target."
        ),
    }
    with open(RESULTS_DIR / "experiment5_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment5_manifest.json'}")


if __name__ == "__main__":
    main()
