"""Generate a Claude-on-Bedrock matched set for Experiment 3.

RESEARCH_PLAN.md: "generate a matched set via Bedrock with the same Claude
model production uses, on PERSUADE prompts — no other corpus covers our
own generator." Model ID and target count confirmed by the user directly
(anthropic.claude-sonnet-5, 1000 essays, generic prompt — not production's
actual system prompt, which wasn't available to this project).

Limitation logged here rather than silently worked around: 7 of PERSUADE's
15 prompts are "text dependent" and instruct the writer to cite a specific
source article, but the PERSUADE corpus only ships a citation/title for
those sources, not the article text itself. For those 7 prompts, the
instruction below asks Claude to write from general knowledge instead of
citing article details it doesn't have. The other 8 ("Independent" task)
prompts are self-contained and generated as originally worded.

This is a data-generation script, not an experiment script: it makes paid,
non-deterministic API calls and writes new source data. Re-running it
produces a different (but statistically similar) sample, unlike every
other script in this repo.

Usage: uv run python scripts/generate_bedrock_claude_essays.py
Output: data/bedrock_claude_essays.csv, results/bedrock_generation_manifest.json
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.essay_prompts import build_prompt_list, build_task_list

MODEL_ID = "us.anthropic.claude-sonnet-5"  # cross-region inference profile; the bare
# model ID (anthropic.claude-sonnet-5) returns AccessDeniedException for on-demand use
REGION = "us-west-2"
AWS_PROFILE = "ml"  # model access is granted under this profile, not the default
TARGET_N = 1000
MAX_TOKENS = 900
MAX_WORKERS = 8
PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
OUTPUT_PATH = Path("data/bedrock_claude_essays.csv")
RESULTS_DIR = Path("results")


def generate_one(client, task: dict) -> dict:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": task["instruction"]}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS},
    )
    text_blocks = [b["text"] for b in response["output"]["message"]["content"] if "text" in b]
    if not text_blocks:
        raise ValueError(f"No text content block in response: {response['output']['message']['content']}")
    text = "".join(text_blocks)
    usage = response["usage"]
    return {
        **task,
        "full_text": text,
        "word_count": len(text.split()),
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    prompts = build_prompt_list(PERSUADE_PATH, TARGET_N)
    print("Essays per prompt:\n" + prompts[["prompt_name", "task", "n_essays"]].to_string(index=False))
    tasks = build_task_list(prompts)
    print(f"\nGenerating {len(tasks)} essays with {MODEL_ID} in {REGION}, {MAX_WORKERS} workers...")

    session = boto3.Session(profile_name=AWS_PROFILE)
    client = session.client(
        "bedrock-runtime", region_name=REGION,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    results = []
    failures = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, client, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                failures.append({"prompt_name": task["prompt_name"], "error": str(e)})
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({len(failures)} failures)")
    elapsed = time.monotonic() - start

    if failures:
        print(f"\n{len(failures)} generation calls failed:")
        for f in failures[:10]:
            print(f"  {f['prompt_name']}: {f['error']}")

    results_df = pd.DataFrame(results)
    results_df.insert(0, "essay_id", [f"bedrock_claude5_{i:04d}" for i in range(len(results_df))])
    results_df["model_id"] = MODEL_ID
    results_df.to_csv(OUTPUT_PATH, index=False)

    total_input_tokens = int(results_df["input_tokens"].sum())
    total_output_tokens = int(results_df["output_tokens"].sum())
    # Bedrock Sonnet 5 promotional pricing through 2026-08-31: $2 / $10 per
    # million input / output tokens (see conversation record for source).
    est_cost = total_input_tokens / 1e6 * 2.0 + total_output_tokens / 1e6 * 10.0

    manifest = {
        "model_id": MODEL_ID, "region": REGION, "target_n": TARGET_N,
        "n_generated": len(results_df), "n_failed": len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(est_cost, 2),
        "prompt_generic_not_production": True,
        "text_dependent_prompts_lack_source_article": True,
        "essays_per_prompt": prompts[["prompt_name", "task", "n_essays"]].to_dict("records"),
    }
    with open(RESULTS_DIR / "bedrock_generation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(results_df)} essays to {OUTPUT_PATH}")
    print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out. "
          f"Estimated cost: ${est_cost:.2f}. Elapsed: {elapsed:.0f}s")
    print(f"Manifest: {RESULTS_DIR / 'bedrock_generation_manifest.json'}")


if __name__ == "__main__":
    main()
