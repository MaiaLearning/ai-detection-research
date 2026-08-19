"""Generate an OpenAI (GPT-5.6 Terra) matched set for Experiment 3.

User-directed addition to the plan: run the exact same generic prompt used
for the Bedrock Claude set (src/essay_prompts.py — same instructions, same
per-prompt distribution) against OpenAI's GPT-5.6 Terra, chosen as a proxy
for what a free-tier ChatGPT user would produce. Not confirmed against
OpenAI documentation that Terra is specifically what the free ChatGPT web
app defaults to — flagged as an assumption, not a verified fact.

API key is read from ~/.openai/api_key (never pasted into the conversation
or committed to this repo) rather than an environment variable, so it can't
leak into shell history or process listings.

This is a data-generation script, not an experiment script: it makes paid,
non-deterministic API calls and writes new source data. Re-running it
produces a different (but statistically similar) sample.

Usage: uv run python scripts/generate_openai_essays.py
Output: data/openai_gpt56terra_essays.csv, results/openai_generation_manifest.json
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.essay_prompts import build_prompt_list, build_task_list

MODEL_ID = "gpt-5.6-terra"
API_KEY_PATH = Path.home() / ".openai" / "api_key"
TARGET_N = 1000
MAX_TOKENS = 1600  # generous headroom: reasoning tokens count against this cap,
# and a handful of essays hit 900/900 with zero visible output text before this was raised
REASONING_EFFORT = "low"  # a plain 400-word essay doesn't need "medium" effort's
# reasoning budget; low reduces the odds of reasoning tokens crowding out the answer
MAX_WORKERS = 8
PERSUADE_PATH = Path("data/persuade_2.0_human_scores_demo_id_github.csv")
OUTPUT_PATH = Path("data/openai_gpt56terra_essays.csv")
RESULTS_DIR = Path("results")

# GPT-5.6 pricing as of 2026-08 (see conversation record): $2.50 / $15 per
# million input / output tokens for Terra.
PRICE_PER_M_INPUT = 2.50
PRICE_PER_M_OUTPUT = 15.0


def load_api_key() -> str:
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(
            f"{API_KEY_PATH} not found. Save the OpenAI API key there (chmod 600) before running this script."
        )
    return API_KEY_PATH.read_text().strip()


def generate_one(client: OpenAI, task: dict) -> dict:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": task["instruction"]}],
        max_completion_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )
    text = response.choices[0].message.content
    usage = response.usage
    if not text or not text.strip():
        raise ValueError(
            f"Empty response content (finish_reason={response.choices[0].finish_reason}, "
            f"output_tokens={usage.completion_tokens}) — likely reasoning tokens "
            f"consumed the whole max_completion_tokens budget."
        )
    return {
        **task,
        "full_text": text,
        "word_count": len(text.split()),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api_key = load_api_key()
    prompts = build_prompt_list(PERSUADE_PATH, TARGET_N)
    print("Essays per prompt:\n" + prompts[["prompt_name", "task", "n_essays"]].to_string(index=False))
    tasks = build_task_list(prompts)
    print(f"\nGenerating {len(tasks)} essays with {MODEL_ID}, {MAX_WORKERS} workers...")

    client = OpenAI(api_key=api_key)

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
    results_df.insert(0, "essay_id", [f"openai_gpt56terra_{i:04d}" for i in range(len(results_df))])
    results_df["model_id"] = MODEL_ID
    results_df.to_csv(OUTPUT_PATH, index=False)

    total_input_tokens = int(results_df["input_tokens"].sum())
    total_output_tokens = int(results_df["output_tokens"].sum())
    est_cost = total_input_tokens / 1e6 * PRICE_PER_M_INPUT + total_output_tokens / 1e6 * PRICE_PER_M_OUTPUT

    manifest = {
        "model_id": MODEL_ID, "target_n": TARGET_N,
        "n_generated": len(results_df), "n_failed": len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(est_cost, 2),
        "prompt_generic_not_production": True,
        "text_dependent_prompts_lack_source_article": True,
        "note_free_tier_assumption": (
            "GPT-5.6 Terra chosen by user as a proxy for free-tier ChatGPT output; "
            "not confirmed against OpenAI documentation that Terra is what the free "
            "ChatGPT web app actually defaults to."
        ),
        "essays_per_prompt": prompts[["prompt_name", "task", "n_essays"]].to_dict("records"),
    }
    with open(RESULTS_DIR / "openai_generation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(results_df)} essays to {OUTPUT_PATH}")
    print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out. "
          f"Estimated cost: ${est_cost:.2f}. Elapsed: {elapsed:.0f}s")
    print(f"Manifest: {RESULTS_DIR / 'openai_generation_manifest.json'}")


if __name__ == "__main__":
    main()
