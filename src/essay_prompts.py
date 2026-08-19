"""Shared PERSUADE-prompt construction for AI essay generation scripts.

Used by generate_bedrock_claude_essays.py and generate_openai_essays.py so
both produce byte-identical instructions for the same prompt/task — that
identity is what makes their outputs a fair side-by-side comparison rather
than a comparison partly confounded by prompt wording.

7 of PERSUADE's 15 prompts are "text dependent" and instruct the writer to
cite a specific source article, but the PERSUADE corpus only ships a
citation/title for those sources, not the article text itself. For those,
the instruction below asks the model to write from general knowledge
instead of citing article details it doesn't have.
"""
from pathlib import Path

import pandas as pd

INDEPENDENT_INSTRUCTION = (
    "Write a persuasive or argumentative essay of about 400-450 words responding "
    "to the following writing prompt. Write only the essay text itself, with no "
    "title, preamble, or commentary before or after it.\n\nPrompt: {assignment}"
)

TEXT_DEPENDENT_INSTRUCTION = (
    "Write an essay of about 400-450 words responding to the following writing "
    "prompt. The prompt references a specific source article, but you do not "
    "have access to it — write using your own general knowledge and reasoning "
    "about the topic instead of citing specific article details. Write only the "
    "essay text itself, with no title, preamble, or commentary before or after it."
    "\n\nPrompt: {assignment}"
)


def build_prompt_list(persuade_path: Path, target_n: int) -> pd.DataFrame:
    df = pd.read_csv(persuade_path)
    prompts = df.drop_duplicates("prompt_name")[["prompt_name", "task", "assignment"]].reset_index(drop=True)
    n_prompts = len(prompts)
    base, remainder = divmod(target_n, n_prompts)
    prompts = prompts.sort_values("prompt_name").reset_index(drop=True)
    prompts["n_essays"] = base
    prompts.loc[: remainder - 1, "n_essays"] += 1
    assert prompts["n_essays"].sum() == target_n
    return prompts


def build_task_list(prompts: pd.DataFrame) -> list[dict]:
    tasks = []
    for _, row in prompts.iterrows():
        template = INDEPENDENT_INSTRUCTION if row["task"] == "Independent" else TEXT_DEPENDENT_INSTRUCTION
        instruction = template.format(assignment=row["assignment"])
        for _ in range(int(row["n_essays"])):
            tasks.append({
                "prompt_name": row["prompt_name"],
                "task": row["task"],
                "instruction": instruction,
            })
    return tasks
