"""Unit tests for src.essay_prompts — shared prompt construction that must
produce byte-identical instructions across every AI-generation script, since
that identity is what makes their outputs comparable."""
import pandas as pd
import pytest

from src.essay_prompts import (
    INDEPENDENT_INSTRUCTION,
    TEXT_DEPENDENT_INSTRUCTION,
    build_prompt_list,
    build_task_list,
)


def _write_persuade_csv(tmp_path, prompt_names_and_tasks):
    rows = [
        {"prompt_name": name, "task": task, "assignment": f"Write about {name}."}
        for name, task in prompt_names_and_tasks
    ]
    path = tmp_path / "persuade.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_prompt_list_distributes_target_n_evenly(tmp_path):
    path = _write_persuade_csv(tmp_path, [("A", "Independent"), ("B", "Independent"), ("C", "Text dependent")])
    prompts = build_prompt_list(path, target_n=10)
    assert prompts["n_essays"].sum() == 10
    # 10 // 3 = 3 remainder 1 -> one prompt gets 4, the other two get 3
    assert sorted(prompts["n_essays"].tolist()) == [3, 3, 4]


def test_build_prompt_list_handles_evenly_divisible_target(tmp_path):
    path = _write_persuade_csv(tmp_path, [("A", "Independent"), ("B", "Independent")])
    prompts = build_prompt_list(path, target_n=6)
    assert prompts["n_essays"].tolist() == [3, 3]


def test_build_task_list_uses_independent_template_for_independent_task(tmp_path):
    path = _write_persuade_csv(tmp_path, [("A", "Independent")])
    prompts = build_prompt_list(path, target_n=2)
    tasks = build_task_list(prompts)
    assert len(tasks) == 2
    expected = INDEPENDENT_INSTRUCTION.format(assignment="Write about A.")
    assert all(t["instruction"] == expected for t in tasks)


def test_build_task_list_uses_text_dependent_template_for_text_dependent_task(tmp_path):
    path = _write_persuade_csv(tmp_path, [("B", "Text dependent")])
    prompts = build_prompt_list(path, target_n=1)
    tasks = build_task_list(prompts)
    expected = TEXT_DEPENDENT_INSTRUCTION.format(assignment="Write about B.")
    assert tasks[0]["instruction"] == expected


def test_build_task_list_preserves_prompt_name_and_task_per_essay(tmp_path):
    path = _write_persuade_csv(tmp_path, [("A", "Independent"), ("B", "Text dependent")])
    prompts = build_prompt_list(path, target_n=4)
    tasks = build_task_list(prompts)
    assert {(t["prompt_name"], t["task"]) for t in tasks} == {("A", "Independent"), ("B", "Text dependent")}
