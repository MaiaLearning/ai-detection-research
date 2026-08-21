"""Unit tests for src.light_edit -- the deterministic manual light-edit
condition (vary sentence length, add contractions) RESEARCH_PLAN.md
pre-registers for Experiment 4."""
import pytest

from src.light_edit import add_contractions, light_edit, vary_sentence_length


def test_add_contractions_replaces_full_forms():
    assert add_contractions("I do not think that is true.") == "I don't think that's true."


def test_add_contractions_preserves_capitalisation_at_sentence_start():
    assert add_contractions("It is raining. It is fine.") == "It's raining. It's fine."


def test_add_contractions_case_insensitive_match():
    assert add_contractions("Do not worry.") == "Don't worry."


def test_add_contractions_leaves_text_without_matches_unchanged():
    assert add_contractions("The cat sat on the mat.") == "The cat sat on the mat."


def test_add_contractions_longest_phrase_wins_over_substring():
    # "cannot" and "can not" both map to "can't" -- neither should double-fire.
    assert add_contractions("You cannot do that.") == "You can't do that."


def test_vary_sentence_length_merges_every_third_sentence():
    text = "One. Two. Three. Four. Five. Six."
    result = vary_sentence_length(text)
    # sentences 3 and 4 (index 2,3) merge; sentences 6 has no following sentence to merge with
    assert result == "One. Two. Three, and four. Five. Six."


def test_vary_sentence_length_leaves_short_text_unchanged():
    assert vary_sentence_length("Just one sentence.") == "Just one sentence."
    assert vary_sentence_length("") == ""


def test_vary_sentence_length_preserves_paragraph_breaks():
    text = "One. Two. Three. Four.\n\nFive. Six. Seven. Eight."
    result = vary_sentence_length(text)
    assert "\n\n" in result
    before, after = result.split("\n\n")
    assert before == "One. Two. Three, and four."
    assert after == "Five. Six. Seven, and eight."


def test_light_edit_composes_both_transforms():
    text = "I do not like this. It is fine. It is not great."
    result = light_edit(text)
    assert "don't" in result
    assert "it's" in result.lower()


def test_light_edit_increases_sentence_length_variance():
    import statistics as stats

    from src.features import sentence_length_std

    # A uniform-length machine-like paragraph: merging every third sentence
    # should increase the standard deviation of sentence lengths.
    text = " ".join(["This is a sentence with six words."] * 6)
    before = sentence_length_std(text)
    after = sentence_length_std(light_edit(text))
    assert after > before
