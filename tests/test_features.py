"""Unit tests for src.features — Tier 1 deterministic essay-texture features.

Every expected value below is computed by hand from the input string, not
copied from the implementation, so a bug in the implementation should show
up as a test failure rather than being baked into the expectation.
"""
import math

import pytest

from src.features import (
    contraction_rate,
    function_word_entropy,
    function_word_rates,
    mean_sentence_length,
    mtld,
    paragraph_length_variance,
    punctuation_variety,
    sentence_length_std,
    split_sentences,
    split_words,
    transition_phrase_rate,
    type_token_ratio,
)


# ---- tokenization primitives ----

def test_split_words_lowercases_and_keeps_contractions():
    assert split_words("Don't Stop. It's fine!") == ["don't", "stop", "it's", "fine"]


def test_split_words_ignores_bare_numbers_and_punctuation():
    assert split_words("I have 3 cats, and 2 dogs.") == [
        "i", "have", "cats", "and", "dogs",
    ]


def test_split_sentences_basic():
    text = "This is one. This is two! Is this three?"
    assert split_sentences(text) == [
        "This is one.",
        "This is two!",
        "Is this three?",
    ]


def test_split_sentences_does_not_split_on_common_abbreviation():
    text = "Dr. Smith went home. He was tired."
    assert split_sentences(text) == ["Dr. Smith went home.", "He was tired."]


# ---- sentence-length features ----

def test_mean_sentence_length_simple():
    # "This is one" -> 3 words, "This is two too" -> 4 words
    text = "This is one. This is two too."
    assert mean_sentence_length(text) == pytest.approx(3.5)


def test_sentence_length_std_zero_for_uniform_sentences():
    # every sentence has exactly 3 words -> stdev is 0
    text = "One two three. Four five six. Seven eight nine."
    assert sentence_length_std(text) == pytest.approx(0.0)


def test_sentence_length_std_single_sentence_is_zero():
    assert sentence_length_std("Just one sentence here.") == pytest.approx(0.0)


def test_sentence_length_std_matches_manual_stdev():
    # word counts per sentence: 2, 4  -> sample stdev (ddof=1) of [2, 4] = sqrt(2)
    text = "One two. One two three four."
    assert sentence_length_std(text) == pytest.approx(math.sqrt(2))


# ---- lexical diversity ----

def test_type_token_ratio_all_unique():
    assert type_token_ratio("the cat sat on mat") == pytest.approx(1.0)


def test_type_token_ratio_with_repeats():
    # tokens: the, cat, the, dog -> 3 unique / 4 total
    assert type_token_ratio("the cat the dog") == pytest.approx(3 / 4)


def test_type_token_ratio_empty_text_is_zero():
    assert type_token_ratio("...") == 0.0


def test_mtld_higher_for_more_diverse_text():
    repetitive = "the dog the dog the dog the dog the dog the dog the dog " * 3
    diverse = (
        "wolves roam distant forests seeking prey beneath silver moonlight "
        "while ancient rivers carve valleys through granite mountains slowly "
    ) * 3
    assert mtld(diverse) > mtld(repetitive)


def test_mtld_returns_full_length_when_never_crosses_threshold():
    # short, fully-unique text never drops TTR below the 0.72 default
    # threshold, so MTLD falls back to the token count (McCarthy & Jarvis's
    # partial-factor handling for a text with zero completed factors).
    text = "wolves roam distant forests seeking prey beneath silver moonlight"
    assert mtld(text) == pytest.approx(len(split_words(text)))


# ---- transition phrases ----

def test_transition_phrase_rate_counts_multiword_phrases():
    text = "On the other hand, some disagree. For example, cats rock."
    # words: on/the/other/hand/some/disagree/for/example/cats/rock = 10 words
    # 2 transition phrases ("on the other hand", "for example") -> rate = 20.0
    assert transition_phrase_rate(text) == pytest.approx(20.0)


def test_transition_phrase_rate_zero_when_absent():
    assert transition_phrase_rate("Cats are great pets for busy people") == 0.0


# ---- paragraph structure ----

def test_paragraph_length_variance_zero_for_single_paragraph():
    assert paragraph_length_variance("Just one paragraph of text here.") == 0.0


def test_paragraph_length_variance_matches_manual_variance():
    # paragraph word counts: 2, 4 -> sample variance (ddof=1) of [2, 4] = 2.0
    text = "One two.\n\nOne two three four."
    assert paragraph_length_variance(text) == pytest.approx(2.0)


# ---- punctuation ----

def test_punctuation_variety_counts_distinct_marks():
    # uses period, comma, exclamation -> 3 distinct marks from the fixed set
    assert punctuation_variety("Wait, no! Really.") == 3


def test_punctuation_variety_ignores_repeats():
    assert punctuation_variety("Really... Really... Really...") == 1


# ---- contractions / colloquialisms ----

def test_contraction_rate_counts_known_forms():
    # "don't" and "gonna" are both in the fixed lexicon; 5 words total
    text = "Cats eat fish don't gonna"
    assert contraction_rate(text) == pytest.approx((2 / 5) * 100)


def test_contraction_rate_zero_when_absent():
    assert contraction_rate("The committee will not attend") == 0.0


# ---- function words ----

def test_function_word_entropy_zero_when_no_function_words_present():
    assert function_word_entropy("Wolves roam forests seeking prey") == 0.0


def test_function_word_entropy_positive_when_varied_function_words_present():
    text = "The cat sat on the mat and the dog ran to the door"
    assert function_word_entropy(text) > 0.0


def test_function_word_rates_reports_rate_per_hundred_words():
    text = "the cat the dog the bird"  # 6 words, "the" appears 3 times
    rates = function_word_rates(text)
    assert rates["the"] == pytest.approx(50.0)
