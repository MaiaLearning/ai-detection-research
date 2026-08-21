"""Deterministic "light edit" transformation for Experiment 4's manual
light-edit condition (`RESEARCH_PLAN.md`: "vary sentence lengths, add
contractions... simulating a student who has been told what detectors look
for"). Deliberately crude and mechanical -- this is not a paraphrase tool,
it is what a student following a surface-level tip sheet would actually do
by hand. Deterministic and reproducible: no model, no randomness.
"""
import re

from src.features import split_sentences

_PARAGRAPH_SPLIT_RE = re.compile(r"(\n\s*\n)")

# Reverse mapping: expanded form -> the contraction a student would type
# instead, applied case-insensitively (preserving the original word's
# capitalisation), longest phrase first so multi-word forms match before
# their sub-strings would.
_CONTRACTION_EXPANSIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't", "were not": "weren't",
    "has not": "hasn't", "have not": "haven't", "had not": "hadn't",
    "will not": "won't", "would not": "wouldn't", "should not": "shouldn't",
    "could not": "couldn't", "cannot": "can't", "can not": "can't",
    "it is": "it's", "that is": "that's", "there is": "there's", "here is": "here's",
    "what is": "what's", "who is": "who's",
    "i am": "i'm", "i have": "i've", "i will": "i'll", "i would": "i'd",
    "you are": "you're", "you have": "you've", "you will": "you'll", "you would": "you'd",
    "we are": "we're", "we have": "we've", "we will": "we'll", "we would": "we'd",
    "they are": "they're", "they have": "they've", "they will": "they'll", "they would": "they'd",
    "he is": "he's", "she is": "she's",
    "let us": "let's",
}
_ORDERED_EXPANSIONS = sorted(_CONTRACTION_EXPANSIONS, key=len, reverse=True)
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _ORDERED_EXPANSIONS) + r")\b", re.IGNORECASE,
)


def add_contractions(text: str) -> str:
    """Replace full forms with contractions wherever the fixed mapping
    matches, preserving the matched text's original capitalisation."""
    def repl(m):
        matched = m.group(0)
        contracted = _CONTRACTION_EXPANSIONS[matched.lower()]
        if matched[0].isupper():
            contracted = contracted[0].upper() + contracted[1:]
        return contracted
    return _CONTRACTION_RE.sub(repl, text)


def _vary_sentence_length_paragraph(paragraph: str) -> str:
    sentences = split_sentences(paragraph)
    if len(sentences) < 2:
        return paragraph
    merged = []
    i = 0
    while i < len(sentences):
        if (i + 1) % 3 == 0 and i + 1 < len(sentences):
            first = sentences[i].rstrip(".!?")
            second = sentences[i + 1]
            second_lower = second[0].lower() + second[1:] if second else second
            merged.append(f"{first}, and {second_lower}")
            i += 2
        else:
            merged.append(sentences[i])
            i += 1
    return " ".join(merged)


def vary_sentence_length(text: str) -> str:
    """Deterministically increase sentence-length variance: merge every
    third sentence into the one after it with ", and ", within each
    paragraph (paragraph breaks are preserved, not collapsed). A crude,
    mechanical edit rather than a rewrite -- matching the "vary sentence
    lengths" advice a student following a tip sheet would actually apply."""
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    return "".join(
        part if _PARAGRAPH_SPLIT_RE.match(part) else _vary_sentence_length_paragraph(part)
        for part in parts
    )


def light_edit(text: str) -> str:
    """The manual light-edit condition RESEARCH_PLAN.md pre-registers for
    Experiment 4: vary sentence lengths, then add contractions."""
    return add_contractions(vary_sentence_length(text))
