"""Tier 1 essay-texture features: deterministic, pure Python, no model or
network dependency (see RESEARCH_PLAN.md). Suitable for porting into the
Lambda as-is.
"""
import math
import re
import statistics

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

# Sentence splitting must not break on common title/initial abbreviations —
# this is a heuristic, not a full sentence boundary detector.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "etc", "gen", "rev", "capt", "lt", "col", "maj", "sgt",
}
_SENTENCE_BOUNDARY_RE = re.compile(r"([.!?]+)(\s+|$)")

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

PUNCTUATION_SET = set(".,;:!?'\"-()")

# Fixed lexicon of discourse/transition markers, checked as whole-word /
# whole-phrase matches against the lowercased text.
TRANSITION_PHRASES = (
    "on the other hand", "in other words", "as a result", "for this reason",
    "in addition", "in conclusion", "in summary", "in fact", "for example",
    "for instance", "however", "therefore", "moreover", "furthermore",
    "nevertheless", "nonetheless", "additionally", "consequently",
    "similarly", "likewise", "conversely", "meanwhile", "subsequently",
    "accordingly", "otherwise", "specifically", "overall", "besides",
    "thus", "hence", "finally", "first", "second", "third",
)
_TRANSITION_PATTERNS = [
    re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
    for phrase in TRANSITION_PHRASES
]

# Contractions and colloquialisms, as they appear after split_words()
# tokenization (apostrophes preserved).
CONTRACTIONS_AND_COLLOQUIALISMS = {
    "don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "doesn't", "didn't", "shouldn't",
    "wouldn't", "couldn't", "that's", "there's", "here's", "what's",
    "let's", "you're", "you've", "you'll", "you'd", "we're", "we've",
    "we'll", "we'd", "they're", "they've", "they'll", "they'd", "he's",
    "she's", "it's", "it'll", "who's", "y'all", "i'm", "i've", "i'd",
    "i'll", "ain't",
    "gonna", "wanna", "kinda", "sorta", "gotta", "dunno", "yeah", "okay",
    "ok", "stuff", "guys", "lol",
}

FUNCTION_WORDS = {
    "the", "a", "an", "and", "but", "or", "nor", "for", "so", "yet",
    "in", "on", "at", "by", "to", "of", "with", "from", "into", "onto",
    "upon", "over", "under", "about", "against", "between", "through",
    "during", "before", "after", "above", "below",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "not", "no",
}


def split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        end_punct = m.group(1)
        preceding = text[start:m.start()]
        last_word_match = re.search(r"[A-Za-z]+$", preceding)
        last_word = last_word_match.group(0).lower() if last_word_match else ""
        if end_punct == "." and last_word in _ABBREVIATIONS:
            continue
        sentence = text[start:m.end(1)].strip()
        if sentence:
            sentences.append(sentence)
        start = m.end()
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def mean_sentence_length(text: str) -> float:
    lengths = [len(split_words(s)) for s in split_sentences(text)]
    if not lengths:
        return 0.0
    return statistics.mean(lengths)


def sentence_length_std(text: str) -> float:
    lengths = [len(split_words(s)) for s in split_sentences(text)]
    if len(lengths) < 2:
        return 0.0
    return statistics.stdev(lengths)


def paragraph_length_variance(text: str) -> float:
    paragraphs = [p for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]
    lengths = [len(split_words(p)) for p in paragraphs]
    if len(lengths) < 2:
        return 0.0
    return statistics.variance(lengths)


def type_token_ratio(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _mtld_one_direction(tokens: list[str], threshold: float) -> float:
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for token in tokens:
        types.add(token)
        token_count += 1
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            types = set()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        factors += (1 - ttr) / (1 - threshold)
    if factors == 0:
        return float(len(tokens))
    return len(tokens) / factors


def mtld(text: str, threshold: float = 0.72) -> float:
    tokens = split_words(text)
    if not tokens:
        return 0.0
    forward = _mtld_one_direction(tokens, threshold)
    backward = _mtld_one_direction(list(reversed(tokens)), threshold)
    return (forward + backward) / 2


def transition_phrase_rate(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    lowered = text.lower()
    count = sum(len(pattern.findall(lowered)) for pattern in _TRANSITION_PATTERNS)
    return (count / len(words)) * 100


def punctuation_variety(text: str) -> int:
    return len({ch for ch in text if ch in PUNCTUATION_SET})


def contraction_rate(text: str) -> float:
    words = split_words(text)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in CONTRACTIONS_AND_COLLOQUIALISMS)
    return (count / len(words)) * 100


def function_word_rates(text: str) -> dict[str, float]:
    words = split_words(text)
    if not words:
        return {w: 0.0 for w in FUNCTION_WORDS}
    total = len(words)
    counts = {w: 0 for w in FUNCTION_WORDS}
    for w in words:
        if w in counts:
            counts[w] += 1
    return {w: (c / total) * 100 for w, c in counts.items()}


def function_word_entropy(text: str) -> float:
    words = [w for w in split_words(text) if w in FUNCTION_WORDS]
    if not words:
        return 0.0
    total = len(words)
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


# Registry of the scalar Tier 1 features, shared by every experiment script
# so the feature set can't silently drift between experiments.
TIER1_FEATURES = {
    "sentence_length_std": sentence_length_std,
    "mean_sentence_length": mean_sentence_length,
    "type_token_ratio": type_token_ratio,
    "mtld": mtld,
    "transition_phrase_rate": transition_phrase_rate,
    "paragraph_length_variance": paragraph_length_variance,
    "punctuation_variety": punctuation_variety,
    "contraction_rate": contraction_rate,
    "function_word_entropy": function_word_entropy,
}
