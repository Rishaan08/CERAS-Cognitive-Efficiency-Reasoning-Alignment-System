"""
CERAS NLP Feature Extraction

Extracts the semantic/linguistic features that feed ml_training_data
and (eventually) the CNN model's lexical/semantic inputs.

Uses:
  - spaCy (en_core_web_sm) — POS tags, named entities, dependency parsing,
    multi-clause detection, cognitive verb detection
  - textstat — readability scores (Flesch reading ease)

Install:
    pip install spacy textstat
    python -m spacy download en_core_web_sm

This module is imported lazily (only when first needed) so server.py
startup time isn't affected if these packages are missing — it logs a
warning and falls back to None for NLP-only fields instead of crashing.
"""

import re
import logging

logger = logging.getLogger("ceras-server")

# Lazy-loaded globals
_nlp = None
_textstat = None
_load_error = None


def _ensure_loaded():
    """Load spaCy model + textstat once, on first use."""
    global _nlp, _textstat, _load_error
    if _nlp is not None or _load_error is not None:
        return

    try:
        import spacy
        import textstat as ts

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded yet
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            _load_error = "spacy_model_missing"
            return
        _textstat = ts
        logger.info("✅ NLP feature extractor (spaCy + textstat) loaded.")
    except ImportError as e:
        logger.warning(f"NLP libraries not installed (spacy/textstat): {e}")
        _load_error = "nlp_libs_missing"


# Cognitive verbs — used for cognitive_verb_count
_COGNITIVE_VERBS = {
    "analyze",
    "evaluate",
    "synthesize",
    "compare",
    "contrast",
    "explain",
    "justify",
    "infer",
    "deduce",
    "hypothesize",
    "interpret",
    "critique",
    "assess",
    "argue",
    "prove",
    "derive",
    "formulate",
    "conclude",
    "examine",
    "investigate",
    "differentiate",
    "categorize",
    "summarize",
    "predict",
    "validate",
    "construct",
    "design",
    "optimize",
    "reason",
    "understand",
    "consider",
    "determine",
    "identify",
    "clarify",
}

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "and",
    "in",
    "on",
    "at",
    "for",
    "with",
    "by",
    "from",
    "as",
    "that",
    "this",
    "it",
    "its",
    "i",
    "you",
    "he",
    "she",
    "they",
    "we",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "but",
    "or",
    "if",
    "so",
    "not",
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
}


def extract_nlp_features(prompt_text: str) -> dict:
    """
    Extract NLP/semantic features for ml_training_data.
    Returns a dict with keys matching ml_training_data columns.
    Falls back to None for any field if spaCy/textstat aren't available
    (server keeps running — these are non-critical enrichment fields).
    """
    _ensure_loaded()

    result = {
        "multi_clause_count": None,
        "cognitive_verb_count": None,
        "lexical_diversity": None,
        "readability_score": None,
        "stopword_ratio": None,
        "punctuation_density": None,
        "named_entity_count": None,
        "keyword_density": None,
        "topic_consistency_score": None,
        "coherence_score": None,
    }

    if not prompt_text or not prompt_text.strip():
        return result

    words = prompt_text.split()
    word_count = max(len(words), 1)

    # Punctuation density (regex — always available)
    punct_count = len(re.findall(r"[.,;:!?\-()\[\]{}\"']", prompt_text))
    result["punctuation_density"] = round(punct_count / max(len(prompt_text), 1), 4)

    # Lexical diversity (type-token ratio — always available)
    lowered_words = [w.lower().strip(".,!?;:\"'()") for w in words]
    unique = set(lowered_words)
    result["lexical_diversity"] = round(len(unique) / word_count, 4)

    # Stopword ratio (always available, simple set lookup)
    stop_hits = sum(1 for w in lowered_words if w in _STOPWORDS)
    result["stopword_ratio"] = round(stop_hits / word_count, 4)

    # Cognitive verb count (regex fallback if no spaCy)
    cog_hits = sum(1 for w in lowered_words if w in _COGNITIVE_VERBS)
    result["cognitive_verb_count"] = cog_hits

    # Readability score (textstat)
    if _textstat is not None:
        try:
            result["readability_score"] = round(
                _textstat.flesch_reading_ease(prompt_text), 2
            )
        except Exception as e:
            logger.debug(f"textstat failed: {e}")

    if _nlp is not None:
        try:
            doc = _nlp(prompt_text)

            # Named entity count
            result["named_entity_count"] = len(doc.ents)

            # Multi-clause count — count subordinate/coordinate clause markers
            # via dependency parse (marks "ccomp", "advcl", "relcl", "conj")
            clause_deps = {"ccomp", "advcl", "relcl", "xcomp", "acl"}
            result["multi_clause_count"] = sum(
                1 for token in doc if token.dep_ in clause_deps
            )

            # Refine cognitive verb count using actual lemmas + POS=VERB
            cog_lemma_hits = sum(
                1
                for token in doc
                if token.pos_ == "VERB" and token.lemma_.lower() in _COGNITIVE_VERBS
            )
            result["cognitive_verb_count"] = max(cog_hits, cog_lemma_hits)

            # Keyword density — proportion of content words (NOUN/PROPN/VERB/ADJ)
            content_pos = {"NOUN", "PROPN", "VERB", "ADJ"}
            content_tokens = [t for t in doc if t.pos_ in content_pos and t.is_alpha]
            result["keyword_density"] = round(len(content_tokens) / max(len(doc), 1), 4)

            # Topic consistency — average pairwise similarity between sentence
            # vectors (requires word vectors; en_core_web_sm has limited vectors,
            # so this is a coarse proxy based on shared lemma overlap across sentences)
            sentences = list(doc.sents)
            if len(sentences) > 1:
                lemma_sets = [
                    {t.lemma_.lower() for t in sent if t.is_alpha and not t.is_stop}
                    for sent in sentences
                ]
                overlaps = []
                for i in range(len(lemma_sets) - 1):
                    a, b = lemma_sets[i], lemma_sets[i + 1]
                    if a and b:
                        overlap = len(a & b) / len(a | b)
                        overlaps.append(overlap)
                result["topic_consistency_score"] = (
                    round(sum(overlaps) / len(overlaps), 4) if overlaps else None
                )
                result["coherence_score"] = result["topic_consistency_score"]
            else:
                # Single sentence — trivially coherent/consistent
                result["topic_consistency_score"] = 1.0
                result["coherence_score"] = 1.0

        except Exception as e:
            logger.warning(f"spaCy feature extraction failed (non-fatal): {e}")

    return result
