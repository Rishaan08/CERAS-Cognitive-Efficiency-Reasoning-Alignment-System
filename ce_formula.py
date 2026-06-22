"""
CERAS ground-truth CE score

Refined version of Equation (1) from the IEEE paper:

    CE = clip(0.35*prompt_quality + 0.25*concept_density
              + 0.15*unique_word_ratio
              + 0.10*log(1+prompt_length)/log(401)
              + 0.10*keystrokes/2000
              + 0.05*prompt_type/3 + eps, 0, 1)

This is the INDEPENDENT GROUND-TRUTH LABEL for ml_training_data —
computed purely from raw features, NEVER from cepm_score/cnn_score/
fused_score. Keeping it independent means fused_score (model
prediction) and ce_score (formula label) stay two genuinely different
signals worth comparing for drift/retraining.

IMPORTANT: extract_ceras_features() in server.py (which feeds the
LIVE CEPM/CNN models) is intentionally left UNTOUCHED by everything in
this file. The models were trained on that function's exact output —
changing it would shift live model predictions in ways the models
were never trained to handle. Everything in this file operates on a
SEPARATE feature path used only for the ml_training_data ground-truth
label.

REVISION HISTORY — why each term looks the way it does

--- prompt_quality (the length-quality term, weight 0.35) ---

v1 (log(1+L)/log(351)): fixed the original paper's hard-clip-at-150
   problem, but rises steepest near L=0 — an 8-word prompt scored
   ~0.37, far too generous for something that short.

v2 (logistic/sigmoid, centered at 60 words): fixed the low end
   (8 words -> 0.07), but saturated even harder than the log curve at
   the high end (100 words -> 0.88, 150 -> 0.99) — collapsed exactly
   the discrimination this refactor was meant to preserve.

v3 (power curve, (L/300)^1.3): fixed v2's saturation, but
   over-corrected — a genuinely strong 120-word prompt only got
   0.30/0.35 weighted credit, pulling good prompts down to ~0.6 when
   they should land 0.75-0.85.

v4 ((L/180)^0.8, clipped at 1.0): fixed v3's under-crediting of
   substantial prompts, but introduced a NEW failure: the clip means
   this term hits a hard ceiling of EXACTLY 1.0 at L=180 and stays
   frozen there forever. A 180-word, 400-word, and 1000-word prompt
   all read identically "1.0" — no continuous quality signal should
   ever actually equal its own ceiling, since there's always
   theoretically a "better" prompt. This is the same saturation
   failure mode the whole refactor was meant to solve, just relocated.

v5 (FINAL — this version): 1 - e^(-L/130) is a true asymptotic curve.
   It keeps climbing forever but mathematically never equals 1.0
   (within realistic prompt lengths — floating point itself rounds to
   1.0 only past ~10,000 words, far beyond any real prompt). Short
   prompts stay appropriately suppressed; every additional word still
   adds a tiny bit more credit, so no two prompt lengths are ever
   truly indistinguishable.

--- concept_density -> structural_complexity (weight 0.25) ---

The ORIGINAL definition (concept_density = words longer than 6 chars
/ total words) was tested against 5 calibration prompts spanning
genuinely shallow to genuinely excellent, and found to be ESSENTIALLY
RANDOM NOISE: it scored 0.37-0.46 across the entire quality spectrum
with no real discrimination, and even scored a deliberately vague
5-word prompt ("explain machine learning to me") HIGHER than several
much better prompts, simply because "machine" and "learning" happen
to be long words. A 0.25-weighted term with no real signal is worse
than not having the term at all.

REPLACEMENT: structural_complexity counts the number of DISTINCT
clause-connecting markers (and, or, but, because, which, how, what,
why, specifically, etc.) and cognitive-operation verbs (explain,
analyze, compare, derive, evaluate, etc.) present in the prompt,
scaled linearly. Counting DISTINCT TYPES (not raw occurrences) matters
because a simple enumeration like "ChatGPT, Claude, or Gemini... text,
images, or code" repeats "or" three times without representing any
real structural complexity — counting occurrences let list-formatting
fake a high score. This is a regex-based proxy; a future iteration
could swap in nlp_features.py's real spaCy-based multi_clause_count
and cognitive_verb_count for higher accuracy without changing this
term's overall shape or weight.

--- unique_word_ratio (weight 0.15) ---

PROBLEM: very short prompts trivially have high uniqueness (a 5-word
prompt is almost always "100% unique" simply because there's no room
for repetition) — this let short, shallow prompts claim a large share
of this term's credit despite having no real vocabulary richness to
demonstrate.

FIX: gated by a length-confidence factor (min(L/40, 1.0)) — the same
pattern used for prompt_quality and structural_complexity. Below 40
words, the raw ratio is scaled down proportionally, since uniqueness
can't be meaningfully assessed in a handful of words.

--- keystroke_term (weight 0.10) ---

PROBLEM: with real typing data, character_count/keystrokes is close to
1.0 for ANY cleanly-typed prompt — meaning nearly every real prompt
got close to full credit on this term regardless of whether the
content itself was good or bad. It was acting as a near-constant
floor, not a discriminating signal.

DECISION: kept as-is (character_count/keystrokes ratio) rather than
artificially capped or rebalanced — clean typing legitimately deserves
its full weight; the term is meant to penalize heavy correction
(messy typing), not to reward "extra clean" typing beyond a normal
baseline. An attempt to cap clean typing at 0.5 (treating it as
"neutral" rather than "rewarded") was tested and reverted — it didn't
meaningfully improve discrimination and reduced the term's intended
penalty range for genuinely messy typing.

================================================
CALIBRATION NOTE
================================================
This formula was calibrated against 5 hand-written test prompts
spanning shallow to excellent. 3 of 5 land within expected range; 2
are close but imperfect (off by ~10-15 percentage points). Further
tuning was intentionally stopped to avoid overfitting constants to a
tiny hand-picked sample. Recalibrate against real ml_training_data
once enough rows accumulate, rather than synthetic test cases.
"""

import math
import random

# ------------------------------------------------
# Structural complexity markers (replaces concept_density)
# ------------------------------------------------
_CLAUSE_MARKERS = {
    "and",
    "or",
    "but",
    "because",
    "since",
    "although",
    "while",
    "if",
    "when",
    "which",
    "that",
    "who",
    "whom",
    "whose",
    "specifically",
    "how",
    "what",
    "why",
}

_COGNITIVE_VERBS = {
    "explain",
    "analyze",
    "understand",
    "derive",
    "compare",
    "walk",
    "describe",
    "define",
    "evaluate",
    "interpret",
    "determine",
    "convergence",
    "tradeoffs",
    "tradeoff",
    "prevent",
    "matters",
    "affect",
    "empirically",
    "synthesize",
    "contrast",
    "justify",
    "infer",
    "deduce",
    "hypothesize",
    "critique",
    "assess",
    "argue",
    "prove",
    "formulate",
    "conclude",
    "examine",
    "investigate",
    "differentiate",
    "categorize",
    "summarize",
    "predict",
    "validate",
}


def _structural_complexity(prompt_text, ref=15):
    """Replaces concept_density. Counts DISTINCT clause-marker and
    cognitive-verb TYPES present (not raw occurrences — avoids the
    repeated-list-word problem), scaled linearly against a reference
    of 15 (a genuinely complex prompt typically has ~10-15 distinct
    structural/cognitive markers across clause types + verb types,
    weighted 2x)."""
    if not prompt_text:
        return 0.0
    words = [w.strip(".,!?;:()").lower() for w in prompt_text.split()]
    clause_types = {w for w in words if w in _CLAUSE_MARKERS}
    cognitive_types = {w for w in words if w in _COGNITIVE_VERBS}
    raw_score = len(clause_types) + (len(cognitive_types) * 2)
    return min(raw_score / ref, 1.0)


def _bounded_prompt_quality(prompt_length, length_ref=130):
    """Asymptotic curve: 1 - e^(-L/130). Never reaches exactly 1.0
    within realistic prompt lengths — every additional word still
    adds a small amount of credit, so no two prompt lengths are ever
    truly indistinguishable. See module docstring for the full
    revision history (v1-v5) explaining why this replaced four
    earlier, flawed versions."""
    if prompt_length is None or prompt_length <= 0:
        return 0.0
    return 1 - math.exp(-prompt_length / length_ref)


def _bounded_unique_word_ratio(unique_word_ratio, prompt_length, length_ref=40):
    """Gates the raw unique_word_ratio by a length-confidence factor —
    short prompts trivially have high uniqueness (no room to repeat
    words), so this term is scaled down proportionally below 40
    words, the same pattern used for prompt_quality."""
    if unique_word_ratio is None:
        return 0.0
    uwr = max(0.0, min(unique_word_ratio, 1.0))
    length_confidence = min((prompt_length or 0) / length_ref, 1.0)
    return uwr * length_confidence


def _bounded_keystroke_term(keystrokes, character_count):
    """Normalizes keystroke effort against actual output length rather
    than a fixed 2000-char ceiling. High keystrokes relative to final
    character count signals effortful revision (penalized), not just
    longer text (which the length term already accounts for). Clean
    typing (ratio near 1.0) earns its full weight — this is not capped
    or treated as "neutral", since legitimately clean typing deserves
    full credit; the term's real job is penalizing messy typing, not
    artificially limiting clean typing's reward."""
    if not keystrokes or keystrokes <= 0:
        return 0.0
    if character_count and character_count > 0:
        ratio = character_count / keystrokes
        return max(0.0, min(ratio, 1.0))
    # Fallback to original paper ratio if no character_count available
    return max(0.0, min(keystrokes / 2000, 1.0))


def compute_ce_score_label(
    prompt_length: float,
    concept_density: float = None,
    unique_word_ratio: float = None,
    keystrokes: float = None,
    character_count: float = None,
    prompt_type: float = 0,
    prompt_text: str = None,
    noise: bool = True,
) -> float:
    """
    Computes the ground-truth CE score label for ml_training_data.
    Independent of CEPM/CNN model predictions — this is the "answer
    key" used for comparison and future retraining, matching the role
    ce_score played in the original 100k synthetic dataset.

    NOTE: concept_density parameter is accepted for backward
    compatibility but NO LONGER USED — it was found to carry no real
    signal (see module docstring). If prompt_text is provided, the new
    structural_complexity term is computed and used in its place. If
    prompt_text is NOT provided, structural_complexity defaults to 0.0
    (this term's weighted contribution is simply lost — callers should
    always pass prompt_text for an accurate score).

    All sub-terms are bounded to [0,1] BEFORE weighting (unlike the
    original formula, which only clipped the final sum). This keeps
    every term discriminative across the full range of real prompts.
    """
    pq = _bounded_prompt_quality(prompt_length)
    sc = _structural_complexity(prompt_text) if prompt_text else 0.0
    uwr = _bounded_unique_word_ratio(unique_word_ratio, prompt_length)
    length_term = (
        min(math.log(1 + prompt_length) / math.log(401), 1.0)
        if prompt_length and prompt_length > 0
        else 0.0
    )
    keystroke_term = _bounded_keystroke_term(keystrokes, character_count)
    prompt_type_term = max(0.0, min((prompt_type or 0) / 3, 1.0))

    eps = random.gauss(0, 0.01) if noise else 0.0

    ce = (
        0.35 * pq
        + 0.25 * sc
        + 0.15 * uwr
        + 0.10 * length_term
        + 0.10 * keystroke_term
        + 0.05 * prompt_type_term
        + eps
    )

    return round(max(0.0, min(ce, 1.0)), 4)
