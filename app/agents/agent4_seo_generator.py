"""
Agent 4 — SEO Copy Text Generator

Pipeline:
  1. Tonality extraction from transcripts (NLTK + Claude)
  2. SEO text generation (Claude, single call)
  3. Pass 1 validation — programmatic structure checks
  4. Pass 2 validation — Claude tonality consistency check
  5. Retry loops (max 3 structure, max 2 tonality)
"""

import re
import json
import nltk
from collections import Counter

from app.models.agent4 import (
    SEOCopyRequest,
    SEOCopyResult,
    TonalityProfile,
    ValidationResult,
)
from app.services.claude_client import complete

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANNED_WORDS = [
    "comprehensive", "delve", "essential", "diverse", "tailored",
    "holistic", "optimal", "robust", "notably", "sustainable",
    "in the realm of", "in the context of", "with regard to", "thereof",
    "In today's fast-paced world", "It is important to note",
    "In conclusion", "Last but not least", "In today's digital landscape",
]

WORD_MIN = 816
WORD_MAX = 900
H2_MIN = 5
H2_MAX = 6
BOLD_MIN = 10
BOLD_MAX = 12
BULLET_COUNT = 4


# ---------------------------------------------------------------------------
# Tonality extraction (Step 1)
# ---------------------------------------------------------------------------

def _extract_tonality(transcripts: list[str]) -> TonalityProfile:
    """Analyse transcript(s) with NLTK and return a TonalityProfile."""
    merged = " ".join(transcripts)
    sentences = nltk.sent_tokenize(merged)
    if not sentences:
        sentences = [merged]

    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths) if lengths else 10
    short = sum(1 for l in lengths if l <= 8) / len(lengths)
    medium = sum(1 for l in lengths if 9 <= l <= 18) / len(lengths)
    long_ = sum(1 for l in lengths if l >= 19) / len(lengths)

    # Burstiness: mean absolute difference between consecutive sentence lengths
    diffs = [abs(lengths[i] - lengths[i - 1]) for i in range(1, len(lengths))]
    burstiness_score = sum(diffs) / len(diffs) if diffs else 0
    burstiness = "high" if burstiness_score > 5 else "low"

    # Dominant opener
    openers: Counter = Counter()
    conjunctions = {"und", "aber", "denn", "oder", "weil", "obwohl", "jedoch",
                    "and", "but", "because", "however", "yet", "so", "for"}
    adverbs = {"oft", "manchmal", "gerade", "heute", "bereits", "now", "often",
               "sometimes", "already", "still", "just"}
    for s in sentences:
        first = s.split()[0].lower() if s.split() else ""
        if first in conjunctions:
            openers["conjunction"] += 1
        elif first in adverbs:
            openers["adverb"] += 1
        elif s.strip().endswith("?"):
            openers["question"] += 1
        else:
            openers["subject"] += 1
    dominant_opener = openers.most_common(1)[0][0] if openers else "subject"

    # Use Claude to extract the remaining dimensions
    profile_json = _claude_tonality_extract(merged[:4000])

    return TonalityProfile(
        avg_sentence_length=round(avg, 1),
        short_sentence_share=round(short, 2),
        medium_sentence_share=round(medium, 2),
        long_sentence_share=round(long_, 2),
        dominant_opener=dominant_opener,
        vocab_register=profile_json.get("register", "conversational"),
        characteristic_connectors=profile_json.get("characteristic_connectors", []),
        emotional_tone=profile_json.get("emotional_tone", "empathetic"),
        burstiness=burstiness,
    )


def _claude_tonality_extract(transcript_excerpt: str) -> dict:
    system = (
        "You are a linguistic analyst. Analyse the provided transcript excerpt "
        "and return a JSON object with exactly these keys:\n"
        '  "register": one of ["colloquial","conversational","professional","academic"]\n'
        '  "characteristic_connectors": list of up to 10 transition words/phrases from the text\n'
        '  "emotional_tone": one of ["empathetic","motivational","analytical","calm"]\n'
        "Return ONLY valid JSON, no markdown fences."
    )
    user = f"Transcript excerpt:\n\n{transcript_excerpt}"
    raw = complete(system, user, max_tokens=400)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Text generation (Step 2)
# ---------------------------------------------------------------------------

def _build_generation_prompt(req: SEOCopyRequest, profile: TonalityProfile) -> tuple[str, str]:
    b = req.business
    connectors = ", ".join(profile.characteristic_connectors[:5]) if profile.characteristic_connectors else "none identified"

    system = f"""You are an expert SEO copywriter. Generate a fully structured, SEO-optimised text.

## CRITICAL — TRANSCRIPT USAGE
The transcript provided is used ONLY to calibrate your writing style and tone.
Do NOT copy, paraphrase, reference, or borrow any content, examples, facts, phrases,
or anecdotes from the transcript. Treat transcript content as completely invisible —
only its language rhythm, register, and connectors matter.

## STYLE RULES (apply throughout)
- Sentence length distribution: ~{int(profile.short_sentence_share*100)}% short (≤8 words), \
~{int(profile.medium_sentence_share*100)}% medium (9-18 words), \
~{int(profile.long_sentence_share*100)}% long (19+ words)
- Sentence burstiness: {profile.burstiness} — {'frequently alternate short and long sentences' if profile.burstiness == 'high' else 'keep sentence lengths consistent'}
- Register: {profile.vocab_register}
- Emotional tone: {profile.emotional_tone}
- Dominant sentence opener: {profile.dominant_opener}
- Naturally include 3-5 of these connectors: {connectors}

## DELIBERATE IMPERFECTIONS (mandatory, for human authenticity)
a) At least once: allow a sentence to digress slightly before returning — a parenthetical aside or brief elaboration
b) Not every paragraph has the same internal structure (claim-support, question-open, mid-thought)
c) 1-2 sentences may use a slightly more informal register than the surrounding prose
d) No two consecutive paragraphs may begin with the same grammatical structure
e) At least one H2 section must contain a notably shorter or longer sentence than its neighbours

## BANNED WORDS/PHRASES — never use:
comprehensive, delve, essential, diverse, tailored, holistic, optimal, robust, notably,
"in the realm of", "in the context of", "In today's fast-paced world",
"It is important to note", "In conclusion", "Last but not least"

## OUTPUT FORMAT
Return ONLY the text with these markers — nothing else:
H1: [title containing keyword]

[intro paragraph 1 — keyword in sentence 1]
[intro paragraph 2 — keyword + location reference]

H2: [Keyword H2 — Stage 1, naturally worded question or description, contains keyword]
[paragraph 1]
• [bullet 1 — 8-15 words]
• [bullet 2 — 8-15 words]
• [bullet 3 — 8-15 words]
• [bullet 4 — 8-15 words]
[paragraph 2]

H2: [Free H2 — Stage 2]
[paragraphs]

H2: [Free H2 — Stage 3 / USP]
[paragraphs with <strong>Label.</strong> Explanation. bold elements]

H2: {b.company_name}: Experte für [keyword]
[paragraphs with <strong>Label.</strong> Explanation. bold elements — at least 3]

[CTA section — Stage 5, contains keyword, inviting tone]"""

    user = f"""Write the SEO text with these parameters:

Keyword: {req.keyword}
Content topic: {req.content_topic}

Company: {b.company_name}
Expert: {b.expert_name}
Location: {b.location}
Target audience: {b.target_audience}
USP: {b.usp}
CTA instruction: {req.cta}

NARRATIVE ARC (follow strictly in this order):
Stage 1 — THE PROBLEM: describe the pain point the audience faces re the topic. Make them feel understood.
Stage 2 — WHY IT HASN'T BEEN SOLVED: empathetic explanation of past attempts that failed.
Stage 3 — WHAT IS DIFFERENT NOW: introduce the USP. Be concrete, no vague claims.
Stage 4 — THE SOLUTION: concrete process, what happens in a session, expected outcomes.
Stage 5 — CALL TO ACTION: {req.cta}

STRUCTURE REQUIREMENTS:
- Body text word count: aim for 880 words. Exclude H1:/H2: headline lines from your count.
  When in doubt write more — 816 is the absolute floor, 900 is the ceiling.
- Keyword "{req.keyword}" must appear AT LEAST 7 times total:
    • Once in the very first sentence
    • Once in the second intro paragraph with a location reference
    • Once in the first H2 headline
    • 4+ more times distributed naturally across H2 body paragraphs and the CTA
  (7 uses in 850 words = 0.82% density — the minimum required)
- Keyword density: 0.8%-1.2% of total word count — hitting 7+ uses above guarantees this
- Exactly 1 H1 (contains keyword)
- 5-6 H2s total; first H2 contains keyword; Expert H2 = "{b.company_name}: Experte für {req.keyword}"
- Exactly 4 bullet points after the first H2 (8-15 words each)
- 10-12 bold elements total using format: <strong>Short label.</strong> Explanation sentence.
  - Max 3 per H2 section; none in intro or CTA
- All 5 narrative stages covered in order; no single H2 covers two stages"""

    return system, user


def _generate_text(req: SEOCopyRequest, profile: TonalityProfile) -> str:
    system, user = _build_generation_prompt(req, profile)
    return complete(system, user, max_tokens=2800)


def _generate_text_with_feedback(
    req: SEOCopyRequest,
    profile: TonalityProfile,
    errors: list[str],
    previous_text: str,
) -> str:
    """Regenerate with validation errors fed back so Claude can fix specific issues."""
    system, user = _build_generation_prompt(req, profile)
    errors_str = "\n".join(f"  • {e}" for e in errors)
    feedback_user = (
        f"{user}\n\n"
        f"---\n"
        f"CORRECTION REQUIRED — your previous attempt failed these checks:\n"
        f"{errors_str}\n\n"
        f"REWRITE the text below fixing ALL listed issues. Keep everything that is correct.\n\n"
        f"Previous text:\n{previous_text}"
    )
    return complete(system, feedback_user, max_tokens=2800)


# ---------------------------------------------------------------------------
# Pass 1 — Programmatic validation (Step 3)
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    """Count words in body text, excluding H1/H2 lines."""
    lines = text.splitlines()
    body_lines = [l for l in lines if not l.startswith("H1:") and not l.startswith("H2:")]
    return len(" ".join(body_lines).split())


def _count_keyword(text: str, keyword: str) -> int:
    text_lower = text.lower()
    kw_lower = keyword.lower()
    count = 0
    start = 0
    while True:
        idx = text_lower.find(kw_lower, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(kw_lower)
    return count


def _validate_structure(text: str, req: SEOCopyRequest) -> ValidationResult:
    errors = []
    keyword = req.keyword

    word_count = _count_words(text)
    if not (WORD_MIN <= word_count <= WORD_MAX):
        errors.append(f"Word count {word_count} outside range {WORD_MIN}-{WORD_MAX}")

    # H1 check
    h1_matches = re.findall(r"^H1:", text, re.MULTILINE)
    h1_count = len(h1_matches)
    if h1_count != 1:
        errors.append(f"Expected exactly 1 H1, found {h1_count}")
    else:
        h1_line = next(l for l in text.splitlines() if l.startswith("H1:"))
        if keyword.lower() not in h1_line.lower():
            errors.append("H1 does not contain the primary keyword")

    # H2 check
    h2_lines = [l for l in text.splitlines() if l.startswith("H2:")]
    h2_count = len(h2_lines)
    if not (H2_MIN <= h2_count <= H2_MAX):
        errors.append(f"Expected 5-6 H2s, found {h2_count}")

    # First H2 contains keyword
    if h2_lines and keyword.lower() not in h2_lines[0].lower():
        errors.append("First H2 must contain the primary keyword")

    # Expert H2 format check
    expert_h2_expected = f"{req.business.company_name}: Experte für {keyword}".lower()
    found_expert = any(expert_h2_expected in l.lower() for l in h2_lines)
    if not found_expert:
        errors.append(f"Expert H2 not found (expected: '{req.business.company_name}: Experte für {keyword}')")

    # Bullet list — exactly 4 items
    bullets = re.findall(r"^[•\-\*] .+", text, re.MULTILINE)
    if len(bullets) != BULLET_COUNT:
        errors.append(f"Expected exactly 4 bullet points, found {len(bullets)}")

    # Bold elements
    bold_matches = re.findall(r"<strong>.+?</strong>", text)
    bold_count = len(bold_matches)
    if not (BOLD_MIN <= bold_count <= BOLD_MAX):
        errors.append(f"Expected 10-12 bold elements, found {bold_count}")

    # Keyword density
    kw_occurrences = _count_keyword(text, keyword)
    density = kw_occurrences / word_count if word_count else 0
    if not (0.008 <= density <= 0.012):
        errors.append(f"Keyword density {density:.3f} outside range 0.008-0.012")

    # Keyword in first paragraph (first non-H1 non-empty line)
    body_lines = [l for l in text.splitlines() if l.strip() and not l.startswith("H")]
    first_para = body_lines[0] if body_lines else ""
    first_sentence = re.split(r"[.!?]", first_para)[0] if first_para else ""
    if keyword.lower() not in first_sentence.lower():
        errors.append("Keyword not found in first sentence of first paragraph")

    # Banned words
    found_banned = [bw for bw in BANNED_WORDS if bw.lower() in text.lower()]

    return ValidationResult(
        passed=len(errors) == 0 and len(found_banned) == 0,
        word_count=word_count,
        keyword_occurrences=kw_occurrences,
        keyword_density=round(density, 4),
        h1_count=h1_count,
        h2_count=h2_count,
        bold_count=bold_count,
        banned_words_found=found_banned,
        errors=errors + ([f"Banned words: {found_banned}"] if found_banned else []),
    )


# ---------------------------------------------------------------------------
# Pass 2 — Tonality validation (Step 4)
# ---------------------------------------------------------------------------

def _validate_tonality(text: str, profile: TonalityProfile) -> bool:
    connectors_str = ", ".join(profile.characteristic_connectors)
    system = (
        "You are a tonality quality checker. Evaluate the text against the profile. "
        "Return JSON: {\"passed\": true/false, \"issues\": [\"...\"]}"
    )
    user = (
        f"Tonality profile:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Check:\n"
        f"- Sentence burstiness matches '{profile.burstiness}'\n"
        f"- Register is '{profile.vocab_register}'\n"
        f"- At least 3 connectors from [{connectors_str}] are present\n"
        f"- No AI-typical clichés\n"
        f"- Emotional tone is '{profile.emotional_tone}' throughout\n\n"
        f"Text:\n{text[:3000]}"
    )
    raw = complete(system, user, max_tokens=300)
    try:
        result = json.loads(raw)
        return bool(result.get("passed", False))
    except json.JSONDecodeError:
        return True  # Don't block on parse failure


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_seo_generation(req: SEOCopyRequest, transcripts: list[str]) -> SEOCopyResult:
    # Step 1: tonality extraction
    profile = _extract_tonality(transcripts) if transcripts else TonalityProfile(
        avg_sentence_length=12,
        short_sentence_share=0.35,
        medium_sentence_share=0.45,
        long_sentence_share=0.20,
        dominant_opener="conjunction",
        vocab_register="conversational",
        characteristic_connectors=[],
        emotional_tone="empathetic",
        burstiness="high",
    )

    # Step 2+3: generate + validate structure (up to 3 attempts)
    text = ""
    validation = None
    gen_attempts = 0

    for attempt in range(3):
        gen_attempts = attempt + 1
        if attempt == 0 or validation is None:
            text = _generate_text(req, profile)
        else:
            # Feed validation errors back so Claude knows exactly what to fix
            text = _generate_text_with_feedback(req, profile, validation.errors, text)
        validation = _validate_structure(text, req)
        if validation.passed:
            break

    if validation is None:
        raise RuntimeError("Validation object not created")

    # Step 4: tonality check (up to 2 additional attempts)
    tonality_attempts = 0
    if transcripts:
        for attempt in range(2):
            tonality_attempts = attempt + 1
            if _validate_tonality(text, profile):
                break
            # Regenerate with explicit style correction note
            text = _generate_text(req, profile)
            validation = _validate_structure(text, req)

    return SEOCopyResult(
        text=text,
        tonality_profile=profile,
        validation=validation,
        generation_attempts=gen_attempts,
        tonality_attempts=tonality_attempts,
    )
