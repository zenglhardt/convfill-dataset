"""Modular proper-noun grounding check for scaffold-mode responses.

Catches the LLM "leaking" proper nouns (movie titles, place names, etc.)
from far-back turns into a response — content the model shouldn't have
access to per the visibility-window rule.

Allowed sources for any proper noun in `response[turn_k][pos]`:
  - The current paired thought (if non-"<sil>")
  - Earlier thoughts in the current turn (positions < pos, non-"<sil>")
  - The current turn's user utterance
  - The previous turn's thoughts (all non-"<sil>")
  - The previous turn's user utterance

Previous-turn responses are intentionally NOT included: those are
LLM-generated and may themselves carry leaked content; grounding
against them would defeat the check.

Proper noun detection is heuristic: contiguous runs of Title Case
words ("Citizen Kane", "San Francisco", "Wild Nights with Emily")
that are NOT at the start of a sentence (where capitalization is
required regardless). Sentence-initial Title Case words ("Sure",
"Hope", "Great") are excluded. Single-letter "I", possessives, and
all-uppercase acronyms are not matched by the regex.

Self-contained: removing the call site in `validation.py` (or setting
`CHECK_SCAFFOLD_PROPER_NOUNS = False` in config) disables this check
without touching anything else.
"""

import re

# Title Case word/phrase: capital + one-or-more lowercase, optionally
# followed by additional Title Case words separated by single whitespace.
# `[A-Z][a-z]+` requires at least one trailing lowercase letter, so:
#   - single-letter pronouns ("I") don't match
#   - acronyms ("NASA") don't match
_TITLE_CASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Split on sentence-final punctuation followed by whitespace. The first
# Title Case word in each resulting fragment is sentence-initial and
# not counted as a proper noun candidate.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def extract_proper_nouns(text: str) -> list[str]:
    """Return Title Case words/phrases that are NOT sentence-initial.

    Each returned phrase is a contiguous run of Title Case words from
    the source text, in the order they appear. The same phrase can
    appear multiple times if it's repeated; callers should compare
    case-insensitively against the allowed corpus.
    """
    if not text:
        return []
    proper_nouns: list[str] = []
    for sent in _SENTENCE_BOUNDARY.split(text):
        sent = sent.strip()
        if not sent:
            continue
        for m in _TITLE_CASE.finditer(sent):
            text_before = sent[: m.start()]
            if not text_before.strip():
                # Sentence-initial — capital is grammatical, not a signal
                # of proper-noun-ness.
                continue
            proper_nouns.append(m.group())
    return proper_nouns


def _build_allowed_corpus(
    conversation: list[dict],
    turn_idx: int,
    pos: int,
) -> str:
    """Concatenate the case-folded allowed text for a response position.

    See module docstring for the inclusion list. Output is lowercased so
    the substring check in `validate_proper_noun_window` can do a
    case-insensitive containment test.
    """
    parts: list[str] = []

    if turn_idx >= 1:
        prev = conversation[turn_idx - 1]
        parts.append(prev.get("user", ""))
        for t in prev.get("thoughts", []):
            if t != "<sil>":
                parts.append(t)

    cur = conversation[turn_idx]
    parts.append(cur.get("user", ""))

    cur_thoughts = cur.get("thoughts", [])
    for j in range(min(pos, len(cur_thoughts))):
        if cur_thoughts[j] != "<sil>":
            parts.append(cur_thoughts[j])
    if pos < len(cur_thoughts) and cur_thoughts[pos] != "<sil>":
        parts.append(cur_thoughts[pos])

    return " ".join(parts).lower()


def validate_proper_noun_window(
    conversation: list[dict],
    errors: dict,
) -> tuple[bool, str | None]:
    """Verify each response's proper nouns fit the visibility window.

    Returns (True, None) on pass; (False, msg) on first violation.
    Increments `errors["PROPER_NOUN_WINDOW_VIOLATION"]` on a fail.
    """
    for turn_idx, turn in enumerate(conversation):
        responses = turn.get("response", [])
        for pos, resp in enumerate(responses):
            proper_nouns = extract_proper_nouns(resp)
            if not proper_nouns:
                continue
            corpus = _build_allowed_corpus(conversation, turn_idx, pos)
            for pn in proper_nouns:
                if pn.lower() not in corpus:
                    errors["PROPER_NOUN_WINDOW_VIOLATION"] = (
                        errors.get("PROPER_NOUN_WINDOW_VIOLATION", 0) + 1
                    )
                    return False, (
                        f"Turn {turn_idx} pos {pos}: response contains "
                        f"proper noun {pn!r} not present in the visibility "
                        "window (current paired thought, earlier thoughts "
                        "in this turn, current/previous turn user, or "
                        "previous turn thoughts)."
                    )
    return True, None


# ---------------------------------------------------------------------------
# Self-test: `python -m src.validators.proper_noun_validator` exercises the validator on
# hand-crafted positive and negative cases. No external dependencies.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # ---- extract_proper_nouns ----
    # All test sentences are crafted so that proper-noun candidates are
    # MID-sentence (the regex correctly excludes sentence-initial Title
    # Case like "Sure" or "Hope" since those are grammatically required
    # to be capitalized, not signals of proper-noun-ness).
    cases = [
        ("", []),
        ("Hope you enjoy Casablanca.", ["Casablanca"]),
        ("I went to San Francisco.", ["San Francisco"]),
        ("Sure, San Jose works.", ["San Jose"]),
        ("Great choice!", []),  # sentence-initial Title Case only
        ("Sure, sure thing.", []),  # no Title Case mid-sentence
        ("I love Wild Nights with Emily a lot.",
         ["Wild Nights", "Emily"]),  # connector "with" breaks the run
        ("Hi there. I love Citizen Kane.", ["Citizen Kane"]),
        ("It is at 791 Auzerais Avenue.", ["Auzerais Avenue"]),
        ("I'd like Mexican food.", ["Mexican"]),  # mid-sentence Title Case
    ]
    for text, expected in cases:
        got = extract_proper_nouns(text)
        assert got == expected, f"extract({text!r}) -> {got}, expected {expected}"
    print(f"extract_proper_nouns: {len(cases)}/{len(cases)} cases pass")

    # ---- validate_proper_noun_window ----
    # Mock conversation: 3 turns. Each turn has user + thoughts + response.
    convo = [
        {  # turn 0
            "user": "Find me a movie. I want to watch Casablanca.",
            "thoughts": ["<sil>", "Casablanca is queued up for you."],
            "response": [
                "Sure, I can do that.",  # no proper noun
                "Casablanca is queued up.",  # in current paired thought
            ],
        },
        {  # turn 1
            "user": "Great. Play it now.",
            "thoughts": ["<sil>", "Movie is starting."],
            "response": [
                "Awesome, here we go.",  # no proper noun
                "Casablanca is starting now.",  # OK: in turn 0 user (within visibility)
            ],
        },
        {  # turn 2
            "user": "Thank you, goodbye.",
            "thoughts": ["<sil>", "Have a good day."],
            "response": [
                "Hope you enjoy Casablanca.",  # VIOLATION: Casablanca is in turn 0, NOT turn 1
                "Have a great day!",
            ],
        },
    ]
    errors = {}
    ok, msg = validate_proper_noun_window(convo, errors)
    assert not ok, "expected violation in turn 2"
    assert "Casablanca" in msg
    assert errors.get("PROPER_NOUN_WINDOW_VIOLATION") == 1
    print(f"detected violation: {msg}")

    # Same convo without turn 2's leak should pass
    clean_convo = [t for t in convo[:2]]  # drop turn 2
    errors = {}
    ok, msg = validate_proper_noun_window(clean_convo, errors)
    assert ok, f"expected pass, got: {msg}"
    print("clean conversation passes")

    # Within-window proper noun (current paired thought)
    one_turn = [{
        "user": "What's the address?",
        "thoughts": ["It's at 791 Auzerais Avenue."],
        "response": ["The address is 791 Auzerais Avenue."],
    }]
    errors = {}
    ok, msg = validate_proper_noun_window(one_turn, errors)
    assert ok, f"expected pass: {msg}"
    print("within-window proper noun (current paired thought) passes")

    # Proper noun grounded only in previous-turn user (allowed)
    prev_user = [
        {
            "user": "Show me restaurants in Tokyo.",
            "thoughts": ["Several places available."],
            "response": ["Several places available."],
        },
        {
            "user": "Pick the cheapest.",
            "thoughts": ["The cheapest is two stars."],
            # Response references "Tokyo" — in previous turn user, allowed.
            "response": ["The cheapest spot in Tokyo is two stars."],
        },
    ]
    errors = {}
    ok, msg = validate_proper_noun_window(prev_user, errors)
    assert ok, f"expected pass: {msg}"
    print("proper noun grounded in previous-turn user passes")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _selftest()
