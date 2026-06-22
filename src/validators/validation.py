"""Structural validation for generated conversations."""

import re

from src.configuration.config import (
    TURNS_MIN,
    TURNS_MAX,
    MIN_SUBSTANCE_LENGTH,
    MAX_FILLER_REUSE,
    SUBSTANCE_MIN,
    SUBSTANCE_MAX,
)
from src.validators.filler_vocab import check_infill_diversity

# Characters allowed in spoken conversation text.
# Letters, digits, whitespace, common punctuation, smart quotes.
_ALLOWED = re.compile(
    r"^[a-zA-Z0-9\s.,!?;:'\-/()\"\u2019\u2018\u201c\u201d]+$"
)


def create_error_tracker() -> dict:
    return {
        "MISSING_FIELD": 0,
        "MISMATCHED_LENGTH": 0,
        "EMPTY_TURN": 0,
        "SIL_COUNT_MISMATCH": 0,
        "SIL_NOT_CONTIGUOUS": 0,
        "TURN_COUNT_OUT_OF_RANGE": 0,
        "EMPTY_P1_TURN": 0,
        "TRIVIAL_SUBSTANCE": 0,
        "INVALID_RESPONSE_ENTRY": 0,
        "SPECIAL_CHARACTERS": 0,
        "FILLER_DIVERSITY": 0,
        "SUBSTANCE_COUNT_OUT_OF_RANGE": 0,
        "MISSING_CONVERSATION_KEY": 0,
        "SCAFFOLD_USER_MISMATCH": 0,
        "SCAFFOLD_THOUGHT_MISMATCH": 0,
        "SCAFFOLD_THOUGHT_BERT_TOO_LOW": 0,
        "PROPER_NOUN_WINDOW_VIOLATION": 0,
        "NLI_CONTRADICTION": 0,
        "NLI_PAIR_LOW_ENTAILMENT": 0,
        "NLI_WHOLE_TURN_CONTRADICTION": 0,
        "NLI_WHOLE_TURN_LOW_ENTAILMENT": 0,
        "BERT_SCORE_TOO_LOW": 0,
        "BERT_ALIGNMENT_FAIL": 0,
        "ONE_SHOT_ATTEMPTED": 0,
        "ONE_SHOT_SUCCEEDED": 0,
        "ONE_SHOT_FAILED": 0,
        "API_EXHAUSTED": 0,
    }


def _normalize_to_list(value) -> list[str]:
    """Convert a string (newline-separated) or list to a cleaned list."""
    if isinstance(value, str):
        value = value.split("\n")
    return [line.strip() for line in value if line.strip()]


def validate_turn(
    turn: dict,
    expected_sil_count: int,
    errors: dict,
    scaffold_mode: bool = False,
) -> tuple[bool, str | None]:
    """Validate a single turn against the specification.

    `scaffold_mode=True` relaxes the spoken-charset whitelist for `user`
    and `thoughts` — both are pinned verbatim from SGD source content,
    which can legitimately contain symbols like "$29.77" that violate the
    freeform spell-out rule. The check still runs on `response` (the LLM
    generates that fresh).
    """

    # 1. Required fields
    for field in ("user", "thoughts", "response"):
        if field not in turn:
            errors["MISSING_FIELD"] += 1
            return False, f"Missing '{field}' field"

    user = turn["user"]
    thoughts = _normalize_to_list(turn["thoughts"])
    response = _normalize_to_list(turn["response"])
    # Persist the normalized arrays so downstream validators (NLI, BERT
    # alignment) and the dataset-write path see the same clean data the
    # structural checks below operate on. Without this, an LLM-emitted
    # blank/whitespace entry would be stripped here but survive in the raw
    # arrays — causing an IndexError later when one array is iterated past
    # the length of the other.
    turn["thoughts"] = thoughts
    turn["response"] = response

    # 2. user must be a non-empty string
    if not isinstance(user, str) or not user.strip():
        errors["EMPTY_P1_TURN"] += 1
        return False, "user is empty or not a string"

    # 3. Arrays must not be empty
    if not response or not thoughts:
        errors["EMPTY_TURN"] += 1
        return False, "Empty thoughts or response"

    # 4. Array lengths must match
    if len(response) != len(thoughts):
        errors["MISMATCHED_LENGTH"] += 1
        return False, (
            f"Mismatched lengths: response={len(response)}, "
            f"thoughts={len(thoughts)}"
        )

    # 5. <sil> count must match the per-turn specification
    actual_sil = sum(1 for t in thoughts if t == "<sil>")
    if actual_sil != expected_sil_count:
        errors["SIL_COUNT_MISMATCH"] += 1
        return False, (
            f"Expected {expected_sil_count} <sil> entries, got {actual_sil}"
        )

    # 6. <sil> entries must be contiguous at the start
    if expected_sil_count > 0:
        for i in range(expected_sil_count):
            if i >= len(thoughts) or thoughts[i] != "<sil>":
                errors["SIL_NOT_CONTIGUOUS"] += 1
                return False, f"Expected <sil> at position {i}, got '{thoughts[i] if i < len(thoughts) else 'N/A'}'"
        for i in range(expected_sil_count, len(thoughts)):
            if thoughts[i] == "<sil>":
                errors["SIL_NOT_CONTIGUOUS"] += 1
                return False, f"Stray <sil> at position {i} after filler block"
    else:
        if any(t == "<sil>" for t in thoughts):
            errors["SIL_COUNT_MISMATCH"] += 1
            return False, "Expected 0 <sil> but found one"

    # 7. Substance line count must be within global range
    substance_count = len(thoughts) - expected_sil_count
    if substance_count < SUBSTANCE_MIN or substance_count > SUBSTANCE_MAX:
        errors["SUBSTANCE_COUNT_OUT_OF_RANGE"] += 1
        return False, (
            f"Substance count {substance_count} not in "
            f"[{SUBSTANCE_MIN}, {SUBSTANCE_MAX}]"
        )

    # 8. All response entries must be non-empty strings
    for i, entry in enumerate(response):
        if not isinstance(entry, str) or not entry.strip():
            errors["INVALID_RESPONSE_ENTRY"] += 1
            return False, f"response[{i}] is empty or not a string"

    # 9. Non-<sil> thoughts entries must have substance.
    # Scaffold mode skips the MIN_SUBSTANCE_LENGTH floor: SGD legitimately
    # contains short utterances like "Yes." that we still want to keep
    # verbatim. We still require non-empty strings.
    for i, entry in enumerate(thoughts):
        if entry != "<sil>":
            if not isinstance(entry, str) or not entry.strip():
                errors["TRIVIAL_SUBSTANCE"] += 1
                return False, f"thoughts[{i}] is empty or not a string"
            if (
                not scaffold_mode
                and len(entry.strip()) < MIN_SUBSTANCE_LENGTH
            ):
                errors["TRIVIAL_SUBSTANCE"] += 1
                return False, f"thoughts[{i}] too short: '{entry}'"

    # 10. No special characters / non-English alphabet.
    # Scaffold mode: user and thoughts are pinned verbatim from SGD; SGD
    # content can have "$29.77"-style symbols that the freeform charset
    # rule was tuned against. The rule still applies to response.
    field_entries = [("response", response)]
    if not scaffold_mode:
        field_entries.append(("user", [user]))
        field_entries.append(("thoughts", [e for e in thoughts if e != "<sil>"]))
    for field_name, entries in field_entries:
        for entry in entries:
            if not _ALLOWED.match(entry):
                errors["SPECIAL_CHARACTERS"] += 1
                return False, (
                    f"{field_name} contains disallowed characters: "
                    f"'{entry[:60]}'"
                )

    return True, None


def validate_against_scaffold(
    conv: dict,
    expected_users: list[str],
    expected_thoughts: list[list[str]],
    errors: dict,
) -> tuple[bool, str | None]:
    """Verify the model copied user + thought-substance lines verbatim.

    `expected_thoughts[i]` is the substance-only sentence list for turn i
    (no "<sil>" entries). The generated thoughts array may include "<sil>"
    entries; only the non-"<sil>" entries are compared, in order.

    Assumes `validate_turn` has already run, so each turn has a string
    `user` and a list `thoughts`.
    """
    turns = conv["conversation"]
    if len(turns) != len(expected_users):
        # Defensive: turn count is checked earlier, but bail cleanly if
        # this validator is somehow called with mismatched lengths.
        return False, (
            f"scaffold turn count {len(expected_users)} != "
            f"generated turn count {len(turns)}"
        )

    for i, turn in enumerate(turns):
        if turn["user"] != expected_users[i]:
            errors["SCAFFOLD_USER_MISMATCH"] += 1
            return False, (
                f"Turn {i}: user does not match scaffold. "
                f"Expected {expected_users[i]!r}, got {turn['user']!r}"
            )

        got_substance = [t for t in turn["thoughts"] if t != "<sil>"]
        if got_substance != expected_thoughts[i]:
            errors["SCAFFOLD_THOUGHT_MISMATCH"] += 1
            return False, (
                f"Turn {i}: thoughts substance lines do not match scaffold. "
                f"Expected {expected_thoughts[i]}, got {got_substance}"
            )

    return True, None


def validate_conversation(
    conv: dict,
    sil_schedule: list[int],
    errors: dict,
    scaffold_expectation: dict | None = None,
) -> tuple[bool, str | None]:
    """Validate an entire conversation structurally.

    Returns (True, None) on success or (False, error_message) on failure.
    NLI validation is run separately after this passes.

    `scaffold_expectation`, when provided, must be a dict with keys
    "users" (list[str]) and "thoughts" (list[list[str]] of substance-only
    sentence lists). Used by scaffold-mode generation to confirm the LLM
    did not modify the pinned user / thought content.
    """
    if not conv:
        return False, "Empty conversation object"

    # Handle turn_1, turn_2, ... alternate format
    if "conversation" not in conv:
        turn_keys = [k for k in conv.keys() if k.startswith("turn_")]
        if turn_keys:
            turns = []
            for i in range(1, len(turn_keys) + 1):
                key = f"turn_{i}"
                if key in conv:
                    turns.append(conv[key])
                else:
                    return False, f"Missing {key}"
            conv["conversation"] = turns
        else:
            errors["MISSING_CONVERSATION_KEY"] += 1
            return False, "Missing 'conversation' field and no turn_* fields"

    turns = conv["conversation"]

    # Must be a list
    if not isinstance(turns, list):
        return False, "conversation is not a list"

    # Turn count range check. Skipped in scaffold mode where the SGD
    # dialogue's natural length defines the conversation; TURNS_MIN/MAX
    # are tuned for freeform generation.
    if scaffold_expectation is None:
        if len(turns) < TURNS_MIN or len(turns) > TURNS_MAX:
            errors["TURN_COUNT_OUT_OF_RANGE"] += 1
            return False, (
                f"Turn count {len(turns)} not in [{TURNS_MIN}, {TURNS_MAX}]"
            )

    # Turn count must match schedule
    if len(turns) != len(sil_schedule):
        errors["SIL_COUNT_MISMATCH"] += 1
        return False, (
            f"Turn count {len(turns)} doesn't match "
            f"sil_schedule length {len(sil_schedule)}"
        )

    # Per-turn structural validation
    scaffold_mode = scaffold_expectation is not None
    for i, turn in enumerate(turns):
        ok, msg = validate_turn(
            turn, sil_schedule[i], errors, scaffold_mode=scaffold_mode
        )
        if not ok:
            return False, f"Turn {i}: {msg}"

    # Scaffold-mode check: confirm user + thought-substance lines are
    # verbatim from the source scaffold. Skipped in freeform mode.
    if scaffold_expectation is not None:
        ok, msg = validate_against_scaffold(
            conv,
            scaffold_expectation["users"],
            scaffold_expectation["thoughts"],
            errors,
        )
        if not ok:
            return False, msg

        # Proper-noun visibility-window check. Modular and toggleable —
        # see proper_noun_validator.py. Set CHECK_SCAFFOLD_PROPER_NOUNS to
        # False in config to disable without code changes; alternatively,
        # remove this entire `if` block to fully decouple.
        from src.configuration.config import CHECK_SCAFFOLD_PROPER_NOUNS
        if CHECK_SCAFFOLD_PROPER_NOUNS:
            from src.validators.proper_noun_validator import validate_proper_noun_window
            ok, msg = validate_proper_noun_window(turns, errors)
            if not ok:
                return False, msg

    # Infill diversity check across entire conversation
    ok, msg = check_infill_diversity(turns, MAX_FILLER_REUSE)
    if not ok:
        errors["FILLER_DIVERSITY"] += 1
        return False, msg

    return True, None
