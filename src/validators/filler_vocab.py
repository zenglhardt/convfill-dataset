"""Infill diversity checking.

Ensures that conversational infill lines (response entries where
thoughts == "<sil>") are not repetitive across the conversation.
"""

from collections import Counter


def check_infill_diversity(
    conversation: list[dict], max_reuse: int
) -> tuple[bool, str | None]:
    """Check that no infill phrase is used more than *max_reuse* times.

    Infill lines are identified as positions where thoughts[i] == "<sil>".
    The corresponding response[i] is the infill text.
    """
    infill_counter: Counter[str] = Counter()

    for turn in conversation:
        thoughts = turn.get("thoughts", [])
        response = turn.get("response", [])
        for i, t in enumerate(thoughts):
            if t == "<sil>" and i < len(response):
                infill_counter[response[i].strip().lower()] += 1

    violations = {
        phrase: count
        for phrase, count in infill_counter.items()
        if count > max_reuse
    }
    if violations:
        detail = "; ".join(f'"{p}" used {c}x' for p, c in violations.items())
        return False, f"Infill diversity violation: {detail}"
    return True, None
