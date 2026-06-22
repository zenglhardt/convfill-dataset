import random
from src.configuration.config import SIL_MIN, SIL_MAX


def generate_sil_schedule(num_turns: int) -> list[int]:
    """Generate a random <sil> count for each turn in the conversation.

    Returns e.g. [2, 0, 3, 1, 4, 0, 2] for a 7-turn conversation,
    where each value is how many filler lines that turn should have.
    """
    return [random.randint(SIL_MIN, SIL_MAX) for _ in range(num_turns)]


def format_sil_schedule_for_prompt(schedule: list[int]) -> str:
    """Format the schedule into a readable block for the prompt template."""
    lines = []
    for i, count in enumerate(schedule):
        if count == 0:
            lines.append(
                f"    Turn {i + 1}: 0 filler lines (jump straight to substance)"
            )
        else:
            s = "s" if count != 1 else ""
            lines.append(
                f"    Turn {i + 1}: {count} filler line{s} before substance"
            )
    return "\n".join(lines)
