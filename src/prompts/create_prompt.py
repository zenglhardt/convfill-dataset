"""Prompt assembly with a pre-filled JSON scaffold.

Builds a complete prompt by rendering the template and appending a JSON
scaffold where the LLM only needs to replace placeholder tags with content.

Two flavors:
  - `create_prompt` (freeform mode): the LLM invents the entire conversation
    around random <sil> / substance counts.
  - `create_scaffold_prompt` (scaffold mode): the user turns and thought
    sentences come from an external scaffold (e.g. SGD); the LLM only
    fills <INFILL_N> and <RESPONSE_N> placeholders.
"""

import json
import random

from src.prompts.sil_schedule import format_sil_schedule_for_prompt


def _template_env():
    from jinja2 import Environment, FileSystemLoader

    return Environment(loader=FileSystemLoader(""))


def build_scaffold(
    num_turns: int,
    sil_schedule: list[int],
    substance_counts: list[int],
) -> dict:
    """Build a JSON-serializable scaffold with placeholder tags.

    Each turn has:
      - "<sil>" entries already placed in thoughts
      - <INFILL_N> placeholders for contextual infill response entries
      - <THOUGHT_N> / <RESPONSE_N> placeholders for substance entries
      - <USER> placeholder for the user field
    Placeholder numbers reset per turn.
    """
    turns = []
    for i in range(num_turns):
        sil = sil_schedule[i]
        sub = substance_counts[i]

        thoughts = ["<sil>"] * sil + [
            f"<THOUGHT_{j}>" for j in range(1, sub + 1)
        ]
        response = [f"<INFILL_{j}>" for j in range(1, sil + 1)] + [
            f"<RESPONSE_{j}>" for j in range(1, sub + 1)
        ]

        turns.append(
            {
                "user": "<USER>",
                "thoughts": thoughts,
                "response": response,
            }
        )

    return {"conversation": turns}


def create_prompt(
    topic: str,
    p1: str,
    p2: str,
    num_turns: int,
    sil_schedule: list[int],
    substance_min: int,
    substance_max: int,
    example_path: str,
) -> tuple[str, str, list[int]]:
    """Assemble the user prompt and scaffold JSON.

    Returns (user_prompt, scaffold_json, substance_counts).
    The scaffold is provided in a follow-up message so the model
    fills it in rather than generating from scratch.
    """
    # Generate random substance counts per turn
    substance_counts = [
        random.randint(substance_min, substance_max) for _ in range(num_turns)
    ]

    # Build scaffold
    scaffold = build_scaffold(num_turns, sil_schedule, substance_counts)
    scaffold_json = json.dumps(scaffold, indent=2)

    # Load example
    with open(example_path) as f:
        example = f.read()

    # Render template
    from src.configuration.config import PROMPT_TEMPLATE_PATH
    env = _template_env()
    template = env.get_template(PROMPT_TEMPLATE_PATH)

    user_prompt = template.render(
        topic=topic,
        P1=p1,
        P2=p2,
        num_turns=num_turns,
        sil_schedule_text=format_sil_schedule_for_prompt(sil_schedule),
        substance_min=substance_min,
        substance_max=substance_max,
        example=example,
    )

    return user_prompt, scaffold_json, substance_counts


def build_scaffold_from_sgd(sgd_scaffold_turns: list[dict]) -> dict:
    """Build the JSON scaffold passed to the LLM in scaffold mode.

    `sgd_scaffold_turns` is a list of dicts with keys:
      - "user": str, P1's verbatim utterance
      - "thoughts": list[str] -- "<sil>" entries plus the SGD assistant's
                    sentences as substance lines
      - "num_sils": int -- count of leading "<sil>" entries
      - "num_substance": int -- count of substance lines

    The output JSON has user + thoughts pre-filled with real content and
    response slots as <INFILL_N> / <RESPONSE_N> placeholders. Numbering
    resets per turn.
    """
    turns = []
    for t in sgd_scaffold_turns:
        num_sils = t["num_sils"]
        num_sub = t["num_substance"]
        response = (
            [f"<INFILL_{j}>" for j in range(1, num_sils + 1)]
            + [f"<RESPONSE_{j}>" for j in range(1, num_sub + 1)]
        )
        turns.append({
            "user": t["user"],
            "thoughts": list(t["thoughts"]),
            "response": response,
        })
    return {"conversation": turns}


def create_scaffold_prompt(
    p1: str,
    p2: str,
    service: str,
    sgd_scaffold_turns: list[dict],
    example_path: str,
) -> tuple[str, str]:
    """Assemble the prompt + scaffold JSON for scaffold-mode generation.

    `sgd_scaffold_turns` come from `sgd_loader.apply_sil_schedule` (one per
    USER->SYSTEM exchange in the source SGD dialogue).

    Returns (user_prompt, scaffold_json). The scaffold is provided in a
    follow-up message — same shape as freeform mode — so the model fills
    it in rather than generating from scratch.
    """
    from src.configuration.config import PROMPT_TEMPLATE_PATH

    scaffold = build_scaffold_from_sgd(sgd_scaffold_turns)
    scaffold_json = json.dumps(scaffold, indent=2)

    with open(example_path) as f:
        example = f.read()

    env = _template_env()
    template = env.get_template(PROMPT_TEMPLATE_PATH)

    service_pretty = service.replace("_", " ").lower()

    user_prompt = template.render(
        p1=p1,
        p2=p2,
        service=service,
        service_pretty=service_pretty,
        example=example,
    )

    return user_prompt, scaffold_json
