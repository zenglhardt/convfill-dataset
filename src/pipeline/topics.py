"""Topic loading + chunking helpers.

Topics enter the system either from a topics file (fresh run, freeform
mode), from the SGD dataset (fresh run, scaffold mode), or from a prior
`failed_topics.jsonl` (--retry). All paths produce a uniform list of
entry dicts; freeform entries are `{topic, p1, p2}`, scaffold entries
add `scaffold_turns` (and friends) so downstream worker code can
dispatch on shape or on `config.GENERATION_MODE`.
"""

import json
import math
from pathlib import Path

from src.pipeline.sgd_loader import (
    balanced_sample,
    load_all_dialogues,
    load_dialogue_pool,
    uniform_sample,
)


def parse_topic_line(raw: str, default_p2: str, p1: str) -> dict:
    """Resolve a topics-file line into {topic, p1, p2}."""
    if default_p2:
        return {"topic": raw, "p1": p1, "p2": default_p2}
    parts = raw.split(", ", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Topics line missing ', ' separator (need '<persona>, <topic>'): {raw!r}"
        )
    return {"topic": parts[1], "p1": p1, "p2": parts[0]}


def load_topics_from_file(path: str, default_p2: str, p1: str) -> list:
    with open(path) as f:
        raws = [line for line in f.read().splitlines() if line.strip()]
    return [parse_topic_line(r, default_p2, p1) for r in raws]


_FAILED_TOPIC_METADATA_KEYS = ("error_type", "last_error", "attempts")


def load_topics_from_failed(path: Path) -> list:
    """Read failed_topics.jsonl back into entry dicts.

    Preserves every field from the failed record except the failure
    bookkeeping (`error_type`, `last_error`, `attempts`). For freeform
    mode this round-trips `{topic, p1, p2}`; for scaffold mode it also
    round-trips `scaffold_turns` and any service / dialogue metadata.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entry = {
                k: v for k, v in rec.items()
                if k not in _FAILED_TOPIC_METADATA_KEYS
            }
            out.append(entry)
    return out


def load_scaffolds_from_sgd(
    sgd_root: str,
    splits: list[str],
    services: list[str],
    num_requests: int,
    seed: int,
    cap_max: bool = False,
) -> list[dict]:
    """Build the chunk-entry list for scaffold mode from SGD dialogues.

    Two sampling modes:
      - `services` non-empty: bucket by service, balanced_sample across
        the requested services (`N // S` per service + round-robin
        remainder).
      - `services` empty: uniform shuffle across every dialogue in the
        configured `splits`, take the first `num_requests`. No service
        filtering, no balancing — useful for whole-train harvest.

    If `cap_max` is True, `num_requests` is treated as an upper bound:
    in uniform mode it clamps to the full pool size; in balanced mode
    it clamps to `len(services) * min(per-service pool size)`.

    Each returned entry has the freeform-compatible keys (`topic`, `p1`,
    `p2`) plus `scaffold_turns` and `dialogue_id` / `service` for
    debugging and provenance. `topic` is a synthetic label of the form
    `sgd::<dialogue_id>` so existing log/print plumbing has a sensible
    identifier to print.
    """
    if services:
        pool = load_dialogue_pool(sgd_root, splits, services)
        scaffolds = balanced_sample(
            pool, services, num_requests, seed, cap_max=cap_max,
        )
    else:
        dialogues = load_all_dialogues(sgd_root, splits)
        scaffolds = uniform_sample(
            dialogues, num_requests, seed, cap_max=cap_max,
        )

    out: list[dict] = []
    for sc in scaffolds:
        out.append({
            "topic": f"sgd::{sc['dialogue_id']}",
            "p1": sc["p1"],
            "p2": sc["p2"],
            "service": sc["service"],
            "dialogue_id": sc["dialogue_id"],
            "scaffold_turns": sc["scaffold_turns"],
        })
    return out


def split_list_into_chunks(lst: list, k: int) -> list:
    size = math.ceil(len(lst) / k)
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def create_cycling_topics(topics: list, total: int, cap_max: bool = False) -> list:
    """Pad/cycle topics to reach `total` entries.

    If the topics list is longer than `total`, the prefix is taken;
    if shorter, the list is repeated and clipped.

    If `cap_max` is True, the result is never longer than the input —
    `total` becomes an upper bound and no cycling/repetition occurs.
    """
    if cap_max:
        return topics[:total]
    if len(topics) >= total:
        return topics[:total]
    mult = math.ceil(total / len(topics))
    return (topics * mult)[:total]
