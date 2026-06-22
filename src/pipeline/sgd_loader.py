"""Load DSTC8 Schema-Guided Dialogue (SGD) dialogues and turn them into
conversation scaffolds for the scaffold-mode generation path.

The SGD dataset (https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
is a directory of `train/`, `dev/`, `test/` splits, each containing
`dialogues_*.json` files. Each JSON is an array of dialogues; each dialogue
has `dialogue_id`, `services`, and `turns` (alternating USER/SYSTEM).

A "scaffold" here is the dialogue rebuilt as a list of (user_utterance,
thought_sentences) pairs — one per USER->SYSTEM exchange. The SYSTEM
utterance is split into sentences; each sentence becomes one substance
line in the eventual `thoughts` array. Filler `<sil>` slots are added
later (see `apply_sil_schedule`).

Stage-1 deliverable: this module is standalone (no imports from the rest
of the pipeline) so it can be exercised via the CLI at the bottom before
any wiring happens.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path


# Sentence splitter: split on sentence-final punctuation followed by
# whitespace. Good enough for SGD's well-formed English. We deliberately
# avoid an NLP dep (nltk/spacy) since the input is short, single-paragraph
# system utterances.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(utterance: str) -> list[str]:
    """Split a SYSTEM utterance into sentences. Trims and drops empties."""
    parts = _SENTENCE_SPLIT.split(utterance.strip())
    return [p.strip() for p in parts if p.strip()]


def load_dialogue_pool(
    sgd_root: str | Path,
    splits: list[str],
    services: list[str],
) -> dict[str, list[dict]]:
    """Walk the SGD splits and return dialogues bucketed by primary service.

    A dialogue's "primary" service is the first entry in its `services`
    list. Multi-service dialogues are bucketed under their first service
    only (single-domain dialogues — the common case — have exactly one
    service anyway).

    Returns `{service: [dialogue_dict, ...]}`. Services with no dialogues
    end up as empty lists; callers should validate availability.
    """
    sgd_root = Path(sgd_root)
    pool: dict[str, list[dict]] = {s: [] for s in services}
    wanted = set(services)

    for split in splits:
        split_dir = sgd_root / split
        if not split_dir.is_dir():
            print(
                f"sgd_loader: warning - split dir not found: {split_dir}",
                file=sys.stderr,
            )
            continue
        for path in sorted(split_dir.glob("dialogues_*.json")):
            with open(path) as f:
                dialogues = json.load(f)
            for d in dialogues:
                svcs = d.get("services") or []
                if not svcs:
                    continue
                primary = svcs[0]
                if primary in wanted:
                    pool[primary].append(d)
    return pool


def load_all_dialogues(
    sgd_root: str | Path,
    splits: list[str],
) -> list[dict]:
    """Walk the SGD splits and return every dialogue as a flat list.

    Loading is deterministic: dialogues are concatenated in
    sorted-filename order per split. The randomness in scaffold-mode
    "whole-pool" sampling lives in `uniform_sample`, which takes this
    deterministic list and applies a seeded shuffle. Splitting it this
    way means a given seed reproduces the same draw across runs.
    """
    sgd_root = Path(sgd_root)
    out: list[dict] = []
    for split in splits:
        split_dir = sgd_root / split
        if not split_dir.is_dir():
            print(
                f"sgd_loader: warning - split dir not found: {split_dir}",
                file=sys.stderr,
            )
            continue
        for path in sorted(split_dir.glob("dialogues_*.json")):
            with open(path) as f:
                out.extend(json.load(f))
    return out


def dialogue_to_scaffold(dialogue: dict) -> dict | None:
    """Convert one SGD dialogue into a scaffold dict.

    Walks `turns` in order, pairing each USER turn with the immediately-
    following SYSTEM turn. Skips trailing USER turns (orphans). Each
    SYSTEM utterance is sentence-split.

    Returns:
        {
          "dialogue_id": str,
          "service": str,
          "p1": str,
          "p2": str,
          "scaffold_turns": [
            {"user": str, "thought_sentences": [str, ...]},
            ...
          ],
        }
        or None if the dialogue yields zero valid pairs.
    """
    turns = dialogue.get("turns") or []
    if not turns:
        return None

    services = dialogue.get("services") or []
    primary = services[0] if services else "unknown"

    scaffold_turns = []
    i = 0
    while i + 1 < len(turns):
        u = turns[i]
        s = turns[i + 1]
        if u.get("speaker") == "USER" and s.get("speaker") == "SYSTEM":
            raw_assistant = (s.get("utterance") or "").strip()
            sentences = split_into_sentences(raw_assistant)
            user_utt = (u.get("utterance") or "").strip()
            if user_utt and sentences:
                scaffold_turns.append({
                    "user": user_utt,
                    "thought_sentences": sentences,
                    # Verbatim original SGD assistant utterance, for the
                    # scaffold-thought BERT-score floor in Stage 4.
                    "original_thought": raw_assistant,
                })
            i += 2
        else:
            # Malformed pairing (e.g. two USER turns in a row). Advance
            # one and try to resync. Real SGD data is well-formed, so
            # this is just defensive.
            i += 1

    if not scaffold_turns:
        return None

    # Human-readable persona derived from the service name. e.g.
    # "Restaurants_1" -> "a restaurants 1 assistant"
    pretty_service = primary.replace("_", " ").lower()
    return {
        "dialogue_id": dialogue.get("dialogue_id"),
        "service": primary,
        "p1": "the user",
        "p2": f"a {pretty_service} assistant",
        "scaffold_turns": scaffold_turns,
    }


def _allocate_counts(services: list[str], num_requests: int) -> dict[str, int]:
    """Allocate num_requests across services as evenly as possible.

    With S = len(services), each service gets `num_requests // S`, and
    the remainder is distributed round-robin in services-list order so
    earlier-listed services pick up the extras.
    """
    s = len(services)
    base, rem = divmod(num_requests, s)
    counts = {svc: base for svc in services}
    for i in range(rem):
        counts[services[i]] += 1
    return counts


def balanced_sample(
    pool_by_service: dict[str, list[dict]],
    services: list[str],
    num_requests: int,
    seed: int,
    cap_max: bool = False,
) -> list[dict]:
    """Sample `num_requests` scaffolds, balanced across `services`.

    Each service's pool is shuffled with `seed`, then sliced. Services
    are taken in the order given in `services`, so order matters when
    `num_requests` doesn't divide evenly.

    If `cap_max` is True and `num_requests` would require more dialogues
    than available (under balanced allocation), `num_requests` is silently
    clamped to `len(services) * min(per-service pool size)` — the largest
    request that still preserves balance without duplicating any dialogue.

    Raises ValueError if any requested service lacks enough dialogues
    (cap_max=False) or if some dialogues fail conversion (rare; only on
    degenerate data).
    """
    if not services:
        raise ValueError("balanced_sample: services list is empty")
    if num_requests <= 0:
        raise ValueError(
            f"balanced_sample: num_requests must be > 0, got {num_requests}"
        )

    if cap_max:
        min_pool = min(len(pool_by_service.get(svc, [])) for svc in services)
        max_balanced = min_pool * len(services)
        if num_requests > max_balanced:
            print(
                f"sgd_loader: cap_max - clamping num_requests "
                f"{num_requests} -> {max_balanced} "
                f"(min pool {min_pool} x {len(services)} services)",
                file=sys.stderr,
            )
            num_requests = max_balanced
        if num_requests <= 0:
            raise ValueError(
                "balanced_sample: cap_max produced num_requests=0; one of "
                f"the requested services has an empty pool: "
                f"{[s for s in services if not pool_by_service.get(s)]}"
            )

    rng = random.Random(seed)
    counts = _allocate_counts(services, num_requests)

    for svc in services:
        avail = len(pool_by_service.get(svc, []))
        need = counts[svc]
        if avail < need:
            raise ValueError(
                f"sgd_loader: service {svc!r} has {avail} dialogues but "
                f"{need} are required (num_requests={num_requests}, "
                f"services={services})"
            )

    selected_dialogues: list[dict] = []
    for svc in services:
        shuffled = list(pool_by_service[svc])
        rng.shuffle(shuffled)
        selected_dialogues.extend(shuffled[: counts[svc]])

    scaffolds: list[dict] = []
    for d in selected_dialogues:
        sc = dialogue_to_scaffold(d)
        if sc is not None:
            scaffolds.append(sc)

    if len(scaffolds) < num_requests:
        # Degenerate case: a sampled dialogue had zero valid USER->SYSTEM
        # pairs and was dropped. Surface clearly so the caller knows the
        # output is short of the request.
        print(
            f"sgd_loader: warning - {num_requests - len(scaffolds)} of "
            f"{num_requests} sampled dialogues yielded no scaffold turns "
            "and were dropped",
            file=sys.stderr,
        )
    return scaffolds


def uniform_sample(
    dialogues: list[dict],
    num_requests: int,
    seed: int,
    cap_max: bool = False,
) -> list[dict]:
    """Sample `num_requests` scaffolds uniformly from a flat dialogue pool.

    Shuffles `dialogues` with `random.Random(seed)`, slices the prefix,
    converts each via `dialogue_to_scaffold`, and drops any None results.
    Used when no per-service balancing is desired.

    If `cap_max` is True, `num_requests` is silently clamped to
    `len(dialogues)` rather than raising.

    Raises ValueError if the pool is smaller than `num_requests` and
    `cap_max` is False.
    """
    if num_requests <= 0:
        raise ValueError(
            f"uniform_sample: num_requests must be > 0, got {num_requests}"
        )
    if cap_max and num_requests > len(dialogues):
        print(
            f"sgd_loader: cap_max - clamping num_requests "
            f"{num_requests} -> {len(dialogues)} (full pool size)",
            file=sys.stderr,
        )
        num_requests = len(dialogues)
    if len(dialogues) < num_requests:
        raise ValueError(
            f"uniform_sample: pool has {len(dialogues)} dialogues but "
            f"{num_requests} are required"
        )

    rng = random.Random(seed)
    shuffled = list(dialogues)
    rng.shuffle(shuffled)

    scaffolds: list[dict] = []
    for d in shuffled[:num_requests]:
        sc = dialogue_to_scaffold(d)
        if sc is not None:
            scaffolds.append(sc)

    if len(scaffolds) < num_requests:
        print(
            f"sgd_loader: warning - {num_requests - len(scaffolds)} of "
            f"{num_requests} sampled dialogues yielded no scaffold turns "
            "and were dropped",
            file=sys.stderr,
        )
    return scaffolds


def apply_sil_schedule(
    scaffold_turn: dict,
    sil_min: int,
    sil_max: int,
    rng: random.Random,
) -> dict:
    """Decorate one scaffold turn with a programmatic <sil> count.

    The structural validator requires <sil> entries to be contiguous at
    the start of the thoughts array (validation.py:107-115), so we
    prepend the sils. Returns a dict suitable for direct use as the
    `thoughts` portion of the scaffold passed to the LLM.

    Output:
        {
          "user": str,
          "thoughts": ["<sil>", ..., sentence_1, sentence_2, ...],
          "num_sils": int,
          "num_substance": int,
        }
    """
    sentences = scaffold_turn["thought_sentences"]
    num_sils = rng.randint(sil_min, sil_max)
    thoughts = ["<sil>"] * num_sils + list(sentences)
    return {
        "user": scaffold_turn["user"],
        "thoughts": thoughts,
        "num_sils": num_sils,
        "num_substance": len(sentences),
        # Pass-through for Stage-4 scaffold-thought BERT validator.
        "original_thought": scaffold_turn.get("original_thought", ""),
    }


# ---------------------------------------------------------------------------
# CLI: ad-hoc inspection for Stage 1
# ---------------------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Inspect SGD scaffolds (Stage-1 sanity check).",
    )
    p.add_argument(
        "--sgd-root", default="dstc8-schema-guided-dialogue",
        help="Path to the SGD dataset root (the dir holding train/dev/test).",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train"],
        help="Which split dirs to draw from. Default: train.",
    )
    p.add_argument(
        "--services", nargs="+", required=True,
        help="One or more service names (e.g. Restaurants_1 Hotels_2).",
    )
    p.add_argument(
        "--num", type=int, default=2,
        help="Total number of scaffolds to sample (split across services).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the shuffle and sil application.",
    )
    p.add_argument(
        "--apply-sils", action="store_true",
        help="Also show what the thoughts array looks like after a "
             "random <sil> schedule is applied (uses --sil-min / --sil-max).",
    )
    p.add_argument("--sil-min", type=int, default=0)
    p.add_argument("--sil-max", type=int, default=3)
    p.add_argument(
        "--json", action="store_true",
        help="Dump full scaffolds as JSON instead of a human-readable view.",
    )
    args = p.parse_args()

    pool = load_dialogue_pool(args.sgd_root, args.splits, args.services)
    print("Pool sizes:")
    for svc in args.services:
        print(f"  {svc}: {len(pool.get(svc, []))} dialogues")
    print()

    scaffolds = balanced_sample(pool, args.services, args.num, args.seed)
    print(f"Sampled {len(scaffolds)} scaffolds.\n")

    if args.json:
        print(json.dumps(scaffolds, indent=2))
        return

    rng = random.Random(args.seed) if args.apply_sils else None
    for sc in scaffolds:
        print(
            f"=== dialogue_id={sc['dialogue_id']} service={sc['service']} "
            f"({len(sc['scaffold_turns'])} turns) ==="
        )
        for i, t in enumerate(sc["scaffold_turns"]):
            print(f"  turn {i + 1} user: {t['user']}")
            if args.apply_sils:
                applied = apply_sil_schedule(t, args.sil_min, args.sil_max, rng)
                for line in applied["thoughts"]:
                    print(f"    thoughts[]: {line}")
                print(
                    f"    (num_sils={applied['num_sils']}, "
                    f"num_substance={applied['num_substance']})"
                )
            else:
                for s in t["thought_sentences"]:
                    print(f"    sentence: {s}")
        print()


if __name__ == "__main__":
    _cli()
