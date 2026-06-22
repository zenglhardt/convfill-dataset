"""NLI-based semantic validation using DeBERTa.

Verifies that response entries are semantically aligned with their
corresponding thoughts entries for all non-<sil> pairs.
"""

from src.configuration.config import (
    NLI_MODEL_NAME,
    CONTRADICTION_PAIR_MAX,
    ENTAILMENT_PAIR_MIN,
    CONTRADICTION_WHOLE_TURN_MAX,
    ENTAILMENT_WHOLE_TURN_MIN,
)


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_nli_model(device: str | None = None):
    """Load NLI tokenizer and model. Returns (tokenizer, model, device)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = get_device()
    print(f"Loading NLI model ({NLI_MODEL_NAME}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=False)
    model = (
        AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        .to(device)
        .eval()
    )
    return tokenizer, model, device


def compute_nli_probs(
    premises: list[str],
    hypotheses: list[str],
    tokenizer,
    model,
    device: str,
) -> list[dict[str, float]]:
    """Compute NLI probabilities for premise-hypothesis pairs in one batch."""
    import torch

    if not premises:
        return []

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    label2idx = {v: k for k, v in id2label.items()}

    inputs = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu()

    results = []
    for row in probs:
        results.append(
            {
                "entailment": float(row[label2idx["entailment"]]),
                "neutral": float(row[label2idx["neutral"]]),
                "contradiction": float(row[label2idx["contradiction"]]),
            }
        )
    return results


def validate_nli(
    conversation: list[dict],
    tokenizer,
    model,
    device: str,
    errors: dict,
) -> tuple[bool, str | None, list[dict]]:
    """Run NLI validation on a conversation.

    Two-layer rejection:
    1. Per-pair check: reject if any non-<sil> (thought, response) pair has
       contradiction > CONTRADICTION_PAIR_MAX or entailment < ENTAILMENT_PAIR_MIN.
    2. Whole-turn check: per turn, build (user + non-sil thoughts) vs
       (user + all response entries). Reject if contradiction or entailment
       crosses the whole-turn thresholds.

    Returns:
        (ok, msg, per_turn_metadata) where per_turn_metadata is a list aligned
        with conversation, each element {"nli_pair": [...], "nli_whole_turn": {...}}.
    """
    # ------------------------------------------------------------------
    # Build inputs for both layers in one pass, grouped per turn.
    # ------------------------------------------------------------------
    pair_premises: list[str] = []
    pair_hypotheses: list[str] = []
    pair_meta: list[tuple[int, int]] = []  # (turn_idx, entry_idx)

    whole_premises: list[str] = []
    whole_hypotheses: list[str] = []
    whole_turn_index: list[int] = []  # turn_idx for each whole-turn pair

    per_turn_metadata: list[dict] = [
        {"nli_pair": [], "nli_whole_turn": None} for _ in conversation
    ]

    for turn_idx, turn in enumerate(conversation):
        user = turn.get("user", "")
        thoughts = turn.get("thoughts", [])
        response = turn.get("response", [])

        # Per-pair: non-<sil> thought/response pairs
        for i, (t_entry, r_entry) in enumerate(zip(thoughts, response)):
            if t_entry.strip() in ("<sil>", ""):
                continue
            pair_premises.append(t_entry)
            pair_hypotheses.append(r_entry)
            pair_meta.append((turn_idx, i))

        # Whole-turn: needs at least one non-<sil> thought
        non_sil = [t for t in thoughts if t.strip() not in ("<sil>", "")]
        if non_sil:
            whole_premises.append(user + " " + " ".join(non_sil))
            whole_hypotheses.append(user + " " + " ".join(response))
            whole_turn_index.append(turn_idx)

    # ------------------------------------------------------------------
    # Score everything in batches.
    # ------------------------------------------------------------------
    pair_probs = compute_nli_probs(
        pair_premises, pair_hypotheses, tokenizer, model, device
    )
    whole_probs = compute_nli_probs(
        whole_premises, whole_hypotheses, tokenizer, model, device
    )

    # ------------------------------------------------------------------
    # Attach scores to per-turn metadata.
    # ------------------------------------------------------------------
    for (turn_idx, entry_idx), score in zip(pair_meta, pair_probs):
        per_turn_metadata[turn_idx]["nli_pair"].append(
            {
                "index": entry_idx,
                "entailment": score["entailment"],
                "neutral": score["neutral"],
                "contradiction": score["contradiction"],
            }
        )

    for turn_idx, score in zip(whole_turn_index, whole_probs):
        per_turn_metadata[turn_idx]["nli_whole_turn"] = {
            "entailment": score["entailment"],
            "neutral": score["neutral"],
            "contradiction": score["contradiction"],
        }

    # ------------------------------------------------------------------
    # Layer 1: per-pair gates
    # ------------------------------------------------------------------
    for (turn_idx, entry_idx), score in zip(pair_meta, pair_probs):
        if score["contradiction"] > CONTRADICTION_PAIR_MAX:
            errors["NLI_CONTRADICTION"] += 1
            return (
                False,
                (
                    f"Turn {turn_idx} entry {entry_idx}: "
                    f"contradiction={score['contradiction']:.3f} > "
                    f"{CONTRADICTION_PAIR_MAX}"
                ),
                per_turn_metadata,
            )
        if score["entailment"] < ENTAILMENT_PAIR_MIN:
            errors["NLI_PAIR_LOW_ENTAILMENT"] += 1
            return (
                False,
                (
                    f"Turn {turn_idx} entry {entry_idx}: "
                    f"entailment={score['entailment']:.3f} < "
                    f"{ENTAILMENT_PAIR_MIN}"
                ),
                per_turn_metadata,
            )

    # ------------------------------------------------------------------
    # Layer 2: whole-turn gates
    # ------------------------------------------------------------------
    for turn_idx, score in zip(whole_turn_index, whole_probs):
        if score["contradiction"] > CONTRADICTION_WHOLE_TURN_MAX:
            errors["NLI_WHOLE_TURN_CONTRADICTION"] += 1
            return (
                False,
                (
                    f"Turn {turn_idx}: whole-turn contradiction "
                    f"{score['contradiction']:.3f} > "
                    f"{CONTRADICTION_WHOLE_TURN_MAX}"
                ),
                per_turn_metadata,
            )
        if score["entailment"] < ENTAILMENT_WHOLE_TURN_MIN:
            errors["NLI_WHOLE_TURN_LOW_ENTAILMENT"] += 1
            return (
                False,
                (
                    f"Turn {turn_idx}: whole-turn entailment "
                    f"{score['entailment']:.3f} < "
                    f"{ENTAILMENT_WHOLE_TURN_MIN}"
                ),
                per_turn_metadata,
            )

    return True, None, per_turn_metadata
