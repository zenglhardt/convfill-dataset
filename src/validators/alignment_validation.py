"""BERT-score alignment validation.

Detects misalignment between thoughts and responses within a turn. For
each non-<sil> position i:
  - Anchor:  BERT_F1(thoughts[i], response[i])
  - Cross:   BERT_F1(thoughts[i], response[j])  for every j != i
             (both <sil> infill positions and other non-<sil> positions)
  - Require: anchor > max(cross)  AND  anchor >= THOUGHT_RESPONSE_BERT_MIN

If any other response position is more semantically similar to a thought
than that thought's own response is, the arrays are likely misaligned
(content meant for a different position landed at i, or vice versa).
"""

from __future__ import annotations

from src.configuration.config import (
    BERT_SCORE_MODEL,
    BERT_SCORE_NUM_LAYERS,
    SCAFFOLD_THOUGHT_BERT_MIN,
    THOUGHT_RESPONSE_BERT_MIN,
)


def load_bert_scorer():
    """Load the BERT scorer once per process.

    Some models (e.g., microsoft/deberta-xlarge-mnli) ship with
    `model_max_length` set to a sentinel int that overflows the fast
    tokenizer's i32 on `enable_truncation`. We clamp to the model's
    actual position-embedding limit (typically 512) post-load.
    """
    print(
        f"Loading BERT scorer ({BERT_SCORE_MODEL}, "
        f"layers={BERT_SCORE_NUM_LAYERS})..."
    )
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=BERT_SCORE_MODEL,
        num_layers=BERT_SCORE_NUM_LAYERS,
        lang="en",
        rescale_with_baseline=False,
    )
    tok = scorer._tokenizer
    cap = getattr(scorer._model.config, "max_position_embeddings", 512) or 512
    if tok.model_max_length is None or tok.model_max_length > cap:
        tok.model_max_length = cap
    return scorer


def _split_positions(thoughts: list[str]) -> tuple[list[int], list[int]]:
    """Return (sil_positions, nonsil_positions) given a thoughts list."""
    sil = [i for i, t in enumerate(thoughts) if t == "<sil>"]
    nonsil = [i for i, t in enumerate(thoughts) if t != "<sil>"]
    return sil, nonsil


def validate_alignment(
    conversation: list[dict],
    scorer: BERTScorer,
    errors: dict,
) -> tuple[bool, str | None, list[dict]]:
    """Run alignment validation on every turn.

    Returns:
        (ok, msg, per_turn_metadata)
        - ok: True if all turns pass.
        - msg: error string if ok is False.
        - per_turn_metadata: list aligned with conversation; each element
          contains {"bert_anchor": [...], "bert_cross": [...]}.
    """
    per_turn_metadata: list[dict] = []

    for turn_idx, turn in enumerate(conversation):
        thoughts = turn.get("thoughts", [])
        response = turn.get("response", [])
        sil_positions, nonsil_positions = _split_positions(thoughts)

        if not nonsil_positions:
            # No substance to anchor; nothing to score.
            per_turn_metadata.append({"bert_anchor": [], "bert_cross": []})
            continue

        # Build pair lists (cands, refs). bert-score: F1 is symmetric for our
        # use, but we put the thought as the candidate and the response as the
        # reference for clarity.
        anchor_cands = [thoughts[i] for i in nonsil_positions]
        anchor_refs = [response[i] for i in nonsil_positions]

        # For each non-sil thought, compare against EVERY response position
        # except its own anchor — both sil/infill and other non-sil entries.
        cross_cands: list[str] = []
        cross_refs: list[str] = []
        cross_index: list[tuple[int, int]] = []  # (i_idx, j_absolute)
        for i_idx, i in enumerate(nonsil_positions):
            for j in range(len(response)):
                if j == i:
                    continue
                cross_cands.append(thoughts[i])
                cross_refs.append(response[j])
                cross_index.append((i_idx, j))

        # Single batched call combining anchor + cross
        all_cands = anchor_cands + cross_cands
        all_refs = anchor_refs + cross_refs
        _, _, f1_tensor = scorer.score(all_cands, all_refs)
        f1 = [float(x) for x in f1_tensor.tolist()]

        anchor_f1 = f1[: len(anchor_cands)]
        cross_f1 = f1[len(anchor_cands) :]

        # Build metadata records
        bert_anchor = [
            {"index": i, "f1": anchor_f1[i_idx]}
            for i_idx, i in enumerate(nonsil_positions)
        ]
        bert_cross = [
            {
                "thought_index": nonsil_positions[i_idx],
                "response_index": j,
                "f1": cross_f1[k],
            }
            for k, (i_idx, j) in enumerate(cross_index)
        ]
        per_turn_metadata.append(
            {"bert_anchor": bert_anchor, "bert_cross": bert_cross}
        )

        # Constraint 1: minimum threshold on anchor scores
        for i_idx, score in enumerate(anchor_f1):
            if score < THOUGHT_RESPONSE_BERT_MIN:
                errors["BERT_SCORE_TOO_LOW"] += 1
                pos = nonsil_positions[i_idx]
                return (
                    False,
                    (
                        f"Turn {turn_idx} pos {pos}: anchor BERT F1="
                        f"{score:.3f} < {THOUGHT_RESPONSE_BERT_MIN}"
                    ),
                    per_turn_metadata,
                )

        # Constraint 2: anchor > max cross involving same thought.
        # Now considers every other response position, not just <sil>.
        if cross_index:
            max_cross_per_i = [-1.0] * len(nonsil_positions)
            argmax_cross_per_i = [-1] * len(nonsil_positions)
            for k, (i_idx, j) in enumerate(cross_index):
                if cross_f1[k] > max_cross_per_i[i_idx]:
                    max_cross_per_i[i_idx] = cross_f1[k]
                    argmax_cross_per_i[i_idx] = j

            for i_idx, anchor_score in enumerate(anchor_f1):
                if anchor_score <= max_cross_per_i[i_idx]:
                    errors["BERT_ALIGNMENT_FAIL"] += 1
                    pos = nonsil_positions[i_idx]
                    offender = argmax_cross_per_i[i_idx]
                    return (
                        False,
                        (
                            f"Turn {turn_idx} pos {pos}: anchor BERT F1="
                            f"{anchor_score:.3f} <= cross F1="
                            f"{max_cross_per_i[i_idx]:.3f} at response "
                            f"position {offender} "
                            f"(another response aligns more strongly with "
                            f"this thought than its own response does)"
                        ),
                        per_turn_metadata,
                    )

    return True, None, per_turn_metadata


def validate_scaffold_thought_consistency(
    conversation: list[dict],
    original_thought_strings: list[str],
    scorer: BERTScorer,
    errors: dict,
) -> tuple[bool, str | None, list[dict]]:
    """Confirm generated thoughts are semantically near-identical to the SGD source.

    Per-turn BERT_F1(joined substance thoughts, original SGD assistant utterance).
    Scaffold mode also pins thoughts via verbatim string match (see
    `validate_against_scaffold` in validation.py); this provides a tight
    semantic floor as a backstop and contributes per-turn metadata.

    Returns:
        (ok, msg, per_turn_metadata) — per_turn_metadata aligned with
        `conversation`; each element is {"scaffold_thought_bert": {"f1": float}}.
    """
    if len(conversation) != len(original_thought_strings):
        return (
            False,
            (
                f"scaffold thought-consistency: turn count {len(conversation)} "
                f"!= scaffold count {len(original_thought_strings)}"
            ),
            [],
        )

    if not conversation:
        return True, None, []

    # Join substance lines per turn into a single candidate string.
    candidates = []
    for turn in conversation:
        substance = [t for t in turn.get("thoughts", []) if t != "<sil>"]
        candidates.append(" ".join(substance))

    # Single batched call: one (cand, ref) pair per turn.
    _, _, f1_tensor = scorer.score(candidates, original_thought_strings)
    f1_scores = [float(x) for x in f1_tensor.tolist()]

    per_turn_metadata = [
        {"scaffold_thought_bert": {"f1": f1}} for f1 in f1_scores
    ]

    for i, f1 in enumerate(f1_scores):
        if f1 < SCAFFOLD_THOUGHT_BERT_MIN:
            errors["SCAFFOLD_THOUGHT_BERT_TOO_LOW"] += 1
            return (
                False,
                (
                    f"Turn {i}: scaffold thought BERT F1={f1:.3f} < "
                    f"{SCAFFOLD_THOUGHT_BERT_MIN}"
                ),
                per_turn_metadata,
            )

    return True, None, per_turn_metadata
