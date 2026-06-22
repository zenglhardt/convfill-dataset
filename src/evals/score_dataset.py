"""Score an existing conversations JSONL with the configured NLI + BERT
metrics and dump every per-turn measurement plus summary stats to a JSON
file. Useful for threshold calibration: change models / layers / thresholds
in your config, re-run on a known-good dataset, observe distributions.

Usage:
    python -m src.evals.score_dataset \
        --config <path.json> \
        --input  generated_data/conversations_v3_education_full.jsonl \
        --output generated_data/conversations_v3_education_full.scores.json \
        [--limit N]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Config bootstrap (must run before any config-dependent imports below).
# Same pattern as conv_dataset_gen.py — lift --config out of argv early so
# config.py auto-loads from CONVFILL_CONFIG_PATH on import.
# ---------------------------------------------------------------------------
def _extract_config_arg(argv):
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


_cfg_path = _extract_config_arg(sys.argv[1:])
if _cfg_path and not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    os.environ["CONVFILL_CONFIG_PATH"] = os.path.abspath(_cfg_path)


from src.configuration import config
compute_nli_probs = None
load_nli_model = None
load_bert_scorer = None


CONFIG_KEYS_TO_SNAPSHOT = (
    "BERT_SCORE_MODEL", "BERT_SCORE_NUM_LAYERS",
    "NLI_MODEL_NAME",
    "THOUGHT_RESPONSE_BERT_MIN",
    "CONTRADICTION_PAIR_MAX", "ENTAILMENT_PAIR_MIN",
    "CONTRADICTION_WHOLE_TURN_MAX", "ENTAILMENT_WHOLE_TURN_MIN",
)


# ---------------------------------------------------------------------------
# Per-turn scoring (no gating — every value is computed and returned)
# ---------------------------------------------------------------------------
def score_turn(turn: dict, tokenizer, nli_model, nli_device, scorer) -> dict:
    """Compute NLI per-pair, NLI whole-turn, BERT anchor, and BERT cross
    (against every other response position) for one turn. Mirrors what
    the validator computes, minus the gating short-circuits.
    """
    user = turn.get("user", "")
    thoughts = turn.get("thoughts", [])
    response = turn.get("response", [])

    # ---- NLI: per-pair (non-sil thought vs same-position response) ----
    pair_premises, pair_hypotheses, pair_idx = [], [], []
    for i, (t, r) in enumerate(zip(thoughts, response)):
        if t.strip() in ("<sil>", ""):
            continue
        pair_premises.append(t)
        pair_hypotheses.append(r)
        pair_idx.append(i)

    pair_probs = compute_nli_probs(
        pair_premises, pair_hypotheses, tokenizer, nli_model, nli_device
    )
    nli_pair = [{"index": pair_idx[k], **probs} for k, probs in enumerate(pair_probs)]

    # ---- NLI: whole turn (user + non-sil thoughts vs user + all responses) ----
    non_sil_thoughts = [t for t in thoughts if t.strip() not in ("<sil>", "")]
    nli_whole = None
    if non_sil_thoughts:
        whole_p = [user + " " + " ".join(non_sil_thoughts)]
        whole_h = [user + " " + " ".join(response)]
        whole_probs = compute_nli_probs(
            whole_p, whole_h, tokenizer, nli_model, nli_device
        )
        if whole_probs:
            nli_whole = whole_probs[0]

    # ---- BERT-score: anchor for each non-sil; cross against EVERY other position ----
    nonsil_positions = [i for i, t in enumerate(thoughts) if t != "<sil>"]
    anchor_cands = [thoughts[i] for i in nonsil_positions]
    anchor_refs = [response[i] for i in nonsil_positions]

    cross_cands, cross_refs, cross_pairs = [], [], []
    for i in nonsil_positions:
        for j in range(len(response)):
            if j == i:
                continue
            cross_cands.append(thoughts[i])
            cross_refs.append(response[j])
            cross_pairs.append((i, j))

    bert_anchor, bert_cross = [], []
    if anchor_cands or cross_cands:
        all_cands = anchor_cands + cross_cands
        all_refs = anchor_refs + cross_refs
        _, _, f1_t = scorer.score(all_cands, all_refs)
        f1 = [float(x) for x in f1_t.tolist()]
        anchor_f1 = f1[: len(anchor_cands)]
        cross_f1 = f1[len(anchor_cands):]
        bert_anchor = [
            {"index": i, "f1": anchor_f1[k]}
            for k, i in enumerate(nonsil_positions)
        ]
        bert_cross = [
            {
                "thought_index": cross_pairs[k][0],
                "response_index": cross_pairs[k][1],
                "f1": cross_f1[k],
            }
            for k in range(len(cross_cands))
        ]

    return {
        "nli_pair": nli_pair,
        "nli_whole_turn": nli_whole,
        "bert_anchor": bert_anchor,
        "bert_cross": bert_cross,
    }


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def stats_dict(values: list) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "min": s[0],
        "p10": s[(n * 10) // 100],
        "p25": s[n // 4],
        "median": s[n // 2],
        "p75": s[(n * 3) // 4],
        "p90": s[(n * 9) // 10],
        "max": s[-1],
        "mean": sum(s) / n,
    }


def _show(label: str, s: dict) -> None:
    if not s.get("n"):
        print(f"  {label}: (empty)")
        return
    print(
        f"  {label}: n={s['n']:<6}  "
        f"min={s['min']:.3f}  median={s['median']:.3f}  "
        f"mean={s['mean']:.3f}  max={s['max']:.3f}  "
        f"(p10={s['p10']:.3f} p90={s['p90']:.3f})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a conversations JSONL with the configured NLI + BERT metrics.",
    )
    parser.add_argument("--config", required=True,
                        help="JSON config file (sets which models + thresholds to use).")
    parser.add_argument("--input", required=True,
                        help="Conversations JSONL to score.")
    parser.add_argument("--output", required=True,
                        help="Output JSON file (raw per-turn scores + summary).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N conversations.")
    args = parser.parse_args()

    global compute_nli_probs, load_nli_model, load_bert_scorer
    from src.validators.nli_validation import compute_nli_probs as _compute_nli_probs
    from src.validators.nli_validation import load_nli_model as _load_nli_model
    from src.validators.alignment_validation import load_bert_scorer as _load_bert_scorer

    compute_nli_probs = _compute_nli_probs
    load_nli_model = _load_nli_model
    load_bert_scorer = _load_bert_scorer

    # Count input lines for progress
    with open(args.input) as f:
        n_total = sum(1 for line in f if line.strip())
    n_to_process = min(args.limit, n_total) if args.limit else n_total

    print(f"Input : {args.input}  ({n_total} conversations, processing {n_to_process})", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"NLI   : {config.NLI_MODEL_NAME}", flush=True)
    print(f"BERT  : {config.BERT_SCORE_MODEL}  layers={config.BERT_SCORE_NUM_LAYERS}", flush=True)
    print()

    print("Loading models...", flush=True)
    tok, nli_model, device = load_nli_model()
    scorer = load_bert_scorer()
    print()

    anchor_f1s, cross_f1s = [], []
    nli_pair_e, nli_pair_n, nli_pair_c = [], [], []
    nli_whole_e, nli_whole_n, nli_whole_c = [], [], []

    out_conversations = []
    t0 = time.monotonic()
    last_log = t0

    with open(args.input) as f:
        for ci, line in enumerate(f):
            if args.limit and ci >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns_md = []
            for turn in d.get("conversation", []):
                md = score_turn(turn, tok, nli_model, device, scorer)
                turns_md.append(md)
                for a in md["bert_anchor"]:
                    anchor_f1s.append(a["f1"])
                for c in md["bert_cross"]:
                    cross_f1s.append(c["f1"])
                for p in md["nli_pair"]:
                    nli_pair_e.append(p["entailment"])
                    nli_pair_n.append(p["neutral"])
                    nli_pair_c.append(p["contradiction"])
                if md["nli_whole_turn"]:
                    nli_whole_e.append(md["nli_whole_turn"]["entailment"])
                    nli_whole_n.append(md["nli_whole_turn"]["neutral"])
                    nli_whole_c.append(md["nli_whole_turn"]["contradiction"])
            out_conversations.append(
                {"index": ci, "n_turns": len(turns_md), "turns": turns_md}
            )

            now = time.monotonic()
            done = ci + 1
            if now - last_log >= 5 or done == n_to_process:
                rate = done / max(now - t0, 0.001)
                eta = (n_to_process - done) / max(rate, 0.001)
                print(
                    f"  [{done}/{n_to_process}] rate={rate:.2f} conv/s  eta={eta:.0f}s",
                    flush=True,
                )
                last_log = now

    elapsed = time.monotonic() - t0

    summary = {
        "anchor_f1": stats_dict(anchor_f1s),
        "cross_f1": stats_dict(cross_f1s),
        "nli_pair_entailment": stats_dict(nli_pair_e),
        "nli_pair_neutral": stats_dict(nli_pair_n),
        "nli_pair_contradiction": stats_dict(nli_pair_c),
        "nli_whole_entailment": stats_dict(nli_whole_e),
        "nli_whole_neutral": stats_dict(nli_whole_n),
        "nli_whole_contradiction": stats_dict(nli_whole_c),
    }

    config_snapshot = {
        k: getattr(config, k, None) for k in CONFIG_KEYS_TO_SNAPSHOT
    }

    out = {
        "input_file": str(Path(args.input).resolve()),
        "n_conversations": len(out_conversations),
        "elapsed_seconds": round(elapsed, 1),
        "config_snapshot": config_snapshot,
        "summary": summary,
        "conversations": out_conversations,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"Wrote {args.output} in {elapsed:.0f}s")
    print()
    print("=== Summary ===")
    _show("anchor F1 (correct pair)        ", summary["anchor_f1"])
    _show("cross  F1 (every other position)", summary["cross_f1"])
    print()
    _show("NLI pair entailment             ", summary["nli_pair_entailment"])
    _show("NLI pair contradiction          ", summary["nli_pair_contradiction"])
    _show("NLI whole-turn entailment       ", summary["nli_whole_entailment"])
    _show("NLI whole-turn contradiction    ", summary["nli_whole_contradiction"])


if __name__ == "__main__":
    main()
