"""LLM call wrappers, validation pipeline, and the per-topic worker loop.

Designed to run inside a `concurrent.futures.ThreadPoolExecutor`:

- Validation models load **once** in the main thread via `init_models()`.
  All worker threads then read from the same `_model_state` dict.
- Validation forward passes are serialised by a process-wide `Lock`,
  which sidesteps thread-safety questions about transformers /
  BERTScorer internals at negligible cost (~hundreds of ms vs ~30s API).
- The Anthropic / OpenAI HTTP client is naturally thread-safe; the
  `@limits` decorator from `ratelimit` is process-wide, so the configured
  `CALLS_PER_MINUTE` becomes a global cap across all worker threads.
"""

import json
import random
import re
import threading
import time

try:
    from ratelimit import limits, RateLimitException
except ModuleNotFoundError:
    class RateLimitException(Exception):
        pass

    def limits(*_args, **_kwargs):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                raise ModuleNotFoundError(
                    "ratelimit is required to run dataset generation"
                )

            return wrapper

        return decorator

from src.configuration.config import (
    CALLS_PER_MINUTE,
    EXAMPLE_PATH,
    GENERATION_MODE,
    INCLUDE_METADATA,
    SIL_MAX,
    SIL_MIN,
    SUBSTANCE_MAX,
    SUBSTANCE_MIN,
    TURNS_MAX,
    TURNS_MIN,
)
from src.validators.validation import create_error_tracker, validate_conversation
from src.prompts.sil_schedule import generate_sil_schedule
from src.prompts.create_prompt import create_prompt, create_scaffold_prompt
from src.pipeline.sgd_loader import apply_sil_schedule
from src.pipeline.llm_client import chat, LLMError, LLMRateLimitError, LLMEmptyResponseError
from src.configuration.paths import count_lines, read_idx, write_idx


# ---------------------------------------------------------------------------
# Shared state across worker threads
# ---------------------------------------------------------------------------
_validation_lock = threading.Lock()
_model_state: dict | None = None
_validate_nli = None
_validate_alignment = None
_validate_scaffold_thought_consistency = None


def init_models() -> dict:
    """Load NLI + BERT-scorer once. Idempotent.

    Returns the shared dict so callers can also reference it directly,
    though the canonical usage is to leave it in `_model_state` and let
    `process_topic_chunk` read from there.
    """
    global _model_state
    global _validate_nli
    global _validate_alignment
    global _validate_scaffold_thought_consistency
    if _model_state is None:
        from src.validators.nli_validation import load_nli_model, validate_nli
        from src.validators.alignment_validation import (
            load_bert_scorer,
            validate_alignment,
            validate_scaffold_thought_consistency,
        )

        _validate_nli = validate_nli
        _validate_alignment = validate_alignment
        _validate_scaffold_thought_consistency = (
            validate_scaffold_thought_consistency
        )
        _model_state = {
            "nli": load_nli_model(),
            "bert": load_bert_scorer(),
        }
    return _model_state


# ---------------------------------------------------------------------------
# LLM call wrappers (rate-limited, with one-shot correction)
# ---------------------------------------------------------------------------
SCAFFOLD_ASSISTANT_ACK = (
    "Understood. I will fill in the JSON scaffold you provide, "
    "replacing all placeholders with appropriate content while "
    "keeping the structure, array lengths, and \"<sil>\" entries "
    "exactly as given. Please provide the scaffold."
)

# Bounded backoff schedule for transient API failures (timeouts, 5xx,
# rate-limit). Anthropic SDK already retries 2x internally with its own
# exponential backoff; this layer catches anything that survives that.
# Total wall (~8 min) before we give up and surface the error to the
# topic-level MAX_RETRIES loop as a normal failure.
_API_RETRY_DELAYS = (5, 15, 45, 120, 300)


def _call_with_backoff(label: str, fn, *args, **kwargs):
    """Run fn() with iterative bounded backoff on API errors.

    Returns (result, None) on success or (None, last_exc) if the retry
    schedule is exhausted. Catches both Anthropic-side errors wrapped by
    llm_client and our own self-imposed CALLS_PER_MINUTE rate-limit
    exceptions from the @limits decorator.
    """
    last_exc = None
    for attempt in range(len(_API_RETRY_DELAYS) + 1):
        try:
            return fn(*args, **kwargs), None
        except LLMEmptyResponseError as e:
            # Empty response is content-driven (e.g. refusal) and almost
            # always deterministic. Don't burn the backoff schedule —
            # surface immediately so the outer topic retry can try a
            # fresh sil schedule / generation.
            return None, e
        except (RateLimitException, LLMRateLimitError, LLMError) as e:
            last_exc = e
            if attempt >= len(_API_RETRY_DELAYS):
                return None, e
            delay = _API_RETRY_DELAYS[attempt]
            print(
                f"  {label} attempt {attempt + 1} failed "
                f"({type(e).__name__}: {e}) — backing off {delay}s"
            )
            time.sleep(delay)
    # Loop guarantees one of the return paths above; this is unreachable.
    return None, last_exc


# Single shared @limits decorator instance — applied to BOTH the primary
# generation call and the correction follow-up so they share one counter.
# Two independent @limits invocations would create two independent counters,
# letting the effective combined rate reach 2 x CALLS_PER_MINUTE.
_rate_limit = limits(calls=CALLS_PER_MINUTE, period=60)


@_rate_limit
def generate_with_rate_limit(prompt: str, scaffold: str = "") -> str:
    """Send a prompt to the configured LLM and return the assistant reply.

    Uses a multi-turn structure: the instructions go first, the model
    commits to following the scaffold, then the scaffold is provided as
    the final user message so it's the last thing the model sees.
    """
    if scaffold:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": SCAFFOLD_ASSISTANT_ACK},
            {
                "role": "user",
                "content": (
                    "Fill in this scaffold. Return ONLY the completed JSON:\n\n"
                    + scaffold
                ),
            },
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    return chat(messages)


@_rate_limit
def request_correction(prompt: str, scaffold: str, bad_output: str,
                       err_msg: str, err_type: str) -> str:
    """Follow-up correction request after a structural failure.

    Continues the same chat: the model sees its own bad output and a
    targeted error message. Returns the model's corrected reply.
    """
    correction_prompt = (
        f"That output had a {err_type}: {err_msg}\n\n"
        "Return the CORRECTED JSON. Match the scaffold structure exactly:\n"
        "- Each turn must have the exact number of \"<sil>\" entries from the scaffold\n"
        "- thoughts and response arrays must have the same length per turn\n"
        "- All placeholders must be replaced with actual content\n"
        "- Use only English alphabet characters\n\n"
        "Return ONLY the corrected JSON, no commentary."
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": SCAFFOLD_ASSISTANT_ACK},
        {
            "role": "user",
            "content": (
                "Fill in this scaffold. Return ONLY the completed JSON:\n\n"
                + scaffold
            ),
        },
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": correction_prompt},
    ]
    return chat(messages)


def parse_and_format_check(
    output: str,
    sil_schedule: list[int],
    errors: dict,
    scaffold_expectation: dict | None = None,
):
    """Parse the model output and run formatting checks.

    Returns (ok, conv, err_msg, err_type).
    On success: (True, conv_dict, None, None).
    On failure: (False, conv_or_None, error_message, error_type_string).

    `scaffold_expectation` is forwarded to `validate_conversation`; freeform
    callers leave it as None and the new check is skipped.
    """
    cleaned = (
        output.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        conv = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, None, str(e), "JSON decode error"

    if re.search(r"<(USER|THOUGHT_\d+|RESPONSE_\d+|INFILL_\d+)>", json.dumps(conv)):
        return False, conv, "Unreplaced placeholder tags found", "placeholder leakage"

    ok, msg = validate_conversation(
        conv, sil_schedule, errors, scaffold_expectation=scaffold_expectation
    )
    if not ok:
        return False, conv, msg, "structural validation error"

    return True, conv, None, None


# ---------------------------------------------------------------------------
# Failure logging
# ---------------------------------------------------------------------------
def log_failure(paths, wid, topic, p1, p2, error_type, msg,
                raw="", cleaned="", parsed=None):
    """Append a debug entry to the per-worker failures.txt log."""
    path = paths.worker_failures_txt(wid)
    with open(path, "a") as f:
        f.write(f"=== {error_type} ===\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"P1: {p1}\nP2: {p2}\n")
        f.write(f"Error: {msg}\n")
        if raw:
            f.write(f"Raw Output:\n{raw}\n")
        if cleaned:
            f.write(f"Cleaned Output:\n{cleaned}\n")
        if parsed is not None:
            f.write(f"Parsed JSON:\n{json.dumps(parsed, indent=2)}\n")
        f.write(f"{'=' * 50}\n\n")


def log_failed_topic(paths, wid, entry, error_type, last_error, attempts):
    """Append a permanent-failure record (machine-readable) for --retry.

    Persists every field from `entry` (including `scaffold_turns` for
    scaffold mode) plus the failure metadata. `topics.load_topics_from_failed`
    strips the metadata fields and returns the rest as the entry on retry.
    """
    rec = {
        **entry,
        "error_type": error_type,
        "last_error": last_error,
        "attempts": attempts,
    }
    with open(paths.worker_failed_topics(wid), "a") as f:
        json.dump(rec, f)
        f.write("\n")


# ---------------------------------------------------------------------------
# Single-conversation generation + validation
# ---------------------------------------------------------------------------
def generate_conversation_data(topic, p1, p2, errors, wid, model_state, paths):
    """Generate and validate a single conversation.

    Returns (conv_dict, last_error_str). conv_dict is None on failure;
    last_error_str describes the last error encountered (empty on success).
    """
    num_turns = random.randint(TURNS_MIN, TURNS_MAX)
    sil_schedule = generate_sil_schedule(num_turns)

    prompt, scaffold, _substance_counts = create_prompt(
        topic, p1, p2, num_turns, sil_schedule,
        SUBSTANCE_MIN, SUBSTANCE_MAX, EXAMPLE_PATH,
    )

    try:
        output, api_err = _call_with_backoff(
            "API call", generate_with_rate_limit, prompt, scaffold
        )
        if api_err is not None:
            errors["API_EXHAUSTED"] += 1
            msg = f"API RETRIES EXHAUSTED ({type(api_err).__name__}): {api_err}"
            print(f"  {msg}")
            log_failure(paths, wid, topic, p1, p2, "API RETRIES EXHAUSTED",
                        str(api_err))
            return None, msg

        ok, conv, err_msg, err_type = parse_and_format_check(
            output, sil_schedule, errors
        )

        if not ok:
            print(f"  {err_type}: {err_msg}")
            print(f"  Attempting one-shot correction...")
            errors["ONE_SHOT_ATTEMPTED"] += 1
            corrected_output, api_err = _call_with_backoff(
                "correction",
                request_correction,
                prompt, scaffold, output, err_msg, err_type,
            )
            if api_err is not None:
                print(f"  One-shot correction API retries exhausted: {api_err}")
                errors["ONE_SHOT_FAILED"] += 1
                errors["API_EXHAUSTED"] += 1
                log_failure(paths, wid, topic, p1, p2, "ONE-SHOT API ERROR",
                            f"{err_type}: {err_msg} | API error: {api_err}",
                            output, "", conv)
                return None, f"ONE-SHOT API ERROR: {err_type}: {err_msg} | {api_err}"

            ok, conv, err_msg, err_type = parse_and_format_check(
                corrected_output, sil_schedule, errors
            )
            if not ok:
                print(f"  One-shot correction also failed: {err_type}: {err_msg}")
                errors["ONE_SHOT_FAILED"] += 1
                log_failure(paths, wid, topic, p1, p2, "ONE-SHOT FAILED",
                            f"{err_type}: {err_msg}",
                            corrected_output, "", conv)
                return None, f"ONE-SHOT FAILED: {err_type}: {err_msg}"

            print(f"  One-shot correction succeeded.")
            errors["ONE_SHOT_SUCCEEDED"] += 1
            output = corrected_output  # for downstream failure logging

        # Validation behind a shared lock so concurrent threads don't fight
        # over the GPU or transformers/BERTScorer internals.
        tokenizer, nli_model, device = model_state["nli"]
        scorer = model_state["bert"]
        with _validation_lock:
            ok, msg, nli_metadata = _validate_nli(
                conv["conversation"], tokenizer, nli_model, device, errors
            )
            if ok:
                ok, msg, bert_metadata = _validate_alignment(
                    conv["conversation"], scorer, errors
                )
                stage = "ALIGNMENT VALIDATION FAILED"
            else:
                stage = "NLI VALIDATION FAILED"
                bert_metadata = None

        if not ok:
            print(f"  {stage}: {msg}")
            log_failure(paths, wid, topic, p1, p2, stage, msg, output, "", conv)
            return None, f"{stage}: {msg}"

        if INCLUDE_METADATA:
            for turn, nli_md, bert_md in zip(
                conv["conversation"], nli_metadata, bert_metadata
            ):
                turn["metadata"] = {**nli_md, **bert_md}

        return conv, ""

    except Exception as e:
        # API errors are now handled by _call_with_backoff above; anything
        # reaching this branch is genuinely unexpected (programming bug,
        # bad input shape from the LLM that slipped past validation, etc.).
        import traceback
        tb = traceback.format_exc()
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        log_failure(paths, wid, topic, p1, p2, "UNEXPECTED ERROR",
                    f"{type(e).__name__}: {e}\n{tb}")
        return None, f"UNEXPECTED ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Scaffold-mode single-conversation generation
# ---------------------------------------------------------------------------
def generate_conversation_from_scaffold(entry, errors, wid, model_state, paths):
    """Generate + validate one scaffold-mode conversation.

    Mirrors `generate_conversation_data` but driven by an SGD scaffold:
    user turns and thought sentences are pinned (verbatim), <sil> slots
    are added programmatically, and the LLM only fills response substance
    + infill positions. Adds the scaffold-thought BERT-consistency check
    on top of the standard NLI + alignment cascade.

    Returns (conv_dict, last_error_str). conv_dict is None on failure.
    """
    topic = entry["topic"]
    p1 = entry["p1"]
    p2 = entry["p2"]
    service = entry.get("service", "unknown")

    # Decorate each scaffold turn with a randomised <sil> count and build
    # the parallel structures the validators expect.
    decorated = [
        apply_sil_schedule(t, SIL_MIN, SIL_MAX, random)
        for t in entry["scaffold_turns"]
    ]
    sil_schedule = [t["num_sils"] for t in decorated]
    scaffold_expectation = {
        "users": [t["user"] for t in decorated],
        "thoughts": [
            list(src["thought_sentences"]) for src in entry["scaffold_turns"]
        ],
    }
    original_thought_strings = [
        src["original_thought"] for src in entry["scaffold_turns"]
    ]

    prompt, scaffold_json = create_scaffold_prompt(
        p1=p1, p2=p2, service=service,
        sgd_scaffold_turns=decorated,
        example_path=EXAMPLE_PATH,
    )

    try:
        output, api_err = _call_with_backoff(
            "API call", generate_with_rate_limit, prompt, scaffold_json
        )
        if api_err is not None:
            errors["API_EXHAUSTED"] += 1
            msg = f"API RETRIES EXHAUSTED ({type(api_err).__name__}): {api_err}"
            print(f"  {msg}")
            log_failure(paths, wid, topic, p1, p2, "API RETRIES EXHAUSTED",
                        str(api_err))
            return None, msg

        ok, conv, err_msg, err_type = parse_and_format_check(
            output, sil_schedule, errors,
            scaffold_expectation=scaffold_expectation,
        )

        if not ok:
            print(f"  {err_type}: {err_msg}")
            print(f"  Attempting one-shot correction...")
            errors["ONE_SHOT_ATTEMPTED"] += 1
            corrected_output, api_err = _call_with_backoff(
                "correction", request_correction,
                prompt, scaffold_json, output, err_msg, err_type,
            )
            if api_err is not None:
                print(f"  One-shot correction API retries exhausted: {api_err}")
                errors["ONE_SHOT_FAILED"] += 1
                errors["API_EXHAUSTED"] += 1
                log_failure(paths, wid, topic, p1, p2, "ONE-SHOT API ERROR",
                            f"{err_type}: {err_msg} | API error: {api_err}",
                            output, "", conv)
                return None, f"ONE-SHOT API ERROR: {err_type}: {err_msg} | {api_err}"

            ok, conv, err_msg, err_type = parse_and_format_check(
                corrected_output, sil_schedule, errors,
                scaffold_expectation=scaffold_expectation,
            )
            if not ok:
                print(f"  One-shot correction also failed: {err_type}: {err_msg}")
                errors["ONE_SHOT_FAILED"] += 1
                log_failure(paths, wid, topic, p1, p2, "ONE-SHOT FAILED",
                            f"{err_type}: {err_msg}",
                            corrected_output, "", conv)
                return None, f"ONE-SHOT FAILED: {err_type}: {err_msg}"

            print(f"  One-shot correction succeeded.")
            errors["ONE_SHOT_SUCCEEDED"] += 1
            output = corrected_output

        # Validators behind the shared lock.
        tokenizer, nli_model, device = model_state["nli"]
        scorer = model_state["bert"]
        bert_metadata = None
        consistency_metadata = None
        with _validation_lock:
            ok, msg, nli_metadata = _validate_nli(
                conv["conversation"], tokenizer, nli_model, device, errors
            )
            stage = "NLI VALIDATION FAILED" if not ok else None

            if ok:
                ok, msg, bert_metadata = _validate_alignment(
                    conv["conversation"], scorer, errors
                )
                if not ok:
                    stage = "ALIGNMENT VALIDATION FAILED"

            if ok:
                ok, msg, consistency_metadata = (
                    _validate_scaffold_thought_consistency(
                        conv["conversation"],
                        original_thought_strings,
                        scorer,
                        errors,
                    )
                )
                if not ok:
                    stage = "SCAFFOLD THOUGHT CONSISTENCY FAILED"

        if not ok:
            print(f"  {stage}: {msg}")
            log_failure(paths, wid, topic, p1, p2, stage, msg, output, "", conv)
            return None, f"{stage}: {msg}"

        if INCLUDE_METADATA:
            for turn, nli_md, bert_md, cons_md in zip(
                conv["conversation"], nli_metadata, bert_metadata,
                consistency_metadata,
            ):
                turn["metadata"] = {**nli_md, **bert_md, **cons_md}
            # Conversation-level provenance: which SGD service / dialogue
            # this scaffold came from. Useful for downstream analysis by
            # category and for tracing back to the original SGD source.
            conv["scaffold_metadata"] = {
                "service": service,
                "dialogue_id": entry.get("dialogue_id"),
            }

        return conv, ""

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        log_failure(paths, wid, topic, p1, p2, "UNEXPECTED ERROR",
                    f"{type(e).__name__}: {e}\n{tb}")
        return None, f"UNEXPECTED ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Worker-thread entry point
# ---------------------------------------------------------------------------
def append_to_jsonl(path, obj) -> None:
    with open(path, "a") as f:
        json.dump(obj, f)
        f.write("\n")


def process_topic_chunk(args):
    """Process one chunk of {topic, p1, p2} dicts in a worker thread.

    Resumes from the persisted worker_idx file if present. Permanent
    failures (exhausted retries) are appended to
    worker_N.failed_topics.jsonl for later --retry.
    """
    chunk, wid, paths, retries = args
    if _model_state is None:
        raise RuntimeError(
            "process_topic_chunk: _model_state not initialised. "
            "Call init_models() in the main thread before submitting work."
        )

    errors = create_error_tracker()

    idx_path = paths.worker_idx(wid)
    tmp_path = paths.worker_tmp(wid)

    idx = read_idx(idx_path)
    completed = count_lines(tmp_path)

    if idx > 0 or completed > 0:
        print(f"[w{wid}] resuming at chunk index {idx} (already completed {completed})")

    is_scaffold = GENERATION_MODE == "scaffold"

    while idx < len(chunk):
        entry = chunk[idx]
        topic = entry["topic"]

        print(f"[w{wid}] topic {idx + 1}/{len(chunk)} (completed {completed})")
        conv = None
        last_error = ""
        attempts = 0
        topic_t0 = time.monotonic()
        for r in range(retries + 1):
            attempts = r + 1
            if is_scaffold:
                conv, err = generate_conversation_from_scaffold(
                    entry, errors, wid, _model_state, paths
                )
            else:
                conv, err = generate_conversation_data(
                    entry["topic"], entry["p1"], entry["p2"],
                    errors, wid, _model_state, paths,
                )
            if conv:
                break
            last_error = err
            if r < retries:
                print(f"[w{wid}]   retry {r + 1}/{retries}")
        topic_dt = time.monotonic() - topic_t0

        if conv:
            append_to_jsonl(str(tmp_path), conv)
            completed += 1
            # Rough generated-content size: char count of the JSON body.
            chars = len(json.dumps(conv, ensure_ascii=False))
            print(
                f"[w{wid}] topic {idx + 1} done in {topic_dt:.1f}s "
                f"(attempts={attempts}, ~{chars} chars)"
            )
        else:
            print(
                f"[w{wid}] topic {idx + 1} exhausted retries in {topic_dt:.1f}s "
                f"(attempts={attempts}): '{topic[:50]}...'"
            )
            log_failed_topic(paths, wid, entry,
                             "EXHAUSTED_RETRIES", last_error, attempts)

        idx += 1
        write_idx(idx_path, idx)

    return completed, wid, errors
