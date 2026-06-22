"""Conversation dataset generator — orchestrator entry point.

Generates spoken conversations with variable filler counts, diverse
fillers, and NLI- + BERT-score-based semantic validation. The heavy
lifting lives in `generation.py`; this file just handles config bootstrap,
mode dispatch (fresh / --resume / --retry), and end-of-run merge.
"""

import argparse
import glob
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Config bootstrap (must run before any config-dependent imports below).
#
# argparse can't run yet — it would fail on --resume/--retry without
# --config — so we scan argv just enough to extract --config and stash
# the path in the env var that config.py auto-loads from.
# ---------------------------------------------------------------------------
def _extract_config_arg(argv: list) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


_cfg_path = _extract_config_arg(sys.argv[1:])
if _cfg_path and not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    os.environ["CONVFILL_CONFIG_PATH"] = os.path.abspath(_cfg_path)
# If --config is missing, the formal argparse below will surface the
# error. Defer rather than fail here so --help still works.


from src.configuration.config import (
    CACHE_DIR,
    CAP_MAX,
    EXPERIMENT_NAME,
    GENERATION_MODE,
    MAX_RETRIES,
    MODEL_NAME,
    NUM_REQUESTS,
    NUM_WORKERS,
    OUTPUT_DIR,
    P2_PERSONA,
    RANDOM_SEED,
    SGD_DATA_PATH,
    SGD_SERVICES,
    SGD_SPLITS,
    TOPICS_FILENAME,
    TURNS_MAX,
    TURNS_MIN,
)


def _pick_seed() -> int:
    """Honor config.RANDOM_SEED if set; else generate a fresh random seed."""
    if RANDOM_SEED is not None:
        return int(RANDOM_SEED)
    return random.randint(0, 2**31 - 1)
from src.validators.validation import create_error_tracker
from src.configuration.paths import ExperimentPaths
from src.configuration.state import (
    read_state,
    validate_state_against_config,
    write_effective_config,
    write_state,
)
from src.pipeline.topics import (
    create_cycling_topics,
    load_scaffolds_from_sgd,
    load_topics_from_failed,
    load_topics_from_file,
    split_list_into_chunks,
)

# ---------------------------------------------------------------------------
# End-of-run merge + cleanup
# ---------------------------------------------------------------------------
def merge_temp_files(paths: ExperimentPaths, *, append: bool) -> int:
    """Merge per-worker .tmp files into the final JSONL.

    append=False overwrites; append=True extends an existing dataset
    (used by --retry so successes are added to the original output).
    """
    mode = "a" if append else "w"
    total = 0
    pattern = str(paths.cache_root / "worker_*.tmp")
    with open(paths.output_jsonl, mode) as main:
        for tmp in sorted(glob.glob(pattern)):
            with open(tmp) as t:
                for line in t:
                    main.write(line)
                    total += 1
            os.remove(tmp)
    return total


def merge_failed_topics(paths: ExperimentPaths) -> int:
    """Rewrite failed_topics.jsonl from per-worker shards.

    For fresh/resume runs this is a fresh write of every permanent
    failure. For retry runs the rewrite naturally excludes anything
    that succeeded this run (those went to worker_N.tmp, not the
    failure shards).
    """
    pattern = str(paths.cache_root / "worker_*.failed_topics.jsonl")
    shards = sorted(glob.glob(pattern))
    total = 0
    with open(paths.failed_topics, "w") as out:
        for shard in shards:
            with open(shard) as f:
                for line in f:
                    out.write(line)
                    total += 1
            os.remove(shard)
    if total == 0 and paths.failed_topics.exists():
        paths.failed_topics.unlink()
    return total


def merge_failures_txt(paths: ExperimentPaths, *, append: bool) -> None:
    """Concatenate per-worker debug logs into the consolidated failures.txt."""
    pattern = str(paths.cache_root / "worker_*.failures.txt")
    shards = sorted(glob.glob(pattern))
    if not shards:
        return
    mode = "a" if append else "w"
    with open(paths.failures_txt, mode) as out:
        for shard in shards:
            with open(shard) as f:
                out.write(f.read())
            os.remove(shard)


def cleanup_state(paths: ExperimentPaths) -> None:
    """Remove the run-in-progress sentinels.

    Keeps `config.json` (effective-config snapshot) and `failures.txt`
    (consolidated debug log) — both are forensic records useful after the
    run completes. Re-running fresh/retry overwrites `config.json`; the
    fresh-run guard errors on the output `.jsonl` already existing, so
    leaving `config.json` behind doesn't conflict with anything.
    """
    if paths.state.exists():
        paths.state.unlink()
    for idx_file in glob.glob(str(paths.cache_root / "worker_*.idx")):
        os.remove(idx_file)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
def plan_fresh_run(paths: ExperimentPaths) -> tuple[list, int]:
    """Build chunks for a fresh run. Returns (chunks, seed).

    Branches on `GENERATION_MODE`:
      - "freeform" — load topics from `TOPICS_FILENAME`, shuffle + cycle.
      - "scaffold" — sample SGD dialogues balanced across `SGD_SERVICES`.
    """
    if paths.output_jsonl.exists():
        sys.exit(
            f"Output file {paths.output_jsonl} already exists. "
            "Move/rename it to start a fresh run, or pass --resume / --retry."
        )
    if paths.state.exists():
        sys.exit(
            f"State file {paths.state} exists from a prior run. "
            "Pass --resume to continue it, or remove the cache directory "
            "to start over."
        )

    seed = _pick_seed()

    if GENERATION_MODE == "scaffold":
        topics = load_scaffolds_from_sgd(
            SGD_DATA_PATH, SGD_SPLITS, SGD_SERVICES, NUM_REQUESTS, seed,
            cap_max=CAP_MAX,
        )
    else:
        topics = load_topics_from_file(TOPICS_FILENAME, P2_PERSONA, "the user")
        rng = random.Random(seed)
        rng.shuffle(topics)
        topics = create_cycling_topics(topics, NUM_REQUESTS, cap_max=CAP_MAX)

    num_workers = min(NUM_WORKERS, len(topics))
    chunks = split_list_into_chunks(topics, num_workers)
    return chunks, seed


def plan_retry_run(paths: ExperimentPaths) -> tuple[list, int]:
    """Build chunks from failed_topics.jsonl. Returns (chunks, seed)."""
    if paths.state.exists():
        sys.exit(
            f"State file {paths.state} exists. A run is in progress; use "
            "--resume to continue it before starting a new --retry."
        )
    if not paths.failed_topics.exists():
        sys.exit(
            f"No failed_topics.jsonl found at {paths.failed_topics}. "
            "Nothing to retry."
        )
    topics = load_topics_from_failed(paths.failed_topics)
    if not topics:
        sys.exit("failed_topics.jsonl is empty. Nothing to retry.")

    seed = _pick_seed()
    num_workers = min(NUM_WORKERS, len(topics))
    chunks = split_list_into_chunks(topics, num_workers)
    return chunks, seed


def plan_resume_run(paths: ExperimentPaths) -> tuple[list, int, str]:
    """Reload chunks from state.json. Returns (chunks, seed, mode)."""
    if not paths.state.exists():
        sys.exit(
            f"No state file at {paths.state}; nothing to resume. "
            "Start a fresh run (no flags) or use --retry."
        )
    state = read_state(paths.state)
    validate_state_against_config(state)
    return state["chunks"], state.get("random_seed", 0), state.get("mode", "fresh")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the convfill conversation dataset.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the JSON config file (see default_config.json).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run from cached state.",
    )
    mode_group.add_argument(
        "--retry", action="store_true",
        help="Re-run the topics in failed_topics.jsonl.",
    )
    args_cli = parser.parse_args()

    from src.pipeline import generation

    paths = ExperimentPaths(EXPERIMENT_NAME, OUTPUT_DIR, CACHE_DIR)
    paths.ensure_dirs()

    if args_cli.resume:
        chunks, seed, mode = plan_resume_run(paths)
    elif args_cli.retry:
        chunks, seed = plan_retry_run(paths)
        mode = "retry"
        write_state(paths.state, mode, chunks, seed)
        write_effective_config(paths.config_copy)
    else:
        chunks, seed = plan_fresh_run(paths)
        mode = "fresh"
        write_state(paths.state, mode, chunks, seed)
        write_effective_config(paths.config_copy)

    random.seed(seed)
    num_workers = len(chunks)
    total_topics = sum(len(c) for c in chunks)

    print(f"Mode: {mode} | Topics in plan: {total_topics} | Workers: {num_workers}")
    print(f"Model: {MODEL_NAME} | Turns: {TURNS_MIN}-{TURNS_MAX}")
    print(f"Output: {paths.output_jsonl}")
    print(f"Cache:  {paths.cache_root}\n")

    # Load validation models once, shared across all worker threads.
    generation.init_models()

    pool_args = [
        (chunk, i, paths, MAX_RETRIES)
        for i, chunk in enumerate(chunks)
    ]

    if num_workers == 1:
        results = [generation.process_topic_chunk(pool_args[0])]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as exe:
            results = list(exe.map(generation.process_topic_chunk, pool_args))

    written = merge_temp_files(paths, append=(mode == "retry"))
    failed_count = merge_failed_topics(paths)
    merge_failures_txt(paths, append=(mode in ("retry",)))
    cleanup_state(paths)

    total_completed = sum(r[0] for r in results)
    merged_errors = create_error_tracker()
    for _, _, worker_errors in results:
        for key in merged_errors:
            merged_errors[key] += worker_errors.get(key, 0)

    print("\n" + "=" * 50)
    print("Generation complete.")
    print(f"Total completed this run: {total_completed}")
    print(f"Lines written to {paths.output_jsonl}: {written}")
    if failed_count:
        print(f"Permanently failed topics in {paths.failed_topics}: {failed_count}")

    print("\nValidation Error Summary:")
    any_errors = False
    for error_type, count in merged_errors.items():
        if count > 0:
            print(f"  {error_type}: {count}")
            any_errors = True
    if not any_errors:
        print("  (no errors)")

    if paths.failures_txt.exists():
        print(f"\nFailed attempts (debug log): {paths.failures_txt}")


if __name__ == "__main__":
    main()
