"""state.json read/write + config-snapshot validation.

The state file is the run-in-progress sentinel: its presence means a run
is interrupted (or live), its absence means the cache directory is clean.
On --resume, validate_state_against_config() compares the loaded JSON's
snapshot against the current config and refuses to continue if any
critical field has changed.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.configuration import config


# Subset of config fields baked into state.json for resume validation.
CONFIG_SNAPSHOT_KEYS = (
    "EXPERIMENT_NAME", "TOPICS_FILENAME", "NUM_REQUESTS", "NUM_WORKERS",
    "PROVIDER", "MODEL_NAME", "TURNS_MIN", "TURNS_MAX",
    "SUBSTANCE_MIN", "SUBSTANCE_MAX", "MAX_RETRIES",
    "GENERATION_MODE", "SGD_DATA_PATH", "SGD_SPLITS", "SGD_SERVICES",
)

# Fields that MUST match between original run and resume — changing any
# of these invalidates the chunk plan or the run identity.
CRITICAL_KEYS = (
    "EXPERIMENT_NAME", "TOPICS_FILENAME", "NUM_REQUESTS", "NUM_WORKERS",
    "GENERATION_MODE", "SGD_SERVICES",
)


def make_config_snapshot() -> dict:
    """Partial config snapshot for resume validation."""
    return {k: getattr(config, k, None) for k in CONFIG_SNAPSHOT_KEYS}


def make_effective_config() -> dict:
    """Full set of effective config values (every DEFAULTS key)."""
    return {k: getattr(config, k, None) for k in config.DEFAULTS}


def write_effective_config(path: Path) -> None:
    """Snapshot the effective config to the cache dir.

    This is the post-defaults, post-unknown-key-drop view — the precise
    set of values the run is using, regardless of which JSON file the
    user originally passed via --config.
    """
    with open(path, "w") as f:
        json.dump(make_effective_config(), f, indent=2, sort_keys=True)


def write_state(path: Path, mode: str, chunks: list, seed: int) -> None:
    state = {
        "experiment": config.EXPERIMENT_NAME,
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "config_snapshot": make_config_snapshot(),
        "chunks": chunks,
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def read_state(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_state_against_config(state: dict) -> None:
    """Hard-error on critical mismatch; warn on non-critical."""
    snap = state.get("config_snapshot", {})
    cur = make_config_snapshot()
    for key in CRITICAL_KEYS:
        if snap.get(key) != cur.get(key):
            sys.exit(
                f"Resume aborted: critical config field {key} changed "
                f"({snap.get(key)!r} -> {cur.get(key)!r}). "
                "Critical fields must match the original run."
            )
    for key in CONFIG_SNAPSHOT_KEYS:
        if key in CRITICAL_KEYS:
            continue
        if snap.get(key) != cur.get(key):
            print(
                f"Warning: {key} changed since original run "
                f"({snap.get(key)!r} -> {cur.get(key)!r}); proceeding."
            )
