"""JSON-driven configuration loader.

Configuration values for a run come from a JSON file passed via the
`--config` flag on the CLI; see `default_config.json` for the schema and
defaults. This module exposes those values as module-level globals so that
existing package imports across the codebase continue to work.

Loading mechanism:
- src/pipeline/conv_dataset_gen.py extracts `--config` from argv early and sets the
  CONVFILL_CONFIG_PATH env var. When this module is first imported, the
  env-var auto-load below picks up that path and populates globals.
- Multiprocessing workers inherit the env var on fork/spawn, so each
  subprocess re-imports this module and sees the same values.
- If the env var is unset (e.g. `--help`, ad-hoc imports, tests), the
  module falls back to the DEFAULTS table; values that have no sane
  default (None below) stay None and any code that relies on them will
  fail later with a clear AttributeError-equivalent.
"""

import json
import os
import sys


# ---------------------------------------------------------------------------
# Schema: every accepted key + its default. Required keys are listed
# explicitly so optional None defaults, such as RANDOM_SEED, stay optional.
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "EXPERIMENT_NAME",
    "TOPICS_FILENAME",
    "EXAMPLE_PATH",
}

DEFAULTS: dict = {
    # Provider / model
    "PROVIDER": "anthropic",            # "anthropic" or "openai"
    "MODEL_NAME": "claude-opus-4-6",
    "TEMPERATURE": 1.0,
    "MAX_TOKENS": 4000,
    "CALLS_PER_MINUTE": 100,

    # Experiment identity
    "EXPERIMENT_NAME": None,
    "OUTPUT_DIR": "generated_data",
    "CACHE_DIR": "generated_data",

    # Dataset
    "TOPICS_FILENAME": None,
    "EXAMPLE_PATH": None,
    "NUM_REQUESTS": 1000,
    "NUM_WORKERS": 8,
    # If True, never request more conversations than the dataset can supply
    # without duplication, and disable cycling. In freeform mode this caps
    # at len(topics); in scaffold mode with SGD_SERVICES empty (uniform
    # sampling) it caps at the total dialogue pool; in scaffold mode with
    # SGD_SERVICES non-empty (balanced sampling) it caps at
    # len(SGD_SERVICES) * min(per-service pool size) so per-service balance
    # is preserved. NUM_REQUESTS is the upper bound when CAP_MAX=True; the
    # effective count may be smaller and is reflected in the persisted
    # chunks (so --resume picks up the same effective set).
    "CAP_MAX": False,
    "TURNS_MIN": 7,
    "TURNS_MAX": 10,
    "MAX_RETRIES": 25,
    "P2_PERSONA": "",

    # Sil schedule
    "SIL_MIN": 0,
    "SIL_MAX": 3,

    # Substance lines
    "SUBSTANCE_MIN": 1,
    "SUBSTANCE_MAX": 5,

    # Filler diversity
    "MAX_FILLER_REUSE": 2,

    # Structural validation
    "MIN_SUBSTANCE_LENGTH": 5,

    # NLI validation
    "NLI_MODEL_NAME": "MoritzLaurer/DeBERTa-v3-base-mnli",
    "CONTRADICTION_PAIR_MAX": 0.2,
    "ENTAILMENT_PAIR_MIN": 0.0,
    "CONTRADICTION_WHOLE_TURN_MAX": 0.30,
    "ENTAILMENT_WHOLE_TURN_MIN": 0.0,

    # BERT-score alignment validation
    "BERT_SCORE_MODEL": "microsoft/deberta-xlarge-mnli",
    "BERT_SCORE_NUM_LAYERS": 40,
    "THOUGHT_RESPONSE_BERT_MIN": 0.75,

    # Output
    "INCLUDE_METADATA": True,

    # Run-level random seed for shuffling and SGD scaffold selection. If
    # None (default) a fresh random seed is generated at run start and
    # persisted to state.json. If set to an int, that exact value is used,
    # making the run reproducible across invocations (same SGD dialogue
    # picks for the same NUM_REQUESTS/SGD_SERVICES). On --resume, the seed
    # always comes from state.json regardless of this value.
    "RANDOM_SEED": None,

    # Prompt template path. Default keeps the freeform template; scaffold
    # mode (added later) will point this at prompt_template_scaffold.txt.
    # Existing configs that don't set this key continue to work unchanged.
    "PROMPT_TEMPLATE_PATH": "prompt_template.txt",

    # ---- Scaffold-mode (SGD) keys --------------------------------------
    # Mode selector. "freeform" preserves the original behavior (topics
    # file -> LLM invents the conversation). "scaffold" loads dialogues
    # from the SGD dataset and asks the LLM to fill only the response
    # array around fixed user/thought content.
    "GENERATION_MODE": "freeform",
    # Filesystem root of the SGD dataset (the dir that holds train/dev/test).
    # Only consulted when GENERATION_MODE == "scaffold".
    "SGD_DATA_PATH": "dstc8-schema-guided-dialogue",
    # Which SGD splits to draw from. Used only in scaffold mode.
    "SGD_SPLITS": ["train"],
    # Service / domain names to sample from (e.g. Restaurants_1, Hotels_2).
    # Required (non-empty) in scaffold mode; ignored in freeform mode.
    "SGD_SERVICES": [],
    # BERT-score floor between the ORIGINAL SGD assistant utterance and
    # the joined substance lines of the generated `thoughts` array, per
    # turn. Since scaffold-mode thoughts are also string-matched verbatim
    # against the source, F1 should be ~1.0 in normal cases; this is a
    # tight semantic floor to catch any silent drift. Used only in
    # scaffold mode.
    "SCAFFOLD_THOUGHT_BERT_MIN": 0.95,

    # Toggle for the proper-noun visibility-window check (see
    # proper_noun_validator.py). Catches LLM "leaking" proper nouns from
    # turns earlier than the previous-turn-user boundary into a generated
    # response. Only active in scaffold mode. Set to False to disable
    # without removing call sites.
    "CHECK_SCAFFOLD_PROPER_NOUNS": True,
}

ENV_VAR = "CONVFILL_CONFIG_PATH"

# Initialize module globals to defaults so attribute access never raises
# even before load() runs (e.g. during a test import).
for _k, _v in DEFAULTS.items():
    globals()[_k] = _v
API_KEY = None  # populated after PROVIDER is known


def _api_key_for(provider: str):
    """Read the provider-appropriate API key from the environment.

    Returns None if the env var is missing — failure is deferred to the
    LLM client so that --help / tests / argparse errors don't require a
    valid key.
    """
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return None


def load(path: str) -> None:
    """Load JSON config from `path`, validate, and populate module globals.

    Also sets the CONVFILL_CONFIG_PATH env var so that subprocesses
    re-importing this module pick up the same file via the auto-load
    block at module bottom.
    """
    if not os.path.exists(path):
        sys.exit(f"config: file not found: {path}")
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"config: invalid JSON in {path}: {e}")

    if not isinstance(data, dict):
        sys.exit(f"config: top level of {path} must be a JSON object")

    unknown = sorted(set(data.keys()) - set(DEFAULTS.keys()))
    if unknown:
        # Typo protection: warn, don't crash. Likely user error.
        print(f"config: warning - unknown keys ignored: {unknown}", file=sys.stderr)

    final = {}
    missing = []
    for key, default in DEFAULTS.items():
        if key in data:
            final[key] = data[key]
        elif key in REQUIRED_KEYS:
            missing.append(key)
        else:
            final[key] = default

    if missing:
        sys.exit(
            f"config: required field(s) missing in {path}: {sorted(missing)}"
        )

    if final["PROVIDER"] not in ("anthropic", "openai"):
        sys.exit(
            f"config: PROVIDER must be 'anthropic' or 'openai', "
            f"got {final['PROVIDER']!r}"
        )

    if final["GENERATION_MODE"] not in ("freeform", "scaffold"):
        sys.exit(
            f"config: GENERATION_MODE must be 'freeform' or 'scaffold', "
            f"got {final['GENERATION_MODE']!r}"
        )

    if final["GENERATION_MODE"] == "scaffold":
        # SGD_SERVICES may be empty: that means "uniform sample across all
        # dialogues in the configured SGD_SPLITS" (no per-service balancing).
        if not os.path.isdir(final["SGD_DATA_PATH"]):
            sys.exit(
                f"config: scaffold mode requires SGD_DATA_PATH to point at "
                f"the SGD dataset root; not a directory: "
                f"{final['SGD_DATA_PATH']!r}"
            )

    globals().update(final)
    globals()["API_KEY"] = _api_key_for(final["PROVIDER"])

    # Make the path discoverable to forked subprocesses.
    os.environ[ENV_VAR] = os.path.abspath(path)


# ---------------------------------------------------------------------------
# Auto-load on import if the env var is set (the path src/pipeline/conv_dataset_gen.py
# extracts from argv before triggering config-dependent imports). Keeps
# multiprocessing workers in sync without explicit re-load calls.
# ---------------------------------------------------------------------------
_auto_path = os.environ.get(ENV_VAR)
if _auto_path:
    load(_auto_path)
