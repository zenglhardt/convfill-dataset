"""Path layout + per-worker resume cursor helpers.

`ExperimentPaths` resolves every file path the orchestrator and workers
touch from the (`EXPERIMENT_NAME`, `OUTPUT_DIR`, `CACHE_DIR`) triple. A
single instance is shared by the main thread and all worker threads.
"""

from pathlib import Path


class ExperimentPaths:
    """All file paths derived from EXPERIMENT_NAME / OUTPUT_DIR / CACHE_DIR.

    Threads share this object directly via memory; no pickling required.
    """

    def __init__(self, experiment: str, output_dir: str, cache_dir: str):
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.cache_root = Path(cache_dir) / f"{experiment}_cache"

    @property
    def output_jsonl(self) -> Path:
        return self.output_dir / f"{self.experiment}.jsonl"

    @property
    def state(self) -> Path:
        return self.cache_root / "state.json"

    @property
    def failed_topics(self) -> Path:
        return self.cache_root / "failed_topics.jsonl"

    @property
    def failures_txt(self) -> Path:
        return self.cache_root / "failures.txt"

    @property
    def config_copy(self) -> Path:
        return self.cache_root / "config.json"

    def worker_tmp(self, wid: int) -> Path:
        return self.cache_root / f"worker_{wid}.tmp"

    def worker_idx(self, wid: int) -> Path:
        return self.cache_root / f"worker_{wid}.idx"

    def worker_failed_topics(self, wid: int) -> Path:
        return self.cache_root / f"worker_{wid}.failed_topics.jsonl"

    def worker_failures_txt(self, wid: int) -> Path:
        return self.cache_root / f"worker_{wid}.failures.txt"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)


def read_idx(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text().strip()
    return int(text) if text else 0


def write_idx(path: Path, value: int) -> None:
    path.write_text(str(value))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)
