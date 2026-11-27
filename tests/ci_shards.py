"""Disjoint test-file groups for the three CI workers."""

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

STEP_FILES = (
    "tests/test_run.py",
    "tests/test_interaction_steps.py",
    "tests/test_interaction_step_faults.py",
    "tests/test_interaction_recovery.py",
    "tests/test_interaction_continuation.py",
    "tests/test_interaction_http_boundary.py",
)


SERVICE_FILES = (
    "tests/test_terminal_contracts.py",
    "tests/test_fault_matrix.py",
    "tests/test_interaction_endings.py",
    "tests/test_operator_http.py",
    "tests/test_interaction_service_boundary.py",
)


def validate(shards: Mapping[str, Sequence[str]], files: Sequence[str]) -> None:
    assigned = [path for group in shards.values() for path in group]
    if set(assigned) != set(files) or any(assigned.count(path) != 1 for path in files):
        raise ValueError("each test file must belong to exactly one shard")


def groups() -> dict[str, tuple[str, ...]]:
    if any(Path("tests").rglob("*_test.py")):
        raise ValueError("test files must use the test_*.py naming pattern")
    files = tuple(sorted(path.as_posix() for path in Path("tests").rglob("test_*.py")))
    shards = {
        "steps": STEP_FILES,
        "service": SERVICE_FILES,
        "remaining": tuple(path for path in files if path not in STEP_FILES + SERVICE_FILES),
    }
    validate(shards, files)
    return shards


def main(arguments: Sequence[str]) -> int:
    shards = groups()
    if list(arguments) == ["--check"]:
        return 0
    if not arguments or arguments[0] not in shards:
        raise ValueError("choose steps, service or remaining")
    return subprocess.call([sys.executable, "-m", "pytest", *shards[arguments[0]], *arguments[1:]])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
