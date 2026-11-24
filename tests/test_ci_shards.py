"""Every collected test file has exactly one CI worker."""

from pathlib import Path

import pytest

from tests import ci_shards


def test_ci_shards_partition_all_test_files() -> None:
    groups = ci_shards.groups()
    files = sorted(path.as_posix() for path in Path("tests").rglob("test_*.py"))
    ci_shards.validate(groups, files)
    assert set(groups) == {"steps", "remaining"}
    assert all(groups.values())
    assert ci_shards.main(["--check"]) == 0


@pytest.mark.parametrize("assignments", [{"a": ["x"]}, {"a": ["x", "y"], "b": ["y"]}])
def test_ci_shards_refuse_missing_or_duplicate_files(assignments: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ci_shards.validate(assignments, ["x", "y"])


def test_ci_runs_matrix_and_checks_once() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "shard: [steps, remaining]" in workflow
    assert "fail-fast: false" in workflow
    assert workflow.count("timeout-minutes: 45") == 2
    for command in ("ruff check", "ruff format --check", "pyright"):
        assert workflow.count("uv run " + command) == 1
    assert workflow.count("uv run python -m tests.ci_shards --check") == 1
    assert (
        workflow.count("uv run --extra groups python -m tests.ci_shards ${{ matrix.shard }}") == 1
    )


@pytest.mark.parametrize("shard", ["steps", "remaining"])
def test_ci_shard_invokes_its_files_only(monkeypatch: pytest.MonkeyPatch, shard: str) -> None:
    commands: list[list[str]] = []

    def call(command: list[str]) -> int:
        commands.append(command)
        return 7

    monkeypatch.setattr(ci_shards.subprocess, "call", call)
    assert ci_shards.main([shard, "-q"]) == 7
    assert commands == [
        [ci_shards.sys.executable, "-m", "pytest", *ci_shards.groups()[shard], "-q"]
    ]
