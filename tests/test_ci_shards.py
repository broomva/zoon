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


def test_ci_shard_check_refuses_an_unassigned_file(monkeypatch: pytest.MonkeyPatch) -> None:
    assignments = ci_shards.groups()
    # Simulate a shard definition that silently drops one discoverable file.
    missing = assignments["remaining"][0]
    original = ci_shards.validate

    def validate(shards: dict[str, tuple[str, ...]], files: tuple[str, ...]) -> None:
        shards["remaining"] = tuple(p for p in shards["remaining"] if p != missing)
        original(shards, files)

    monkeypatch.setattr(ci_shards, "validate", validate)
    with pytest.raises(ValueError, match="exactly one"):
        ci_shards.main(["--check"])


def test_pytest_collection_matches_ci_pattern(tmp_path: Path) -> None:
    import subprocess
    import sys
    import tomllib

    configuration = Path("pyproject.toml").read_bytes()
    settings = tomllib.loads(configuration.decode())["tool"]["pytest"]["ini_options"]
    assert settings["python_files"] == ["test_*.py"]
    (tmp_path / "pyproject.toml").write_bytes(configuration)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_included.py").write_text("def test_included(): assert True\n")
    (tmp_path / "tests/excluded_test.py").write_text("def test_excluded(): assert False\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests/test_included.py::test_included" in result.stdout
    assert "test_excluded" not in result.stdout


@pytest.mark.parametrize("directory", ["tests", "tests/nested"])
def test_ci_shards_refuse_files_outside_the_collection_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str
) -> None:
    parent = tmp_path / directory
    parent.mkdir(parents=True)
    (parent / "omitted_test.py").write_text("def test_omitted(): assert False\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="naming pattern"):
        ci_shards.main(["--check"])
