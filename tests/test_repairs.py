"""Registered inputs remain fixed while explicit code repairs are recorded."""

import json
from pathlib import Path

import pytest

from tests.test_run import COMMIT
from tests.test_run import runner as runner
from zoon.pilot import clients, run

NEW_COMMIT = "c" * 40


def test_code_change_requires_clean_explicit_repair_and_records_history(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.execute("judge-check")
    changed = runner.root / "src" / "zoon" / "pilot" / "clients.py"
    changed.write_text(changed.read_text() + "\n")
    restarted = run.Run(runner.root, synthetic=True)
    with pytest.raises(clients.StopRun, match="repair"):
        restarted.execute("judge")

    def dirty(root: Path, *args: str) -> str:
        return NEW_COMMIT if args == ("rev-parse", "HEAD") else " M changed"

    monkeypatch.setattr(run, "git", dirty)
    with pytest.raises(clients.StopRun, match="clean work tree"):
        restarted.execute("judge", repair="Restore reply handling")

    def clean(root: Path, *args: str) -> str:
        return NEW_COMMIT if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(run, "git", clean)
    assert "completed" in restarted.execute("judge", repair="Restore reply handling")[0]
    repair = clients.obj(clients.items(restarted.binding()["repairs"])[0])
    assert repair["reason"] == "Restore reply handling"
    assert repair["commit"] == NEW_COMMIT
    assert repair["files"] == ["src/zoon/pilot/clients.py"]
    assert restarted.state("judge-check")["commit"] == COMMIT
    assert restarted.state("judge")["commit"] == NEW_COMMIT
    for step in ("calibrate", "held-out"):
        restarted.execute(step, NEW_COMMIT)
    lines = restarted.execute("report")
    assert any("[stub Repairs]" in line and "Restore reply handling" in line for line in lines)
    assert any(f"judge-check commit = {COMMIT}" in line for line in lines)
    assert any(f"judge commit = {NEW_COMMIT}" in line for line in lines)
    assert any(f"registered commit = {NEW_COMMIT}" in line for line in lines)


@pytest.mark.parametrize(
    "target",
    [
        "prereg/h1.md",
        "prereg/requests/decision-evaluative.json",
        "pilot/rewrites/outside-01.json",
        "prereg/cohorts.json",
        "pilot/task-texts.json",
        "pilot/task-texts.sha256",
        "pilot/rewrites.sha256",
        "data-manifest.json",
        "data-manifest-outside.json",
        "data-manifest-materials.json",
    ],
)
def test_registered_changes_refuse_even_with_repair(runner: run.Run, target: str) -> None:
    runner.execute("judge-check")
    file = runner.root / target
    file.write_bytes(file.read_bytes() + b"\n")
    restarted = run.Run(runner.root, synthetic=True)
    with pytest.raises(clients.StopRun, match=r"registered.*digests changed"):
        restarted.execute("judge", repair="Restore reply handling")


def test_registered_guard_requires_tag_ancestry_unchanged_inputs_and_current_repair(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = [COMMIT]
    ancestor = [True]
    changed = [False]

    def git(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return NEW_COMMIT
        if args == ("rev-parse", "refs/tags/registered^{commit}"):
            return tag[0]
        if args[0] == "merge-base" and not ancestor[0]:
            raise clients.StopRun("not an ancestor")
        if args[0] == "diff":
            return "changed" if changed[0] else ""
        return ""

    monkeypatch.setattr(run, "git", git)
    with pytest.raises(clients.StopRun, match="tag"):
        run.registration_guard(runner.root, "a" * 40)
    ancestor[0] = False
    with pytest.raises(clients.StopRun, match="ancestor"):
        run.registration_guard(runner.root, COMMIT)
    ancestor[0] = True
    changed[0] = True
    with pytest.raises(clients.StopRun, match="registered files"):
        run.registration_guard(runner.root, COMMIT)
    changed[0] = False
    with pytest.raises(clients.StopRun, match="repair"):
        run.registration_guard(runner.root, COMMIT)
    runner.execute("judge-check")
    code = runner.root / "src" / "zoon" / "pilot" / "clients.py"
    code.write_bytes(code.read_bytes() + b"\n")
    repaired = run.Run(runner.root, synthetic=True)
    repaired.execute("judge", repair="Restore reply handling")
    assert run.registration_guard(runner.root, COMMIT, binding=repaired.binding()) == COMMIT
    code.write_bytes(code.read_bytes() + b"\n")
    with pytest.raises(clients.StopRun, match="repair"):
        run.registration_guard(runner.root, COMMIT, binding=repaired.binding())


def test_report_without_repairs_says_none(runner: run.Run) -> None:
    runner.execute("judge-check")
    runner.save("judge", {}, status="failed", reason="interrupted")
    lines = runner.execute("report")
    assert "[stub Repairs] history = none" in lines
    assert f"[stub run] registered commit = {COMMIT}" in lines
    assert "\n## Repairs\n" in (runner.directory / "result.md").read_text()
    saved = json.loads((runner.directory / "judge-check.json").read_text())
    assert saved["registered_files"] and saved["code_files"]
